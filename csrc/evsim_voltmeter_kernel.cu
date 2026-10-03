// neurosim_cu_esim — CUDA kernel for the DVS-Voltmeter stochastic event model.
//
// Reference: Lin et al., "DVS-Voltmeter: Stochastic Process-based Event
// Simulator for Dynamic Vision Sensors", ECCV 2022.
//   https://github.com/Lynn0306/DVS-Voltmeter
//
// Each thread owns one pixel. From the previous frame L0 and the new frame L1
// it derives the drift mu and variance-rate sigma^2 of a Brownian motion with
// drift (paper Eq. 10/11), then samples threshold-crossing events across the
// inter-frame interval. First-passage times of drifted Brownian motion are
// Inverse Gaussian (Levy when drift == 0), so event timestamps are sampled
// stochastically rather than equally spaced.
//
// Differences from the reference (for speed):
//   * iterative per-thread loop instead of tensor recursion
//   * counter-based Philox RNG, no stored per-pixel RNG state
//   * relative-time arithmetic in float (avoids large-magnitude precision loss)
//   * numerically stable p_on (no exp overflow), so float32 + --use_fast_math
//   * events are staged as they are sampled, one atomic per warp for their slots
//   * events come out time-ordered, via a counting sort on the integer microsecond

#include "time_sort.h"
#include <curand_kernel.h>

// Guard in case the CUDA <math.h> in use does not expose these.
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// Hard per-pixel cap on events per frame; a frame step rarely produces more than
// a few events per pixel.
#define VOLT_MAX_EVENTS_PER_PIXEL 16
// Hard loop-iteration cap (defends against degenerate tiny-dt floods).
#define VOLT_MAX_ITERS 256
// Philox counter budget per pixel per frame (>= worst-case draws/iter * iters).
#define VOLT_DRAWS_PER_FRAME 1024
// Tuned for 1 kHz frames. For longer frame intervals raise VOLT_MAX_EVENTS_PER_PIXEL (a pixel's events
// past it are dropped silently); intervals over 2047 us cost the time sort's prefix sum extra passes.

// ---- precision helpers (templated so fp64/fp16 can be added later) --------
__device__ __forceinline__ float  volt_erfinv(float x)  { return erfinvf(x); }
__device__ __forceinline__ double volt_erfinv(double x) { return erfinv(x); }

template <typename T>
__device__ __forceinline__ T volt_uniform(curandStatePhilox4_32_10_t* s);
template <>
__device__ __forceinline__ float volt_uniform<float>(curandStatePhilox4_32_10_t* s) {
    return curand_uniform(s);   // (0, 1]
}
template <>
__device__ __forceinline__ double volt_uniform<double>(curandStatePhilox4_32_10_t* s) {
    return curand_uniform_double(s);
}

template <typename T>
__device__ __forceinline__ T volt_normal(curandStatePhilox4_32_10_t* s);
template <>
__device__ __forceinline__ float volt_normal<float>(curandStatePhilox4_32_10_t* s) {
    return curand_normal(s);
}
template <>
__device__ __forceinline__ double volt_normal<double>(curandStatePhilox4_32_10_t* s) {
    return curand_normal_double(s);
}

// First-passage time of dX = c*dt + sigma*dW to level ep > 0.
//   c == 0 -> Levy ;  c != 0 -> Inverse Gaussian (Michael-Schucany-Haas).
template <typename T>
__device__ __forceinline__ T volt_sample_first_passage(
    T ep, T c, T sigma, curandStatePhilox4_32_10_t* state
) {
    const T s2 = sigma * sigma;

    if (c == static_cast<T>(0)) {
        const T scale = (ep / sigma) * (ep / sigma);   // (ep/sigma)^2
        // reference uses U in [0,1); curand_uniform is (0,1] -> use (1 - u)
        const T u = static_cast<T>(1) - volt_uniform<T>(state);
        const T e = volt_erfinv(static_cast<T>(1) - u);
        return scale / (e * e);
    }

    const T mean = ep / c;                       // < 0 when c < 0
    const T lam  = (ep / sigma) * (ep / sigma);  // shape lambda

    T X;
    if (c > static_cast<T>(0)) {
        X = volt_normal<T>(state);
    } else {
        // Truncated normal on (-inf, x_max], x_max = -sqrt(-4*ep*c/s2).
        const T x_max = -sqrt(-static_cast<T>(4) * ep * c / s2);
        const T pmax  = static_cast<T>(0.5) *
                        (static_cast<T>(1) + erf(x_max * static_cast<T>(M_SQRT1_2)));
        const T uni = volt_uniform<T>(state);
        T v = static_cast<T>(2) * (pmax * uni) - static_cast<T>(1);
        v = fmin(fmax(v, static_cast<T>(-0.999999)), static_cast<T>(0.999999));
        X = static_cast<T>(M_SQRT2) * volt_erfinv(v);
        if (X > x_max) X = x_max;
    }

    const T Y = mean * X * X;
    T Z = static_cast<T>(4) * lam * Y + Y * Y;
    if (Z < static_cast<T>(0)) Z = static_cast<T>(0);
    const T Xig = mean + (mean / (static_cast<T>(2) * lam)) * (Y - sqrt(Z));

    const T U = volt_uniform<T>(state);
    return (U > mean / (mean + Xig)) ? (mean * mean / Xig) : Xig;
}

template <typename scalar_t>
__global__ void evsim_voltmeter_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> new_image,
    const uint64_t  new_time,
    const uint64_t  prev_time,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> base_frame,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> delta_vd_res,
    uint64_t* __restrict__ stage,
    int32_t* __restrict__ counters,
    const float k1, const float k2, const float k3,
    const float k4, const float k5, const float k6,
    const unsigned long long seed,
    const unsigned long long frame_index,
    const uint32_t max_events,
    const uint16_t height,
    const uint16_t width
) {
    const int32_t  x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t  y     = blockIdx.y * blockDim.y + threadIdx.y;
    const uint64_t dt_us = new_time - prev_time;

    if (x < width && y < height) {
        const float L0 = base_frame[y][x];
        const float L1 = new_image[y][x];
        const float dL   = L1 - L0;
        const float Lavg = (L1 + L0) * 0.5f;
        const float dt   = static_cast<float>(dt_us);

        const float Dr  = 1.0f / (Lavg + k2);
        const float mu  = k1 * (dL / dt) * Dr + k4 + k5 * Lavg;
        // NB: paper Eq.(11) is labelled a "variance", but the reference passes
        // it into the `sigma` slot of event_generation (and squares it for
        // sigma^2). So this quantity is the diffusion *std* sigma, not sigma^2.
        const float sigma = k3 * sqrtf(Lavg) * Dr + k6;
        const float s2    = sigma * sigma;

        curandStatePhilox4_32_10_t state;
        const unsigned long long pix = static_cast<unsigned long long>(y) * width + x;
        curand_init(seed, pix, frame_index * VOLT_DRAWS_PER_FRAME, &state);

        float res         = delta_vd_res[y][x];
        float start_rel   = 0.0f;
        int32_t n         = 0;
        const float theta = 1.0f;

        #pragma unroll 1
        for (int iter = 0; iter < VOLT_MAX_ITERS; ++iter) {
            const float ep_on  = theta - res;
            const float ep_off = theta + res;

            // Numerically stable two-boundary "on first" probability.
            float p_on;
            if (mu == 0.0f) {
                p_on = 0.5f;
            } else {
                const float a = 2.0f * mu * ep_on  / s2;
                const float b = 2.0f * mu * ep_off / s2;
                if (mu > 0.0f) {
                    p_on = (1.0f - expf(-b)) / (1.0f - expf(-(a + b)));
                } else {
                    const float eab = expf(a + b);
                    const float ea  = expf(a);
                    p_on = (eab - ea) / (eab - 1.0f);
                }
            }
            if (isnan(p_on)) p_on = 1.0f;
            p_on = fminf(fmaxf(p_on, 0.0f), 1.0f);

            const float u  = curand_uniform(&state);
            const bool  on = (u <= p_on);
            const float ep = on ? ep_on : ep_off;
            const float c  = on ? mu : -mu;

            const float dts = volt_sample_first_passage(ep, c, sigma, &state);
            // Degenerate draw (NaN/Inf/<=0): stop without poisoning the residual.
            if (!isfinite(dts) || dts <= 0.0f) break;
            const float t_hit_rel = start_rel + dts;

            if (t_hit_rel < dt) {
                append_event(stage, counters, counters + 1, max_events, x, y, on ? 1 : 0,
                             min(static_cast<uint64_t>(llroundf(t_hit_rel)), dt_us));
                start_rel = t_hit_rel;
                res = 0.0f;
                if (++n >= VOLT_MAX_EVENTS_PER_PIXEL) break;
            } else {
                const float sign = on ? 1.0f : -1.0f;
                res = res + sign * ep * (dt - start_rel) / dts;
                break;
            }
        }

        // Persist state for the next frame.
        delta_vd_res[y][x] = res;
        base_frame[y][x]   = L1;
    }

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim_voltmeter(
    const torch::Tensor new_image,
    const uint64_t new_time,
    const uint64_t prev_time,
    torch::Tensor base_frame,
    torch::Tensor delta_vd_res,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    torch::Tensor time_counters,
    torch::Tensor time_stage,
    const double k1, const double k2, const double k3,
    const double k4, const double k5, const double k6,
    const uint64_t seed,
    const uint64_t frame_index
) {
    CHECK_CUDA_CONTIGUOUS_FLOAT(new_image);
    CHECK_CUDA_CONTIGUOUS_FLOAT(base_frame);
    CHECK_CUDA_CONTIGUOUS_FLOAT(delta_vd_res);
    CHECK_CUDA_CONTIGUOUS(event_x_buf);
    CHECK_CUDA_CONTIGUOUS(event_y_buf);
    CHECK_CUDA_CONTIGUOUS(event_t_buf);
    CHECK_CUDA_CONTIGUOUS(event_p_buf);
    CHECK_CUDA_CONTIGUOUS(time_counters);
    CHECK_CUDA_CONTIGUOUS(time_stage);

    TORCH_CHECK(new_image.dim() == 2,    "new_image must be 2-D (H, W)");
    TORCH_CHECK(base_frame.dim() == 2,   "base_frame must be 2-D (H, W)");
    TORCH_CHECK(delta_vd_res.dim() == 2, "delta_vd_res must be 2-D (H, W)");
    TORCH_CHECK(new_time > prev_time,    "new_time must be > prev_time");

    const uint16_t height     = static_cast<uint16_t>(new_image.size(0));
    const uint16_t width      = static_cast<uint16_t>(new_image.size(1));
    const uint32_t max_events = static_cast<uint32_t>(event_x_buf.size(0));
    TORCH_CHECK(time_stage.numel() >= max_events, "time_stage must hold max_events events");
    zero_time_counters(time_counters, new_time - prev_time);

    // NB: the templated kernel supports float64 too, but it is register-heavy;
    // a double launch at 1024 threads exceeds the register budget, so fp32 is
    // the supported/used precision (the Python simulator feeds float32).
    const dim3 threads(32, 32);
    const dim3 blocks(BLOCKS(width, threads.x), BLOCKS(height, threads.y));

    AT_DISPATCH_FLOATING_TYPES(new_image.scalar_type(), "evsim_voltmeter_cuda", ([&] {
        evsim_voltmeter_kernel<scalar_t><<<blocks, threads>>>(
            new_image.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            new_time,
            prev_time,
            base_frame.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            delta_vd_res.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            reinterpret_cast<uint64_t*>(time_stage.data_ptr<int64_t>()),
            time_counters.data_ptr<int32_t>(),
            static_cast<float>(k1), static_cast<float>(k2), static_cast<float>(k3),
            static_cast<float>(k4), static_cast<float>(k5), static_cast<float>(k6),
            static_cast<unsigned long long>(seed),
            static_cast<unsigned long long>(frame_index),
            max_events,
            height,
            width
        );
    }));

    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(cuda_err));

    const int32_t num_events = scatter_by_time(
        time_counters, time_stage, prev_time, new_time - prev_time, event_x_buf, event_y_buf, event_t_buf, event_p_buf);

    if (num_events == 0) {
        auto opts_u16 = torch::dtype(torch::kUInt16).device(new_image.device());
        auto opts_u64 = torch::dtype(torch::kUInt64).device(new_image.device());
        auto opts_u8  = torch::dtype(torch::kUInt8).device(new_image.device());
        return std::make_tuple(
            torch::empty({0}, opts_u16),
            torch::empty({0}, opts_u16),
            torch::empty({0}, opts_u64),
            torch::empty({0}, opts_u8)
        );
    }

    return std::make_tuple(
        event_x_buf.slice(0, 0, num_events),
        event_y_buf.slice(0, 0, num_events),
        event_t_buf.slice(0, 0, num_events),
        event_p_buf.slice(0, 0, num_events)
    );
}
