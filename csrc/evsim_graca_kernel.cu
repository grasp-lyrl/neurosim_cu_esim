// neurosim_cu_esim — CUDA kernel for the continuous-input LPV DVS pixel model.
//
// Based on the physically-realistic large-signal model from R. Graca &
// T. Delbruck (arXiv:2505.07386, 2025), refactored to match the MATLAB
// DVS_model.m continuous-input approach:
//
//   * NO sub-step loop: the kernel executes the difference equations (DF-I)
//     exactly once per call using dt = new_time - prev_time.
//   * NO impulse injection: the photoreceptor input is the continuously
//     computed target voltage Vpr_target = (UT/kappa_fb) * log(I_pd/dark_current).
//   * NO S_L0 state: the kernel does not track previous-frame intensity.
//     Small-signal parameters (gm_fb, gs_fb) are recomputed from the
//     instantaneous I_pd at each call (LPV linearization).
//
// Signal flow per pixel per call:
//   1. Map linear intensity -> photocurrent I_pd (clamped to [dark_current, full_well_saturation_threshold])
//   2. Compute operating-point-dependent filter coefficients (bilinear transform)
//   3. Photoreceptor Zm: 2nd-order DF-I filter on Vpr_target -> Vpr_signal
//   4. Optional shot noise: Zm-filtered PD noise + Zout-filtered PR noise
//   5. Source follower Asf: 1st-order DF-I -> Vsf
//   6. Optional SF shot noise (Zoutsf)
//   7. Change detector: fixed threshold on Vsf, with optional FPT hidden crossings
//
// All per-pixel analog state lives in a packed (NSTATE, H, W) tensor so it
// persists across forward() calls. Variable per-pixel event counts are written
// out with the same warp prefix-sum scheme as the other kernels.

#include "utils.h"
#include <curand_kernel.h>

#define FULL_MASK 0xffffffff

// Per-pixel: at most 1 event per call (no sub-steps).
#define GRACA_MAX_EVENTS_PER_PIXEL 1
// Philox counter budget per pixel per call (>= draws/call).
#define GRACA_DRAWS_PER_FRAME 256

#define Q_CHARGE 1.602176634e-19

// ---- packed per-pixel state layout (index into dim 0 of the state tensor) ----
enum GracaState {
    S_UPR1 = 0,     // Zm signal input history u[n-1]
    S_UPR2,         // Zm signal input history u[n-2]
    S_YPR1,         // Zm signal output history y[n-1]
    S_YPR2,         // Zm signal output history y[n-2]
    S_VSF,          // source-follower total output (signal + noise)
    S_VPRSF,        // SF input history x[n-1] (= previous vpr_total)
    S_VSFC,         // SF clean output y[n-1] (feedback; Isf noise added separately)
    S_VREF,         // change-detector reference level
    S_TSE,          // microseconds since last event (refractory bookkeeping)
    // ---- noise filter state (used only when add_noise != 0) ----
    S_ZMX1, S_ZMX2, S_ZMY1, S_ZMY2,   // Zm PD-noise DF-I state (white in, V out)
    S_ZOX1, S_ZOX2, S_ZOY1, S_ZOY2,   // Zout PR-noise DF-I state
    S_ZSX1, S_ZSY1,                    // Zoutsf SF-noise DF-I state (1st order)
    // ---- stochastic (first-passage-time) event-generation state ----
    S_VSFN,         // noise-only SF output (to isolate vsf_noise for FPT)
    S_VPRSFN,       // noise-only SF input history
    S_MSI,          // calibrated mean-square Vsf-noise increment <(d noise)^2>
    S_IPD_BASE,     // first frame photocurrent for log normalization
    GRACA_NSTATE
};

// Bilinear (Tustin) transform of a 2nd-order analog TF
//   H(s) = (nb1 s + nb0) / (da2 s^2 + da1 s + 1)
// into normalized digital DF-I coefficients b0,b1,b2 (num) and a1,a2 (den, a0=1).
__device__ __forceinline__ void bilinear2(
    double nb1, double nb0, double da2, double da1, double K0,
    double& b0, double& b1, double& b2, double& a1, double& a2
) {
    const double B0 =  nb1 * K0 + nb0;
    const double B1 =  2.0 * nb0;
    const double B2 = -nb1 * K0 + nb0;
    const double A0 =  da2 * K0 * K0 + da1 * K0 + 1.0;
    const double A1 = -2.0 * da2 * K0 * K0 + 2.0;
    const double A2 =  da2 * K0 * K0 - da1 * K0 + 1.0;
    b0 = B0 / A0; b1 = B1 / A0; b2 = B2 / A0;
    a1 = A1 / A0; a2 = A2 / A0;
}

// Bilinear transform of a 1st-order analog TF  H(s) = nb0 / (da1 s + 1).
__device__ __forceinline__ void bilinear1(
    double nb0, double da1, double K0,
    double& b0, double& b1, double& a1
) {
    const double A0 = da1 * K0 + 1.0;
    b0 = nb0 / A0;
    b1 = nb0 / A0;
    a1 = (1.0 - da1 * K0) / A0;
}

template <typename scalar_t>
__global__ void evsim_graca_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> new_image,
    const uint64_t new_time,
    const uint64_t prev_time,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> state,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_x_buf,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_y_buf,
    torch::PackedTensorAccessor32<uint64_t, 1, torch::RestrictPtrTraits> event_t_buf,
    torch::PackedTensorAccessor32<uint8_t,  1, torch::RestrictPtrTraits> event_p_buf,
    int32_t* __restrict__ event_count,
    // model parameters (SI units)
    const double Cpd, const double Cfb, const double Cpr, const double Csf,
    const double Ipr, const double Isf,
    const double kappa_fb, const double kappa_sf, const double VA, const double UT,
    // intensity -> photocurrent mapping:
    //   Ipd = clamp(L, dark_current, full_well_saturation_threshold)
    const double full_well_saturation_threshold, const double dark_current,
    // change detector
    const double thr_on, const double thr_off, const double refractory_us,
    const int add_noise,
    const int stochastic_events,
    const unsigned long long seed,
    const unsigned long long frame_index,
    const int init_steady_state,
    const uint32_t max_events,
    const uint16_t height,
    const uint16_t width
) {
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t n = 0;                                  // events stored by this pixel
    float   loc_t[GRACA_MAX_EVENTS_PER_PIXEL];
    uint8_t loc_p[GRACA_MAX_EVENTS_PER_PIXEL];

    if (x < width && y < height) {
        // ---- time step ----
        const double dt_us_d = (double)(new_time - prev_time);    // us
        const double Ts      = dt_us_d * 1e-6;                    // s
        const double K0      = 2.0 / Ts;                          // bilinear 2/Ts

        // ---- clamp photocurrent ----
        const double I_pd = fmin(fmax((double)new_image[y][x],
                                      dark_current), full_well_saturation_threshold);

        // ---- small-signal parameters at the operating point ----
        // These are recomputed from the instantaneous I_pd each call (LPV).
        const double gs_fb     = I_pd / UT;
        const double gm_fb     = kappa_fb * I_pd / UT;
        const double gm_amp_n  = kappa_fb * Ipr / UT;
        const double Rout      = VA / (2.0 * Ipr);
        const double Aloop     = gm_amp_n * Rout * kappa_fb;

        // Factored denominator time constants (matching DVS_model.m)
        const double tau_pd = (1.0/gs_fb * (Cpd + (1.0 + gm_amp_n*Rout)*Cfb)) / (Aloop + 1.0);
        const double tau_pr = (Rout * (Cpr + (1.0 - kappa_fb)*Cfb)) / (Aloop + 1.0);
        const double da2    = tau_pd * tau_pr;
        const double da1    = tau_pd + tau_pr;

        // Zm: H(s) = (N1_Zm*s + 1) / (da2*s^2 + da1*s + 1)   [DC gain = 1]
        const double N1_Zm = -(Cfb / gm_amp_n);
        double zb0, zb1, zb2, za1, za2;
        bilinear2(N1_Zm, 1.0, da2, da1, K0, zb0, zb1, zb2, za1, za2);

        // Asf: H(s) = kappa_sf / (tau_sf*s + 1)
        const double gs_sf  = Isf / UT;
        const double tau_sf = Csf / gs_sf;
        double sb0, sb1, sa1;
        bilinear1(kappa_sf, tau_sf, K0, sb0, sb1, sa1);

        // ---- noise filter coefficients (only if needed) ----
        double ob0=0,ob1=0,ob2=0,oa1=0,oa2=0;      // Zout
        double nsb0=0,nsb1=0,nsa1=0;                // Zoutsf
        double std_ipd=0, std_ipr=0, std_isf=0;
        curandStatePhilox4_32_10_t rng;
        if (add_noise) {
            const double Zmdc   = (1.0/gm_fb) * (Aloop / (Aloop + 1.0));
            const double Zoutdc = Rout / (Aloop + 1.0);

            // Zout: H(s) = (N1_Zout*s + 1) / (da2*s^2 + da1*s + 1)  [same denom]
            const double N1_Zout = (Cpd + Cfb) / gs_fb;
            bilinear2(N1_Zout, 1.0, da2, da1, K0, ob0, ob1, ob2, oa1, oa2);

            // Zoutsf: H(s) = (1/gs_sf) / (tau_sf*s + 1)
            bilinear1(1.0/gs_sf, tau_sf, K0, nsb0, nsb1, nsa1);

            // Shot noise std-dev (pre-scaled by DC transimpedance gains)
            std_ipd = Zmdc   * sqrt(2.0*Q_CHARGE*I_pd/Ts);
            std_ipr = Zoutdc * sqrt(2.0*Q_CHARGE*Ipr /Ts);
            std_isf =          sqrt(2.0*Q_CHARGE*Isf /Ts);

            const unsigned long long pix = (unsigned long long)y * width + x;
            curand_init(seed, pix, frame_index * GRACA_DRAWS_PER_FRAME, &rng);
        }

        // ---- load persistent state ----
        float u_pr1 = state[S_UPR1][y][x];
        float u_pr2 = state[S_UPR2][y][x];
        float y_pr1 = state[S_YPR1][y][x];
        float y_pr2 = state[S_YPR2][y][x];
        float Vsf   = state[S_VSF][y][x];
        float vprsf = state[S_VPRSF][y][x];
        float vsfc  = state[S_VSFC][y][x];
        float Vref  = state[S_VREF][y][x];
        float tse   = state[S_TSE][y][x];

        float zmx1=0,zmx2=0,zmy1=0,zmy2=0;
        float zox1=0,zox2=0,zoy1=0,zoy2=0;
        float zsx1=0,zsy1=0;
        if (add_noise) {
            zmx1=state[S_ZMX1][y][x]; zmx2=state[S_ZMX2][y][x];
            zmy1=state[S_ZMY1][y][x]; zmy2=state[S_ZMY2][y][x];
            zox1=state[S_ZOX1][y][x]; zox2=state[S_ZOX2][y][x];
            zoy1=state[S_ZOY1][y][x]; zoy2=state[S_ZOY2][y][x];
            zsx1=state[S_ZSX1][y][x]; zsy1=state[S_ZSY1][y][x];
        }

        // ---- FPT (stochastic event) state ----
        const int use_fpt = (stochastic_events && add_noise);
        float vsfn = 0.0f, vprsfn = 0.0f;
        float msi_stored = 0.0f;
        float vsf_noise_prev = 0.0f;
        if (use_fpt) {
            vsfn       = state[S_VSFN][y][x];
            vprsfn     = state[S_VPRSFN][y][x];
            msi_stored = state[S_MSI][y][x];
            vsf_noise_prev = vsfn + zsy1;  // last call's final Vsf noise
        }
        const float vsf_prev = Vsf;        // previous call's Vsf (for event detection)

        // ---- Initialize Ipd_base if it's 0 (first frame) ----
        if (state[S_IPD_BASE][y][x] == 0.0f) {
            state[S_IPD_BASE][y][x] = (float)I_pd;
        }
        const double ipd_base = (double)state[S_IPD_BASE][y][x];

        // ---- compute continuous input Vpr_target ----
        const float Vpr_target = (float)((UT / kappa_fb) * log(I_pd / ipd_base));

        // ---- Initialize steady state if it's the first frame ----
        if (frame_index == 0 && init_steady_state) {
            u_pr1 = u_pr2 = Vpr_target;
            y_pr1 = y_pr2 = Vpr_target;
            
            float vsfc_steady = Vpr_target * (float)kappa_sf;
            vprsf = Vpr_target;
            vsfc  = vsfc_steady;
            Vsf   = vsfc_steady;
            Vref  = vsfc_steady;
            
            if (use_fpt) {
                msi_stored = 0.0f; // FPT variance accumulator starts at 0
            }
        }

        // ---- photoreceptor Zm signal filter (DF-I, 2nd order) ----
        // The filter has DC gain = 1, so at steady state vpr_sig = Vpr_target.
        float vpr_sig = (float)(zb0)*Vpr_target + (float)(zb1)*u_pr1 + (float)(zb2)*u_pr2
                        - (float)(za1)*y_pr1 - (float)(za2)*y_pr2;
        u_pr2 = u_pr1; u_pr1 = Vpr_target;
        y_pr2 = y_pr1; y_pr1 = vpr_sig;

        // ---- optional shot noise on Vpr ----
        float vpr_total = vpr_sig;
        if (add_noise) {
            // PD noise: white current -> Zm filter (same coefficients, input pre-scaled by Zmdc)
            const float w_ipd = std_ipd * curand_normal(&rng);
            float zmy = (float)(zb0)*w_ipd + (float)(zb1)*zmx1 + (float)(zb2)*zmx2
                        - (float)(za1)*zmy1 - (float)(za2)*zmy2;
            zmx2=zmx1; zmx1=w_ipd; zmy2=zmy1; zmy1=zmy;

            // PR noise: white current -> Zout filter (same denominator, different numerator)
            const float w_ipr = std_ipr * curand_normal(&rng);
            float zoy = (float)(ob0)*w_ipr + (float)(ob1)*zox1 + (float)(ob2)*zox2
                        - (float)(oa1)*zoy1 - (float)(oa2)*zoy2;
            zox2=zox1; zox1=w_ipr; zoy2=zoy1; zoy1=zoy;

            vpr_total = vpr_sig + zmy + zoy;
        }

        // ---- source follower Asf (DF-I, 1st order) ----
        // The Asf recurrence feeds back vsfc (clean output); Isf shot noise zsy
        // is an additive output (Zoutsf) that does NOT re-enter the SF feedback.
        const float vsfc_new = (float)(sb0)*vpr_total + (float)(sb1)*vprsf
                               - (float)(sa1)*vsfc;
        vprsf = vpr_total;
        vsfc = vsfc_new;
        float vsf_new = vsfc;

        if (add_noise) {
            const float w_isf = std_isf * curand_normal(&rng);
            float zsy = (float)(nsb0)*w_isf + (float)(nsb1)*zsx1 - (float)(nsa1)*zsy1;
            zsx1=w_isf; zsy1=zsy;
            vsf_new = vsfc + zsy;

            // Isolate the Vsf-noise component for FPT hidden-crossing calibration:
            // run a parallel noise-only SF on vpr_noise (= vpr_total - vpr_sig).
            if (use_fpt) {
                const float vpr_noise = vpr_total - vpr_sig;     // = zmy + zoy
                const float vsfn_new = (float)(sb0)*vpr_noise + (float)(sb1)*vprsfn
                                       - (float)(sa1)*vsfn;
                vprsfn = vpr_noise; vsfn = vsfn_new;
                const float vsf_noise = vsfn + zsy;
                const float dvn = vsf_noise - vsf_noise_prev;
                // Single-sample MSI estimate (stored for next call's FPT formula)
                msi_stored = (float)((double)dvn * (double)dvn);
                vsf_noise_prev = vsf_noise;
            }
        }
        Vsf = vsf_new;

        // ---- change detector with refractory ----
        // Single threshold check per call.  ON when Vsf-Vref >= +thr_on,
        // OFF when <= -thr_off.  Optional FPT hidden crossing between
        // vsf_prev (last call) and Vsf (this call).
        tse += (float)dt_us_d;
        if (tse >= (float)refractory_us) {
            const float d  = Vsf      - Vref;     // detector value this step
            const float dp = vsf_prev - Vref;     // detector value previous step
            int   pol = -1;                        // -1 none, 1 ON, 0 OFF

            // Convert log-contrast threshold to Vsf voltage threshold
            const float v_thr_on  = (float)(thr_on  * (kappa_sf * UT / kappa_fb));
            const float v_thr_off = (float)(thr_off * (kappa_sf * UT / kappa_fb));

            if (d >= v_thr_on)         { pol = 1; }
            else if (d <= -v_thr_off)  { pol = 0; }
            else if (use_fpt && msi_stored > 0.0f) {
                // Two independent Bernoulli trials (upper / lower barrier)
                if (dp < v_thr_on) {
                    const float P = expf(-2.0f * (v_thr_on - dp)
                                         * (v_thr_on - d) / msi_stored);
                    if (curand_uniform(&rng) < P) { pol = 1; }
                }
                if (pol < 0 && dp > -v_thr_off) {
                    const float P = expf(-2.0f * (dp + v_thr_off)
                                         * (d + v_thr_off) / msi_stored);
                    if (curand_uniform(&rng) < P) { pol = 0; }
                }
            }
            if (pol >= 0) {
                if (n < GRACA_MAX_EVENTS_PER_PIXEL) {
                    loc_t[n] = (float)dt_us_d; loc_p[n] = (uint8_t)pol; n++;
                }
                Vref = Vsf; tse = 0.0f;
            }
        }

        // ---- persist state ----
        state[S_UPR1][y][x]  = u_pr1;
        state[S_UPR2][y][x]  = u_pr2;
        state[S_YPR1][y][x]  = y_pr1;
        state[S_YPR2][y][x]  = y_pr2;
        state[S_VSF][y][x]   = Vsf;
        state[S_VPRSF][y][x] = vprsf;
        state[S_VSFC][y][x]  = vsfc;
        state[S_VREF][y][x]  = Vref;
        state[S_TSE][y][x]   = tse;
        if (add_noise) {
            state[S_ZMX1][y][x]=zmx1; state[S_ZMX2][y][x]=zmx2;
            state[S_ZMY1][y][x]=zmy1; state[S_ZMY2][y][x]=zmy2;
            state[S_ZOX1][y][x]=zox1; state[S_ZOX2][y][x]=zox2;
            state[S_ZOY1][y][x]=zoy1; state[S_ZOY2][y][x]=zoy2;
            state[S_ZSX1][y][x]=zsx1; state[S_ZSY1][y][x]=zsy1;
        }
        if (use_fpt) {
            state[S_VSFN][y][x]   = vsfn;
            state[S_VPRSFN][y][x] = vprsfn;
            state[S_MSI][y][x]    = msi_stored;
        }
    }

    // ------ warp prefix-sum -> contiguous output slots ------
    const uint32_t tid     = threadIdx.y * blockDim.x + threadIdx.x;
    const int8_t   lane_id = tid & 31;
    int32_t scan = n;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int32_t v = __shfl_up_sync(FULL_MASK, scan, offset);
        if (lane_id >= offset) scan += v;
    }
    const int32_t warp_total = __shfl_sync(FULL_MASK, scan, 31);

    if (warp_total > 0) {
        int32_t warp_base = 0;
        if (lane_id == 0) warp_base = atomicAdd(event_count, warp_total);
        warp_base = __shfl_sync(FULL_MASK, warp_base, 0);

        if (n > 0) {
            const int32_t my_base = warp_base + (scan - n);
            for (int32_t k = 0; k < n; ++k) {
                const int32_t idx = my_base + k;
                if (idx < (int32_t)max_events) {
                    event_x_buf[idx] = (uint16_t)x;
                    event_y_buf[idx] = (uint16_t)y;
                    event_t_buf[idx] = prev_time + (uint64_t)llroundf(loc_t[k]);
                    event_p_buf[idx] = loc_p[k];
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim_graca(
    const torch::Tensor new_image,
    const uint64_t new_time,
    const uint64_t prev_time,
    torch::Tensor state,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    const double Cpd, const double Cfb, const double Cpr, const double Csf,
    const double Ipr, const double Isf,
    const double kappa_fb, const double kappa_sf, const double VA, const double UT,
    const double full_well_saturation_threshold, const double dark_current,
    const double thr_on, const double thr_off, const double refractory_us,
    const int64_t add_noise,
    const int64_t stochastic_events,
    const uint64_t seed,
    const uint64_t frame_index,
    const int64_t init_steady_state
) {
    CHECK_CUDA_CONTIGUOUS_FLOAT(new_image);
    CHECK_CUDA_CONTIGUOUS_FLOAT(state);
    CHECK_CUDA_CONTIGUOUS(event_x_buf);
    CHECK_CUDA_CONTIGUOUS(event_y_buf);
    CHECK_CUDA_CONTIGUOUS(event_t_buf);
    CHECK_CUDA_CONTIGUOUS(event_p_buf);

    TORCH_CHECK(new_image.dim() == 2, "new_image must be 2-D (H, W)");
    TORCH_CHECK(state.dim() == 3 && state.size(0) == GRACA_NSTATE,
                "state must be (", (int)GRACA_NSTATE, ", H, W)");
    TORCH_CHECK(new_time > prev_time, "new_time must be > prev_time");

    const uint16_t height     = static_cast<uint16_t>(new_image.size(0));
    const uint16_t width      = static_cast<uint16_t>(new_image.size(1));
    const uint32_t max_events = static_cast<uint32_t>(event_x_buf.size(0));

    auto event_count = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(new_image.device()));

    const dim3 threads(32, 8);
    const dim3 blocks(BLOCKS(width, threads.x), BLOCKS(height, threads.y));

    AT_DISPATCH_FLOATING_TYPES(new_image.scalar_type(), "evsim_graca_cuda", ([&] {
        evsim_graca_kernel<scalar_t><<<blocks, threads>>>(
            new_image.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            new_time, prev_time,
            state.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            event_x_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_y_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_t_buf.packed_accessor32<uint64_t, 1, torch::RestrictPtrTraits>(),
            event_p_buf.packed_accessor32<uint8_t,  1, torch::RestrictPtrTraits>(),
            event_count.data_ptr<int32_t>(),
            Cpd, Cfb, Cpr, Csf, Ipr, Isf,
            kappa_fb, kappa_sf, VA, UT,
            full_well_saturation_threshold, dark_current,
            thr_on, thr_off, refractory_us,
            static_cast<int>(add_noise),
            static_cast<int>(stochastic_events),
            static_cast<unsigned long long>(seed),
            static_cast<unsigned long long>(frame_index),
            static_cast<int>(init_steady_state),
            max_events, height, width
        );
    }));

    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(cuda_err));
    cudaDeviceSynchronize();

    const int32_t num_events =
        std::min(event_count[0].item<int32_t>(), static_cast<int32_t>(max_events));

    if (num_events == 0) {
        auto opts_u16 = torch::dtype(torch::kUInt16).device(new_image.device());
        auto opts_u64 = torch::dtype(torch::kUInt64).device(new_image.device());
        auto opts_u8  = torch::dtype(torch::kUInt8).device(new_image.device());
        return std::make_tuple(
            torch::empty({0}, opts_u16), torch::empty({0}, opts_u16),
            torch::empty({0}, opts_u64), torch::empty({0}, opts_u8));
    }

    return std::make_tuple(
        event_x_buf.slice(0, 0, num_events),
        event_y_buf.slice(0, 0, num_events),
        event_t_buf.slice(0, 0, num_events),
        event_p_buf.slice(0, 0, num_events));
}
