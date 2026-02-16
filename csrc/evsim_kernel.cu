// neurosim_cu_esim — CUDA kernel for frame-differencing event generation.
//
// Each thread processes one pixel. It computes the log-intensity of the
// incoming grayscale frame, compares it against per-pixel upper/lower
// bounds, and emits an event when the difference exceeds the configured
// contrast threshold. Warp-level ballot/prefix-sum is used to aggregate
// events so that only one atomic per warp is needed.

#include "utils.h"

#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void evsim_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> new_image,
    const uint64_t  new_time,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> intensity_state_ub,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> intensity_state_lb,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_x_buf,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_y_buf,
    torch::PackedTensorAccessor32<uint64_t, 1, torch::RestrictPtrTraits> event_t_buf,
    torch::PackedTensorAccessor32<uint8_t,  1, torch::RestrictPtrTraits> event_p_buf,
    int32_t* __restrict__ event_count,
    const float contrast_threshold_neg,
    const float contrast_threshold_pos,
    const uint32_t max_events,
    const uint16_t height,
    const uint16_t width
) {
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    bool has_event = false;
    bool pos_event = false;

    if (x < width && y < height) {
        const scalar_t cur_log = log(new_image[y][x]);
        const scalar_t ub      = intensity_state_ub[y][x];
        const scalar_t lb      = intensity_state_lb[y][x];

        pos_event = cur_log > ub;
        const bool neg_event = cur_log < lb;
        has_event = pos_event || neg_event;

        // Update state bounds – tighten toward current value when no event,
        // reset around current value when an event fires.
        if (has_event) {
            intensity_state_ub[y][x] = cur_log + static_cast<scalar_t>(contrast_threshold_pos);
            intensity_state_lb[y][x] = cur_log - static_cast<scalar_t>(contrast_threshold_neg);
        } else {
            intensity_state_ub[y][x] = min(ub, cur_log + static_cast<scalar_t>(contrast_threshold_pos));
            intensity_state_lb[y][x] = max(lb, cur_log - static_cast<scalar_t>(contrast_threshold_neg));
        }
    }

    // ------ Warp-level aggregation to minimize atomicAdd contention ------
    const uint32_t tid     = threadIdx.y * blockDim.x + threadIdx.x;
    const int8_t   lane_id = tid & 31;

    const uint32_t warp_event_mask  = __ballot_sync(FULL_MASK, has_event);
    const int32_t  warp_event_count = __popc(warp_event_mask);

    if (warp_event_count > 0) {
        int32_t warp_base_idx = 0;
        if (lane_id == 0)
            warp_base_idx = atomicAdd(event_count, warp_event_count);

        // Broadcast base index from lane 0 to all lanes
        warp_base_idx = __shfl_sync(FULL_MASK, warp_base_idx, 0);

        if (has_event) {
            const uint32_t lane_mask          = (1u << lane_id) - 1u;
            const uint32_t preceding_mask     = warp_event_mask & lane_mask;
            const int32_t  thread_event_idx   = __popc(preceding_mask);
            const int32_t  global_event_idx   = warp_base_idx + thread_event_idx;

            if (global_event_idx < static_cast<int32_t>(max_events)) {
                event_x_buf[global_event_idx] = static_cast<uint16_t>(x);
                event_y_buf[global_event_idx] = static_cast<uint16_t>(y);
                event_t_buf[global_event_idx] = new_time;
                event_p_buf[global_event_idx] = pos_event ? 1 : 0;
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim(
    const torch::Tensor new_image,
    const uint64_t new_time,
    torch::Tensor intensity_state_ub,
    torch::Tensor intensity_state_lb,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    const float contrast_threshold_neg,
    const float contrast_threshold_pos
) {
    // Validate inputs
    CHECK_CUDA_CONTIGUOUS_FLOAT(new_image);
    CHECK_CUDA_CONTIGUOUS_FLOAT(intensity_state_ub);
    CHECK_CUDA_CONTIGUOUS_FLOAT(intensity_state_lb);
    CHECK_CUDA_CONTIGUOUS(event_x_buf);
    CHECK_CUDA_CONTIGUOUS(event_y_buf);
    CHECK_CUDA_CONTIGUOUS(event_t_buf);
    CHECK_CUDA_CONTIGUOUS(event_p_buf);

    TORCH_CHECK(new_image.dim() == 2,           "new_image must be 2-D (H, W)");
    TORCH_CHECK(intensity_state_ub.dim() == 2,  "intensity_state_ub must be 2-D (H, W)");
    TORCH_CHECK(intensity_state_lb.dim() == 2,  "intensity_state_lb must be 2-D (H, W)");

    const uint16_t height     = static_cast<uint16_t>(new_image.size(0));
    const uint16_t width      = static_cast<uint16_t>(new_image.size(1));
    const uint32_t max_events = static_cast<uint32_t>(event_x_buf.size(0));

    auto event_count = torch::zeros(
        {1}, torch::dtype(torch::kInt32).device(new_image.device()));

    const dim3 threads(32, 32);
    const dim3 blocks(BLOCKS(width, threads.x), BLOCKS(height, threads.y));

    AT_DISPATCH_FLOATING_TYPES(new_image.scalar_type(), "evsim_cuda", ([&] {
        evsim_kernel<scalar_t><<<blocks, threads>>>(
            new_image.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            new_time,
            intensity_state_ub.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            intensity_state_lb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            event_x_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_y_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_t_buf.packed_accessor32<uint64_t, 1, torch::RestrictPtrTraits>(),
            event_p_buf.packed_accessor32<uint8_t,  1, torch::RestrictPtrTraits>(),
            event_count.data_ptr<int32_t>(),
            contrast_threshold_neg,
            contrast_threshold_pos,
            max_events,
            height,
            width
        );
    }));

    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(cuda_err));
    cudaDeviceSynchronize();

    const int32_t num_events =
        std::min(event_count[0].item<int32_t>(),
                 static_cast<int32_t>(max_events));

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
