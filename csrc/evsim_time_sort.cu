// neurosim_cu_esim — time sort shared by the multi-event and Voltmeter kernels.

#include "time_sort.h"
#include <cub/block/block_scan.cuh>

// One block scans TIME_SORT_THREADS * 2 bins per pass: a frame interval up to
// 2047 us (>= ~500 Hz) is one pass.
#define TIME_SORT_THREADS 1024
#define TIME_SORT_SCATTER_BLOCKS 256

// In-place exclusive prefix sum of the per-microsecond event counts, giving each
// microsecond's first output slot.
__global__ void time_offsets_kernel(
    int32_t* __restrict__ time_hist,
    const int64_t bins
) {
    using BlockScan = cub::BlockScan<int32_t, TIME_SORT_THREADS>;
    __shared__ typename BlockScan::TempStorage scan_storage;

    int32_t carry = 0;
    for (int64_t base = 0; base < bins; base += 2 * TIME_SORT_THREADS) {
        const int64_t i = base + 2 * threadIdx.x;
        int32_t counts[2] = {i < bins ? time_hist[i] : 0, i + 1 < bins ? time_hist[i + 1] : 0};
        int32_t pass_total;
        BlockScan(scan_storage).ExclusiveSum(counts, counts, pass_total);
        if (i < bins) time_hist[i] = carry + counts[0];
        if (i + 1 < bins) time_hist[i + 1] = carry + counts[1];
        carry += pass_total;
        __syncthreads();
    }
}

// Moves each staged event to the next free slot of its microsecond.
__global__ void time_scatter_kernel(
    const uint64_t* __restrict__ stage,
    const int32_t* __restrict__ event_count,
    int32_t* __restrict__ time_offsets,
    const uint32_t max_events,
    const uint64_t prev_time,
    uint16_t* __restrict__ event_x,
    uint16_t* __restrict__ event_y,
    uint64_t* __restrict__ event_t,
    uint8_t*  __restrict__ event_p
) {
    const int32_t n = min(*event_count, static_cast<int32_t>(max_events));
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const uint64_t r     = stage[i];
        const uint64_t t_rel = r >> 33;
        const auto same_time =
            cg::labeled_partition(cg::coalesced_threads(), static_cast<uint32_t>(t_rel));
        int32_t first = 0;
        if (same_time.thread_rank() == 0)
            first = atomicAdd(&time_offsets[t_rel], static_cast<int32_t>(same_time.size()));
        const int32_t dst = same_time.shfl(first, 0) + static_cast<int32_t>(same_time.thread_rank());
        event_x[dst] = static_cast<uint16_t>(r);
        event_y[dst] = static_cast<uint16_t>(r >> 16);
        event_p[dst] = static_cast<uint8_t>((r >> 32) & 1);
        event_t[dst] = prev_time + t_rel;
    }
}

torch::Tensor time_sort_counters(const torch::Tensor& like, const uint64_t interval_us) {
    TORCH_CHECK(interval_us < (1ull << 31), "frame interval must be under 2^31 us");
    return torch::zeros(
        {2 + static_cast<int64_t>(interval_us)},
        torch::dtype(torch::kInt32).device(like.device()));
}

void zero_time_counters(torch::Tensor counters, const uint64_t interval_us) {
    TORCH_CHECK(interval_us < (1ull << 31), "frame interval must be under 2^31 us");
    const int64_t size = 2 + static_cast<int64_t>(interval_us);
    TORCH_CHECK(counters.numel() >= size,
                "time counters hold ", counters.numel(), " ints, the interval needs ", size);
    cudaMemsetAsync(counters.data_ptr<int32_t>(), 0, size * sizeof(int32_t));
}

torch::Tensor time_sort_stage(const torch::Tensor& like, const uint32_t max_events) {
    return torch::empty(
        {static_cast<int64_t>(max_events)},
        torch::dtype(torch::kInt64).device(like.device()));
}

int32_t scatter_by_time(
    torch::Tensor counters,
    torch::Tensor stage,
    const uint64_t prev_time,
    const uint64_t interval_us,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf
) {
    const uint32_t max_events  = static_cast<uint32_t>(event_x_buf.size(0));
    int32_t*       event_count = counters.data_ptr<int32_t>();

    time_offsets_kernel<<<1, TIME_SORT_THREADS>>>(
        event_count + 1, static_cast<int64_t>(interval_us) + 1);
    time_scatter_kernel<<<TIME_SORT_SCATTER_BLOCKS, 256>>>(
        reinterpret_cast<const uint64_t*>(stage.data_ptr<int64_t>()),
        event_count,
        event_count + 1,
        max_events,
        prev_time,
        event_x_buf.data_ptr<uint16_t>(),
        event_y_buf.data_ptr<uint16_t>(),
        event_t_buf.data_ptr<uint64_t>(),
        event_p_buf.data_ptr<uint8_t>()
    );
    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(cuda_err));

    // Blocking on the default stream, so this also waits for the event kernels.
    int32_t count = 0;
    cuda_err = cudaMemcpy(&count, event_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA event kernels failed: ", cudaGetErrorString(cuda_err));
    return std::min(count, static_cast<int32_t>(max_events));
}
