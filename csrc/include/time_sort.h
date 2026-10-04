#pragma once

// Counting sort of one frame's events on their integer microsecond: the event
// kernel stages packed events (x | y << 16 | p << 32 | (t - prev_time) << 33)
// and counts them per microsecond of the frame interval, then scatter_by_time
// writes them to the output buffers in time order.
//
// Counters: [0] event count, [1 : interval_us + 2] events per microsecond of the interval.
//
// The host functions expect the tensors' GPU to be current (their callers hold a
// CUDAGuard) and run on PyTorch's current stream there.

#include "utils.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Stages an event in the next free slot and counts it in its microsecond: one
// atomic per warp for the slots, one per distinct microsecond in the warp for
// the counts (most multi-mode events share the frame's last microsecond).
__device__ __forceinline__ void append_event(
    uint64_t* __restrict__ stage,
    int32_t* __restrict__ event_count,
    int32_t* __restrict__ time_hist,
    const uint32_t max_events,
    const int32_t x, const int32_t y, const uint8_t p, const uint64_t t_rel
) {
    const auto active = cg::coalesced_threads();
    int32_t first = 0;
    if (active.thread_rank() == 0)
        first = atomicAdd(event_count, static_cast<int32_t>(active.size()));
    const int32_t idx = active.shfl(first, 0) + static_cast<int32_t>(active.thread_rank());
    if (idx >= static_cast<int32_t>(max_events)) return;

    stage[idx] = static_cast<uint64_t>(x)
               | (static_cast<uint64_t>(y) << 16)
               | (static_cast<uint64_t>(p) << 32)
               | (t_rel << 33);
    const auto same_time =
        cg::labeled_partition(cg::coalesced_threads(), static_cast<uint32_t>(t_rel));
    if (same_time.thread_rank() == 0)
        atomicAdd(&time_hist[t_rel], static_cast<int32_t>(same_time.size()));
}

// Zeroed counters for one frame interval (allocated per call).
torch::Tensor time_sort_counters(const torch::Tensor& like, uint64_t interval_us);

// Zeroes the part of reusable counters that one frame interval needs.
void zero_time_counters(torch::Tensor counters, uint64_t interval_us);

torch::Tensor time_sort_stage(const torch::Tensor& like, uint32_t max_events);

// Writes the staged events to the buffers in time order; returns how many there are.
int32_t scatter_by_time(
    torch::Tensor counters,
    torch::Tensor stage,
    uint64_t prev_time,
    uint64_t interval_us,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf
);
