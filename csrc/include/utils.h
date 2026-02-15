#pragma once

#include <torch/extension.h>

// ---- Grid/block sizing ----
#define BLOCKS(N, T) (((N) + (T) - 1) / (T))

// ---- Tensor validation macros ----
#define CHECK_CUDA(x) \
    TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_IS_FLOATING(x)                                          \
    TORCH_CHECK(                                                      \
        (x).scalar_type() == at::ScalarType::Float  ||                \
        (x).scalar_type() == at::ScalarType::Half   ||                \
        (x).scalar_type() == at::ScalarType::Double,                  \
        #x " must be a floating-point tensor (float, half, or double)")

// Combined convenience checks
#define CHECK_CUDA_CONTIGUOUS(x) \
    do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (0)

#define CHECK_CUDA_CONTIGUOUS_FLOAT(x) \
    do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_IS_FLOATING(x); } while (0)

// ---- Forward declaration of the main dispatch function ----
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim(
    const torch::Tensor new_image,
    uint64_t new_time,
    torch::Tensor intensity_state_ub,
    torch::Tensor intensity_state_lb,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    float contrast_threshold_neg,
    float contrast_threshold_pos
);
