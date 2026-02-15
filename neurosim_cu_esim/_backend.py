"""Low-level wrapper around the compiled CUDA extension."""

import _neurosim_cu_esim_ext  # type: ignore[import-not-found]
import torch


def evsim_cuda(
    new_image: torch.Tensor,
    new_time: int,
    intensity_state_ub: torch.Tensor,
    intensity_state_lb: torch.Tensor,
    event_x_buf: torch.Tensor,
    event_y_buf: torch.Tensor,
    event_t_buf: torch.Tensor,
    event_p_buf: torch.Tensor,
    contrast_threshold_neg: float = 0.35,
    contrast_threshold_pos: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the CUDA kernel that produces events from a single grayscale frame.

    Parameters
    ----------
    new_image : torch.Tensor
        Grayscale frame of shape ``(H, W)`` on a CUDA device.  Values should
        be strictly positive (the kernel computes ``log``).
    new_time : int
        Timestamp in microseconds associated with this frame.
    intensity_state_ub, intensity_state_lb : torch.Tensor
        Per-pixel upper/lower bound state tensors of shape ``(H, W)`` on CUDA.
    event_x_buf, event_y_buf : torch.Tensor
        Pre-allocated ``uint16`` buffers for event *x*/*y* coordinates.
    event_t_buf : torch.Tensor
        Pre-allocated ``uint64`` buffer for event timestamps.
    event_p_buf : torch.Tensor
        Pre-allocated ``uint8`` buffer for event polarities (1 = positive).
    contrast_threshold_neg, contrast_threshold_pos : float
        Thresholds (in log-intensity units) that a pixel must exceed to
        generate a negative/positive event.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(x, y, t, p)`` — slices of the pre-allocated buffers trimmed to
        the actual number of events produced.
    """
    return _neurosim_cu_esim_ext.evsim(
        new_image,
        new_time,
        intensity_state_ub,
        intensity_state_lb,
        event_x_buf,
        event_y_buf,
        event_t_buf,
        event_p_buf,
        contrast_threshold_neg,
        contrast_threshold_pos,
    )
