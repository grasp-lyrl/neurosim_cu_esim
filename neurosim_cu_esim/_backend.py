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


def evsim_multi_cuda(
    new_image: torch.Tensor,
    new_time: int,
    prev_time: int,
    intensity_state_ub: torch.Tensor,
    intensity_state_lb: torch.Tensor,
    event_x_buf: torch.Tensor,
    event_y_buf: torch.Tensor,
    event_t_buf: torch.Tensor,
    event_p_buf: torch.Tensor,
    contrast_threshold_neg: float = 0.35,
    contrast_threshold_pos: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-event variant of :func:`evsim_cuda`.

    Emits *multiple* events per pixel when the log-intensity change spans more
    than one contrast threshold, instead of capping at one event per pixel.
    The ``n`` events generated for a pixel are assigned timestamps spread
    equally across the inter-frame interval, so that ``t_k = prev_time +
    (new_time - prev_time) * (k + 1) / n`` for ``k = 0 .. n-1`` — the last
    event lands exactly on ``new_time``.

    Parameters
    ----------
    prev_time : int
        Timestamp (microseconds) of the *previous* frame; defines the lower end
        of the interval that emitted events are spread across.  Must satisfy
        ``prev_time <= new_time``.

    All other parameters match :func:`evsim_cuda`.
    """
    return _neurosim_cu_esim_ext.evsim_multi(
        new_image,
        new_time,
        prev_time,
        intensity_state_ub,
        intensity_state_lb,
        event_x_buf,
        event_y_buf,
        event_t_buf,
        event_p_buf,
        contrast_threshold_neg,
        contrast_threshold_pos,
    )


def evsim_voltmeter_cuda(
    new_image: torch.Tensor,
    new_time: int,
    prev_time: int,
    base_frame: torch.Tensor,
    delta_vd_res: torch.Tensor,
    event_x_buf: torch.Tensor,
    event_y_buf: torch.Tensor,
    event_t_buf: torch.Tensor,
    event_p_buf: torch.Tensor,
    time_counters: torch.Tensor,
    time_stage: torch.Tensor,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    k5: float,
    k6: float,
    seed: int,
    frame_index: int,
    exact: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the DVS-Voltmeter stochastic event kernel for one frame step.

    Models each pixel's sensor voltage as a Brownian motion with drift derived
    from the two frames (paper Eq. 10/11) and emits events at threshold
    crossings, with timestamps sampled from the Inverse-Gaussian / Levy
    first-passage-time distribution.

    Parameters
    ----------
    new_image : torch.Tensor
        Grayscale ``(H, W)`` frame on CUDA, **linear intensity** (e.g. 0-255,
        the scale the ``k`` params are calibrated to). *Not* log-intensity.
    new_time, prev_time : int
        Current / previous frame timestamps in microseconds (``new_time >
        prev_time``).
    base_frame : torch.Tensor
        Per-pixel previous intensity ``L0`` state ``(H, W)``; updated in place.
    delta_vd_res : torch.Tensor
        Per-pixel residual voltage state ``(H, W)``; updated in place.
    time_counters : torch.Tensor
        Reusable ``int32`` scratch for the time sort, at least
        ``new_time - prev_time + 2`` long.
    time_stage : torch.Tensor
        Reusable ``int64`` scratch for the time sort, as long as the event buffers.
    k1..k6 : float
        DVS-Voltmeter model parameters (camera-specific calibration).
    seed, frame_index : int
        Philox RNG seed and per-frame counter offset (for reproducibility and
        fresh randomness across frames).
    exact : bool
        ``True`` (default): the exact sampler, which keeps the voltage itself in
        ``delta_vd_res`` and solves the SDE exactly, independent of the frame
        rate. ``False``: the reference's sampler (Algorithm 1).

    Returns
    -------
    tuple of ``(x, y, t, p)`` slices of the pre-allocated buffers.
    """
    return _neurosim_cu_esim_ext.evsim_voltmeter(
        new_image,
        new_time,
        prev_time,
        base_frame,
        delta_vd_res,
        event_x_buf,
        event_y_buf,
        event_t_buf,
        event_p_buf,
        time_counters,
        time_stage,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        seed,
        frame_index,
        exact,
    )
