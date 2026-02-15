"""High-level event simulator backed by a single CUDA kernel.

The algorithm works as follows:

1. Maintain two per-pixel state tensors — an *upper bound* and a *lower bound*
   — initialised from the log-intensity of the first frame ± the contrast
   threshold.
2. For each subsequent frame, compute ``log(pixel)`` and compare against the
   bounds.  If the log-intensity exceeds the upper bound a *positive* event is
   emitted; if it drops below the lower bound a *negative* event is emitted.
3. On event: reset both bounds around the current log-intensity.
   No event: tighten the bounds toward the current value.
"""

import logging
from typing import NamedTuple
from dataclasses import dataclass, field

import torch

from neurosim_cu_esim._backend import evsim_cuda

logger = logging.getLogger(__name__)


class Events(NamedTuple):
    """Container for a batch of events returned by the simulator."""

    x: torch.Tensor
    """Column coordinates (``uint16``)."""
    y: torch.Tensor
    """Row coordinates (``uint16``)."""
    t: torch.Tensor
    """Timestamps in microseconds (``uint64``)."""
    p: torch.Tensor
    """Polarity: 1 = positive, 0 = negative (``uint8``)."""


@dataclass
class EventSimulator:
    """CUDA-accelerated frame-differencing event simulator.

    Parameters
    ----------
    width, height : int
        Sensor resolution.
    contrast_threshold_neg, contrast_threshold_pos : float
        Contrast thresholds in log-intensity units.
    max_events : int | None
        Maximum events per call.  Defaults to ``width * height``.
    device : str | torch.device
        CUDA device to use.
    """

    width: int
    height: int
    contrast_threshold_neg: float = 0.35
    contrast_threshold_pos: float = 0.35
    max_events: int | None = None
    device: str | torch.device = "cuda"

    # ---- internal state ----
    _intensity_state_ub: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )
    _intensity_state_lb: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )
    _event_x_buf: torch.Tensor = field(init=False, repr=False)
    _event_y_buf: torch.Tensor = field(init=False, repr=False)
    _event_t_buf: torch.Tensor = field(init=False, repr=False)
    _event_p_buf: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_events is None:
            self.max_events = self.width * self.height
        self._init_buffers()

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _init_buffers(self) -> None:
        """Allocate persistent event output buffers on the target device."""
        assert self.max_events is not None
        dev = self.device
        self._event_x_buf = torch.empty(self.max_events, dtype=torch.uint16, device=dev)
        self._event_y_buf = torch.empty(self.max_events, dtype=torch.uint16, device=dev)
        self._event_t_buf = torch.empty(self.max_events, dtype=torch.uint64, device=dev)
        self._event_p_buf = torch.empty(self.max_events, dtype=torch.uint8, device=dev)
        logger.debug(
            "Allocated event buffers for %d events on %s", self.max_events, dev
        )

    # ------------------------------------------------------------------
    # Initialisation / reset
    # ------------------------------------------------------------------

    def init(self, first_image: torch.Tensor) -> None:
        """Initialise internal state from the first frame.

        Parameters
        ----------
        first_image : torch.Tensor
            Grayscale ``(H, W)`` tensor with **positive** pixel values.
        """
        first_image = self._prepare_image(first_image)
        log_img = torch.log(first_image)
        self._intensity_state_ub = (log_img + self.contrast_threshold_pos).contiguous()
        self._intensity_state_lb = (log_img - self.contrast_threshold_neg).contiguous()
        logger.debug(
            "Initialised state from image of shape %s", tuple(first_image.shape)
        )

    @property
    def is_initialised(self) -> bool:
        """Whether :meth:`init` has been called."""
        return self._intensity_state_ub is not None

    def reset(self, first_image: torch.Tensor | None = None) -> None:
        """Clear internal state and optionally re-initialise from *first_image*."""
        self._intensity_state_ub = None
        self._intensity_state_lb = None
        if first_image is not None:
            self.init(first_image)

    def forward(self, image: torch.Tensor, timestamp_us: int) -> Events | None:
        """Process a new frame and return generated events.

        If the simulator has not been initialised yet the provided frame is
        used for initialisation and ``None`` is returned.

        Parameters
        ----------
        image : torch.Tensor
            Grayscale ``(H, W)`` tensor with positive pixel values.
        timestamp_us : int
            Timestamp of this frame in **microseconds**.

        Returns
        -------
        Events | None
            A named tuple ``(x, y, t, p)`` of CUDA tensors, or ``None`` if
            there are no events (including the initialisation call).
        """
        if not self.is_initialised:
            self.init(image)
            return None

        image = self._prepare_image(image)

        assert self._intensity_state_ub is not None
        assert self._intensity_state_lb is not None

        x, y, t, p = evsim_cuda(
            image,
            int(timestamp_us),
            self._intensity_state_ub,
            self._intensity_state_lb,
            self._event_x_buf,
            self._event_y_buf,
            self._event_t_buf,
            self._event_p_buf,
            self.contrast_threshold_neg,
            self.contrast_threshold_pos,
        )

        if x.numel() == 0:
            return None

        return Events(x=x, y=y, t=t, p=p)

    # Alias so the object is callable: sim(image, ts)
    __call__ = forward

    def set_contrast_thresholds(
        self,
        neg: float | None = None,
        pos: float | None = None,
    ) -> None:
        """Update one or both contrast thresholds at runtime."""
        if neg is not None:
            self.contrast_threshold_neg = neg
        if pos is not None:
            self.contrast_threshold_pos = pos
        logger.debug(
            "Thresholds updated: neg=%.4f, pos=%.4f",
            self.contrast_threshold_neg,
            self.contrast_threshold_pos,
        )

    @property
    def state(self) -> dict[str, torch.Tensor | float | None]:
        """Snapshot of internal state (useful for debugging)."""
        return {
            "intensity_state_ub": self._intensity_state_ub,
            "intensity_state_lb": self._intensity_state_lb,
            "contrast_threshold_neg": self.contrast_threshold_neg,
            "contrast_threshold_pos": self.contrast_threshold_pos,
        }

    @property
    def buffer_memory_bytes(self) -> int:
        """Total GPU memory used by the pre-allocated event buffers."""
        return sum(
            buf.element_size() * buf.numel()
            for buf in (
                self._event_x_buf,
                self._event_y_buf,
                self._event_t_buf,
                self._event_p_buf,
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Move to device, ensure contiguous 2-D float tensor."""
        if image.dim() == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[-1] == 1:
                image = image.squeeze(-1)
            else:
                raise ValueError(
                    f"Expected single-channel image, got shape {tuple(image.shape)}"
                )
        if image.dim() != 2:
            raise ValueError(f"Expected 2-D (H, W) image, got {image.dim()}-D tensor")
        if not image.is_cuda:
            image = image.to(self.device)
        if not image.is_floating_point():
            image = image.float()
        return image.contiguous()
