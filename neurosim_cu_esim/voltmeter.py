"""DVS-Voltmeter stochastic event simulator (ECCV 2022).

A GPU port of Lin et al., *DVS-Voltmeter: Stochastic Process-based Event
Simulator for Dynamic Vision Sensors* (https://github.com/Lynn0306/DVS-Voltmeter).

Unlike :class:`~neurosim_cu_esim.simulator.EventSimulator` (deterministic
log-intensity threshold crossing), this model treats each pixel's sensor
voltage as a **Brownian motion with drift** whose parameters are derived from
two consecutive frames, and emits events at threshold crossings.  Because
first-passage times of a drifted Brownian motion are Inverse-Gaussian (Levy
when the drift is zero), event timestamps within a frame interval are sampled
*stochastically* — capturing realistic sensor noise rather than equal spacing.

Important: this model consumes **linear intensity** (e.g. 8-bit ``0-255``), not
log-intensity, because the ``k1..k6`` parameters are calibrated to that scale.
"""

import logging
from typing import NamedTuple
from dataclasses import dataclass, field

import torch

from neurosim_cu_esim._backend import evsim_voltmeter_cuda

logger = logging.getLogger(__name__)


# Camera presets from the reference config.py (src/config.py).
CAMERA_PRESETS: dict[str, list[float]] = {
    "DVS346": [0.00018 * 29250, 20.0, 0.0001, 1e-7, 5e-9, 0.00001],
    "DVS240": [0.000094 * 47065, 23.0, 0.0002, 1e-7, 5e-8, 0.00001],
}


class Events(NamedTuple):
    """Container for a batch of events returned by the simulator."""

    x: torch.Tensor
    """Column coordinates (``uint16``)."""
    y: torch.Tensor
    """Row coordinates (``uint16``)."""
    t: torch.Tensor
    """Timestamps in microseconds (``uint64``)."""
    p: torch.Tensor
    """Polarity: 1 = ON (brightness increase), 0 = OFF (``uint8``)."""


@dataclass
class DVSVoltmeterSimulator:
    """CUDA-accelerated DVS-Voltmeter stochastic event simulator.

    Parameters
    ----------
    width, height : int
        Sensor resolution.
    camera_type : str
        Preset name for the ``k1..k6`` parameters (``"DVS346"`` or ``"DVS240"``).
        Ignored if ``k`` is given explicitly.
    k : list[float] | None
        Explicit ``[k1, k2, k3, k4, k5, k6]`` override.
    sampler : str
        How the voltage process is sampled between frames.

        * ``"exact"`` (default): solves the model's SDE exactly. The voltage
          itself is carried across frames; each step draws its exact Gaussian
          increment and tests for a threshold crossing with the Brownian-bridge
          probability, so ON and OFF crossings come from the same physics, and
          the output does not depend on the frame rate. The signal drift is
          integrated as ``k1 * ln((L1 + k2) / (L0 + k2))``, so a brightness step
          gives the same events however many frames it is split across. It
          draws its random numbers from counter-based Philox, one block per step
          with no per-pixel state, and runs faster than the reference on the
          benchmark's moving scene (~1.5% on an RTX 4090, ~7% on a 4070 Laptop).
        * ``"reference"``: Algorithm 1 of the paper, matching the upstream
          code's statistics; use it to compare against upstream. It picks a
          polarity, draws one threshold's first-passage time, and when that lands
          after the frame, moves the residual in a straight line toward the
          threshold and redraws next frame. This throws the diffusion away, so
          the noise shrinks as the frame rate grows (at 1 kHz the timing jitter
          is 5-20x below the model's own), and crossings against the drift are
          timed about 2x too early.
    leak_scale : float
        Multiplier on the leakage *drift* terms ``k4`` (thermal) and ``k5``
        (brightness-proportional parasitic photocurrent), which together cause
        background ON events even where the scene is static.  ``1.0`` (default)
        is the faithful calibrated model; lower it to suppress the background ON
        "flashing" on bright/overexposed regions (``0.0`` = signal-only, no
        leakage events).  Does not touch ``k6`` (the noise variance floor).
    randomize_phase : bool
        If ``True``, initialise each pixel's residual voltage to a random
        sub-threshold value instead of 0.  The reference inits all pixels to 0,
        so on a static scene they share the same leakage phase and — because the
        first-passage variance is tiny — fire in lock-step, producing unphysical
        full-frame "flashes" every ~1/leakage-rate seconds.  Randomising the
        phase spreads the background events uniformly in time (a realistic
        desynchronised sparkle), as a real sensor's pixels would be.  Strongly
        recommended for realistic stationary backgrounds; default ``False`` to
        match the reference.
    input_normalized : bool
        If ``True``, frames are assumed to be in ``[0, 1]`` and are multiplied
        by 255 internally (the ``k`` params are calibrated to the 0-255 scale).
        Lets the same ``[0, 1]`` tensor be fed to either ``EventSimulator`` or
        ``DVSVoltmeterSimulator`` without rescaling at the call site. Default
        ``False`` (expects 0-255 directly).
    max_events : int | None
        Cap on events per call.  A frame step can emit many events per pixel, so
        this defaults to ``width * height * 16``; events beyond it are dropped
        (with a warning).
    seed : int
        Base RNG seed (Philox); combined with an internal per-frame counter for
        reproducible-yet-fresh randomness.
    device : str | torch.device
        CUDA device to use.
    """

    width: int
    height: int
    camera_type: str = "DVS346"
    k: list[float] | None = None
    sampler: str = "exact"
    leak_scale: float = 1.0
    randomize_phase: bool = False
    input_normalized: bool = False
    max_events: int | None = None
    seed: int = 0
    device: str | torch.device = "cuda"

    # ---- internal state ----
    _base_frame: torch.Tensor | None = field(default=None, init=False, repr=False)
    _delta_vd_res: torch.Tensor | None = field(default=None, init=False, repr=False)
    _event_x_buf: torch.Tensor = field(init=False, repr=False)
    _event_y_buf: torch.Tensor = field(init=False, repr=False)
    _event_t_buf: torch.Tensor = field(init=False, repr=False)
    _event_p_buf: torch.Tensor = field(init=False, repr=False)
    _time_counters: torch.Tensor = field(init=False, repr=False)
    _time_stage: torch.Tensor = field(init=False, repr=False)
    _prev_time: int | None = field(default=None, init=False, repr=False)
    _frame_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k is None:
            if self.camera_type not in CAMERA_PRESETS:
                raise ValueError(
                    f"Unknown camera_type {self.camera_type!r}; "
                    f"choose from {sorted(CAMERA_PRESETS)} or pass k=[k1..k6]"
                )
            self.k = list(CAMERA_PRESETS[self.camera_type])
        if len(self.k) != 6:
            raise ValueError(f"k must have 6 elements, got {len(self.k)}")
        if self.sampler not in ("reference", "exact"):
            raise ValueError(
                f"Unknown sampler {self.sampler!r}; choose 'exact' or 'reference'"
            )
        if self.max_events is None:
            self.max_events = self.width * self.height * 16
        self._init_buffers()

    # ------------------------------------------------------------------
    # Buffer / state management
    # ------------------------------------------------------------------

    def _init_buffers(self) -> None:
        assert self.max_events is not None
        dev = self.device
        self._event_x_buf = torch.empty(self.max_events, dtype=torch.uint16, device=dev)
        self._event_y_buf = torch.empty(self.max_events, dtype=torch.uint16, device=dev)
        self._event_t_buf = torch.empty(self.max_events, dtype=torch.uint64, device=dev)
        self._event_p_buf = torch.empty(self.max_events, dtype=torch.uint8, device=dev)
        self._time_counters = torch.empty(0, dtype=torch.int32, device=dev)
        self._time_stage = torch.empty(self.max_events, dtype=torch.int64, device=dev)

    def init(self, first_image: torch.Tensor) -> None:
        """Initialise per-pixel state from the first frame (linear intensity)."""
        first_image = self._prepare_image(first_image)
        self._base_frame = first_image.clone().contiguous()
        if self.randomize_phase:
            # Random sub-threshold residual per pixel (threshold = 1) so pixels
            # start at different points in their leakage cycle -> no phase-locked
            # background flashing. Seeded for reproducibility.
            gen = torch.Generator(device=first_image.device).manual_seed(int(self.seed))
            self._delta_vd_res = torch.rand(
                first_image.shape,
                generator=gen,
                device=first_image.device,
                dtype=first_image.dtype,
            ).contiguous()
        else:
            self._delta_vd_res = torch.zeros_like(first_image).contiguous()

    @property
    def is_initialised(self) -> bool:
        return self._base_frame is not None

    def reset(self, first_image: torch.Tensor | None = None) -> None:
        """Clear state and optionally re-initialise from *first_image*."""
        self._base_frame = None
        self._delta_vd_res = None
        self._prev_time = None
        self._frame_index = 0
        if first_image is not None:
            self.init(first_image)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor, timestamp_us: int) -> Events | None:
        """Process a new frame (linear intensity) and return generated events.

        The first call initialises state and returns ``None``.
        """
        if not self.is_initialised:
            self.init(image)
        if self._prev_time is None:
            self._prev_time = int(timestamp_us)
            return None

        image = self._prepare_image(image)
        ts = int(timestamp_us)
        prev = self._prev_time
        if ts <= prev:
            raise ValueError(
                f"timestamp_us ({ts}) must be > previous timestamp ({prev})"
            )

        assert self._base_frame is not None
        assert self._delta_vd_res is not None
        k1, k2, k3, k4, k5, k6 = self.k  # type: ignore[misc]
        # Scale down the leakage drift terms to reduce background ON activity.
        k4 = k4 * self.leak_scale
        k5 = k5 * self.leak_scale
        # the event count and one counter per microsecond of the interval
        if self._time_counters.numel() < ts - prev + 2:
            self._time_counters = torch.empty(
                ts - prev + 2, dtype=torch.int32, device=self.device
            )

        x, y, t, p = evsim_voltmeter_cuda(
            image,
            ts,
            prev,
            self._base_frame,
            self._delta_vd_res,
            self._event_x_buf,
            self._event_y_buf,
            self._event_t_buf,
            self._event_p_buf,
            self._time_counters,
            self._time_stage,
            float(k1),
            float(k2),
            float(k3),
            float(k4),
            float(k5),
            float(k6),
            int(self.seed),
            int(self._frame_index),
            self.sampler == "exact",
        )

        self._prev_time = ts
        self._frame_index += 1

        if x.numel() == 0:
            return None

        if x.numel() >= self.max_events:
            logger.warning(
                "Event buffer saturated at max_events=%d; some events were "
                "dropped. Increase max_events.",
                self.max_events,
            )

        return Events(x=x, y=y, t=t, p=p)

    __call__ = forward

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Move to device, ensure contiguous 2-D float tensor (linear intensity)."""
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
        # The Voltmeter kernel is float32-only; downcast doubles too.
        if image.dtype != torch.float32:
            image = image.float()
        # k-params are calibrated to 0-255; scale up if caller passes [0, 1].
        if self.input_normalized:
            image = image * 255.0
        return image.contiguous()
