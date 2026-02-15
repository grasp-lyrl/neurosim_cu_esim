"""Tests for the high-level EventSimulator class."""

import pytest
import torch

from neurosim_cu_esim import EventSimulator

pytestmark = pytest.mark.cuda


class TestInit:
    def test_default_construction(self):
        sim = EventSimulator(width=320, height=240)
        assert not sim.is_initialised
        assert sim.max_events == 320 * 240

    def test_custom_max_events(self):
        sim = EventSimulator(width=320, height=240, max_events=1024)
        assert sim.max_events == 1024

    def test_custom_thresholds(self):
        sim = EventSimulator(
            width=64,
            height=64,
            contrast_threshold_neg=0.2,
            contrast_threshold_pos=0.5,
        )
        assert sim.contrast_threshold_neg == pytest.approx(0.2)
        assert sim.contrast_threshold_pos == pytest.approx(0.5)

    def test_init_marks_initialised(self, sim, constant_frame):
        assert not sim.is_initialised
        sim.init(constant_frame)
        assert sim.is_initialised

    def test_reset_clears_state(self, sim, constant_frame):
        sim.init(constant_frame)
        sim.reset()
        assert not sim.is_initialised

    def test_reset_with_image(self, sim, constant_frame):
        sim.reset(constant_frame)
        assert sim.is_initialised


# =====================================================================
# Event generation basics
# =====================================================================


class TestForward:
    def test_first_call_returns_none(self, sim, constant_frame):
        """First call should initialise state and return None."""
        result = sim.forward(constant_frame, timestamp_us=0)
        assert result is None
        assert sim.is_initialised

    def test_identical_frames_no_events(self, sim, constant_frame):
        """Feeding the same frame twice should produce no events."""
        sim.init(constant_frame)
        events = sim.forward(constant_frame, timestamp_us=1000)
        assert events is None

    def test_large_step_generates_events(self, sim, device):
        """A large brightness jump should generate events."""
        h, w = 480, 640
        dark = torch.full((h, w), 0.1, device=device)
        bright = torch.full((h, w), 1.0, device=device)

        sim.init(dark)
        events = sim.forward(bright, timestamp_us=1000)
        assert events is not None
        assert events.x.numel() > 0

    def test_event_fields_shape(self, sim, device):
        """All event tensors must be 1-D and the same length."""
        h, w = 480, 640
        dark = torch.full((h, w), 0.1, device=device)
        bright = torch.full((h, w), 1.0, device=device)

        sim.init(dark)
        events = sim.forward(bright, timestamp_us=500)
        assert events is not None
        n = events.x.numel()
        assert events.y.numel() == n
        assert events.t.numel() == n
        assert events.p.numel() == n

    def test_event_dtypes(self, sim, device):
        """Verify the data types of the returned tensors."""
        h, w = 480, 640
        dark = torch.full((h, w), 0.1, device=device)
        bright = torch.full((h, w), 1.0, device=device)

        sim.init(dark)
        events = sim.forward(bright, timestamp_us=1)
        assert events is not None
        assert events.x.dtype == torch.uint16
        assert events.y.dtype == torch.uint16
        assert events.t.dtype == torch.uint64
        assert events.p.dtype == torch.uint8

    def test_event_coordinates_in_bounds(self, sim, device, resolution):
        """x and y coordinates must lie within the sensor resolution."""
        w, h = resolution
        dark = torch.full((h, w), 0.1, device=device)
        bright = torch.full((h, w), 1.0, device=device)

        sim.init(dark)
        events = sim.forward(bright, timestamp_us=1)
        assert events is not None
        assert events.x.to(torch.int32).max().item() < w
        assert events.y.to(torch.int32).max().item() < h

    def test_all_pixels_fire_on_uniform_jump(self, device):
        """With a big enough jump every pixel should fire."""
        h, w = 64, 64
        sim = EventSimulator(width=w, height=h, device="cuda")
        dark = torch.full((h, w), 0.01, device=device)
        bright = torch.full((h, w), 10.0, device=device)

        sim.init(dark)
        events = sim.forward(bright, timestamp_us=1)
        assert events is not None
        assert events.x.numel() == h * w

    def test_callable_alias(self, sim, constant_frame, device):
        """``sim(image, ts)`` should work like ``sim.forward(...)``."""
        sim.init(constant_frame)
        bright = torch.full_like(constant_frame, 1.0)
        events = sim(bright, 1000)
        # may or may not have events depending on threshold; just
        # verify it doesn't crash
        assert events is None or events.x.numel() >= 0


# =====================================================================
# Polarity correctness
# =====================================================================


class TestPolarity:
    def test_positive_events_on_brightness_increase(self, device):
        h, w = 64, 64
        sim = EventSimulator(width=w, height=h, device="cuda")
        dark = torch.full((h, w), 0.01, device=device)
        bright = torch.full((h, w), 10.0, device=device)

        sim.init(dark)
        events = sim.forward(bright, timestamp_us=1)
        assert events is not None
        # All events should be positive (polarity == 1)
        assert (events.p == 1).all()

    def test_negative_events_on_brightness_decrease(self, device):
        h, w = 64, 64
        sim = EventSimulator(width=w, height=h, device="cuda")
        bright = torch.full((h, w), 10.0, device=device)
        dark = torch.full((h, w), 0.01, device=device)

        sim.init(bright)
        events = sim.forward(dark, timestamp_us=1)
        assert events is not None
        # All events should be negative (polarity == 0)
        assert (events.p == 0).all()


# =====================================================================
# Timestamps
# =====================================================================


class TestTimestamps:
    def test_timestamps_match_input(self, device):
        h, w = 64, 64
        sim = EventSimulator(width=w, height=h, device="cuda")
        dark = torch.full((h, w), 0.01, device=device)
        bright = torch.full((h, w), 10.0, device=device)

        sim.init(dark)
        events = sim.forward(bright, timestamp_us=42_000)
        assert events is not None
        assert (events.t == 42_000).all()


# =====================================================================
# Threshold behaviour
# =====================================================================


class TestThresholds:
    def test_higher_threshold_fewer_events(self, device):
        h, w = 128, 128
        dark = torch.full((h, w), 0.3, device=device)
        bright = torch.full((h, w), 0.6, device=device)

        sim_low = EventSimulator(
            width=w,
            height=h,
            contrast_threshold_neg=0.1,
            contrast_threshold_pos=0.1,
        )
        sim_high = EventSimulator(
            width=w,
            height=h,
            contrast_threshold_neg=0.9,
            contrast_threshold_pos=0.9,
        )

        sim_low.init(dark)
        sim_high.init(dark)

        ev_low = sim_low.forward(bright, 1)
        ev_high = sim_high.forward(bright, 1)

        n_low = ev_low.x.numel() if ev_low else 0
        n_high = ev_high.x.numel() if ev_high else 0
        assert n_low >= n_high

    def test_set_contrast_thresholds(self, sim):
        sim.set_contrast_thresholds(neg=0.1, pos=0.8)
        assert sim.contrast_threshold_neg == pytest.approx(0.1)
        assert sim.contrast_threshold_pos == pytest.approx(0.8)

    def test_set_partial_thresholds(self, sim):
        original_neg = sim.contrast_threshold_neg
        sim.set_contrast_thresholds(pos=0.9)
        assert sim.contrast_threshold_neg == pytest.approx(original_neg)
        assert sim.contrast_threshold_pos == pytest.approx(0.9)


# =====================================================================
# Multiple frames / temporal sequence
# =====================================================================


class TestSequence:
    def test_gradual_ramp_events(self, device):
        """Gradually increasing brightness should eventually generate events."""
        h, w = 64, 64
        sim = EventSimulator(
            width=w,
            height=h,
            device="cuda",
            contrast_threshold_neg=0.3,
            contrast_threshold_pos=0.3,
        )
        base = 0.5
        sim.init(torch.full((h, w), base, device=device))

        total_events = 0
        for i in range(1, 20):
            frame = torch.full((h, w), base + i * 0.1, device=device)
            events = sim.forward(frame, timestamp_us=i * 1000)
            if events is not None:
                total_events += events.x.numel()

        assert total_events > 0

    def test_state_persists_across_frames(self, device):
        """After an event, bounds should reset so the same frame won't fire again."""
        h, w = 32, 32
        sim = EventSimulator(width=w, height=h, device="cuda")
        dark = torch.full((h, w), 0.01, device=device)
        bright = torch.full((h, w), 10.0, device=device)

        sim.init(dark)
        ev1 = sim.forward(bright, 1000)
        assert ev1 is not None

        # Same bright frame again — no change ⇒ no events
        ev2 = sim.forward(bright, 2000)
        assert ev2 is None


# =====================================================================
# Edge cases / robustness
# =====================================================================


class TestEdgeCases:
    def test_single_pixel(self, device):
        sim = EventSimulator(width=1, height=1, device="cuda")
        sim.init(torch.tensor([[0.1]], device=device))
        events = sim.forward(torch.tensor([[10.0]], device=device), 1)
        assert events is not None
        assert events.x.numel() == 1

    def test_non_square(self, device):
        sim = EventSimulator(width=17, height=5, device="cuda")
        dark = torch.full((5, 17), 0.01, device=device)
        bright = torch.full((5, 17), 10.0, device=device)
        sim.init(dark)
        events = sim.forward(bright, 1)
        assert events is not None
        assert events.x.numel() == 5 * 17

    def test_3d_single_channel_input(self, device):
        """A (1, H, W) or (H, W, 1) tensor should be accepted."""
        sim = EventSimulator(width=32, height=32, device="cuda")
        frame_chw = torch.full((1, 32, 32), 0.5, device=device)
        sim.init(frame_chw)  # should not raise

    def test_cpu_input_auto_transferred(self):
        """Images on CPU should be automatically moved to CUDA."""
        sim = EventSimulator(width=16, height=16, device="cuda")
        cpu_frame = torch.full((16, 16), 0.5)  # on CPU
        sim.init(cpu_frame)
        assert sim.is_initialised

    def test_max_events_cap(self, device):
        """When max_events is tiny, output should be capped."""
        h, w = 64, 64
        sim = EventSimulator(width=w, height=h, max_events=10, device="cuda")
        dark = torch.full((h, w), 0.01, device=device)
        bright = torch.full((h, w), 10.0, device=device)
        sim.init(dark)
        events = sim.forward(bright, 1)
        assert events is not None
        assert events.x.numel() <= 10


# =====================================================================
# Diagnostics / properties
# =====================================================================


class TestDiagnostics:
    def test_buffer_memory_bytes(self, sim):
        mem = sim.buffer_memory_bytes
        assert mem > 0

    def test_state_dict(self, sim, constant_frame):
        sim.init(constant_frame)
        s = sim.state
        assert "intensity_state_ub" in s
        assert s["intensity_state_ub"] is not None
        assert "contrast_threshold_neg" in s
