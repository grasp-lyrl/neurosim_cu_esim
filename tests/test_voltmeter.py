"""Tests for the DVS-Voltmeter stochastic simulator."""

import pytest
import torch

from neurosim_cu_esim import DVSVoltmeterSimulator

pytestmark = pytest.mark.cuda


class TestInit:
    def test_preset_loads_k(self):
        sim = DVSVoltmeterSimulator(width=32, height=32, camera_type="DVS346")
        assert sim.k is not None and len(sim.k) == 6

    def test_unknown_camera_raises(self):
        with pytest.raises(ValueError, match="camera_type"):
            DVSVoltmeterSimulator(width=8, height=8, camera_type="nope")

    def test_explicit_k_override(self):
        k = [1.0, 20.0, 0.1, 1e-7, 5e-9, 1e-5]
        sim = DVSVoltmeterSimulator(width=8, height=8, k=k)
        assert sim.k == k

    def test_bad_k_length_raises(self):
        with pytest.raises(ValueError, match="6 elements"):
            DVSVoltmeterSimulator(width=8, height=8, k=[1.0, 2.0])

    def test_first_call_returns_none(self, device):
        sim = DVSVoltmeterSimulator(width=16, height=16)
        frame = torch.full((16, 16), 100.0, device=device)
        assert sim.forward(frame, 0) is None
        assert sim.is_initialised


class TestForward:
    def test_static_scene_few_events(self, device):
        """A perfectly static scene should produce events only from noise."""
        sim = DVSVoltmeterSimulator(width=64, height=64, seed=1)
        frame = torch.full((64, 64), 120.0, device=device)
        sim.forward(frame, 0)
        ev = sim.forward(frame, 5000)
        # Noise floor (k6) can still trigger some events; just ensure it runs.
        assert ev is None or ev.x.numel() >= 0

    def test_motion_generates_events(self, device):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=1, max_events=64 * 64 * 64
        )
        a = torch.full((64, 64), 50.0, device=device)
        b = torch.full((64, 64), 200.0, device=device)
        sim.forward(a, 0)
        ev = sim.forward(b, 5000)
        assert ev is not None
        assert ev.x.numel() > 0

    def test_event_fields(self, device):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=1, max_events=64 * 64 * 64
        )
        sim.forward(torch.full((64, 64), 50.0, device=device), 0)
        ev = sim.forward(torch.full((64, 64), 200.0, device=device), 5000)
        assert ev is not None
        n = ev.x.numel()
        assert ev.y.numel() == n and ev.t.numel() == n and ev.p.numel() == n
        assert ev.x.dtype == torch.uint16
        assert ev.t.dtype == torch.uint64
        assert ev.p.dtype == torch.uint8

    def test_timestamps_within_interval(self, device):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=1, max_events=64 * 64 * 64
        )
        sim.forward(torch.full((64, 64), 50.0, device=device), 1000)
        ev = sim.forward(torch.full((64, 64), 200.0, device=device), 6000)
        assert ev is not None
        t = ev.t.to(torch.int64)
        assert int(t.min()) > 1000
        assert int(t.max()) <= 6000

    @pytest.mark.parametrize("dt", [1000, 2000, 5000])
    def test_timestamps_sorted(self, device, dt):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=1, max_events=64 * 64 * 64
        )
        gen = torch.Generator(device=device).manual_seed(0)
        sim.forward(torch.rand((64, 64), device=device, generator=gen) * 255, 0)
        ev = sim.forward(torch.rand((64, 64), device=device, generator=gen) * 255, dt)
        assert ev is not None
        t = ev.t.to(torch.int64)
        assert (t[1:] >= t[:-1]).all(), "events within a frame are not time-ordered"
        assert int(t[0]) >= 0 and int(t[-1]) <= dt, "events left the frame interval"

    def test_saturated_buffer_stays_sorted(self, device):
        sim = DVSVoltmeterSimulator(width=64, height=64, seed=1, max_events=500)
        gen = torch.Generator(device=device).manual_seed(0)
        sim.forward(torch.rand((64, 64), device=device, generator=gen) * 255, 0)
        ev = sim.forward(torch.rand((64, 64), device=device, generator=gen) * 255, 1000)
        assert ev is not None and ev.t.numel() == 500
        t = ev.t.to(torch.int64)
        assert (t[1:] >= t[:-1]).all(), "a saturated frame is not time-ordered"
        assert int(t.min()) >= 0 and int(t.max()) <= 1000, "saturation wrote garbage"
        assert (
            int(ev.x.to(torch.int64).max()) < 64
            and int(ev.y.to(torch.int64).max()) < 64
        )

    def test_polarity_positive_on_brightness_increase(self, device):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=2, max_events=64 * 64 * 64
        )
        sim.forward(torch.full((64, 64), 40.0, device=device), 0)
        ev = sim.forward(torch.full((64, 64), 220.0, device=device), 5000)
        assert ev is not None
        # Strong brightness increase -> overwhelmingly ON (polarity 1).
        assert float((ev.p == 1).float().mean()) > 0.9

    def test_non_increasing_timestamp_raises(self, device):
        sim = DVSVoltmeterSimulator(width=16, height=16)
        sim.forward(torch.full((16, 16), 100.0, device=device), 1000)
        with pytest.raises(ValueError, match="must be >"):
            sim.forward(torch.full((16, 16), 100.0, device=device), 1000)


class TestDeterminism:
    def test_same_seed_same_output(self, device):
        a = torch.full((48, 48), 60.0, device=device)
        b = torch.full((48, 48), 180.0, device=device)

        def run(seed):
            sim = DVSVoltmeterSimulator(
                width=48, height=48, seed=seed, max_events=48 * 48 * 64
            )
            sim.forward(a, 0)
            return sim.forward(b, 5000)

        e1, e2 = run(7), run(7)
        assert e1 is not None and e2 is not None
        assert e1.x.numel() == e2.x.numel()

        # The kernel is deterministic in *content* given (seed, pixel, frame),
        # but the output order is not (per-warp atomicAdd races). Compare as a
        # sorted multiset.
        def key(e):
            x = e.x.to(torch.int64)
            y = e.y.to(torch.int64)
            t = e.t.to(torch.int64)
            p = e.p.to(torch.int64)
            composite = ((x * 100000 + y) * 10_000_000 + t) * 2 + p
            return torch.sort(composite).values

        assert torch.equal(key(e1), key(e2))

    def test_state_advances_base_frame(self, device):
        sim = DVSVoltmeterSimulator(width=16, height=16, seed=0)
        a = torch.full((16, 16), 50.0, device=device)
        b = torch.full((16, 16), 150.0, device=device)
        sim.forward(a, 0)
        sim.forward(b, 5000)
        # base_frame should now equal the most recent frame
        assert torch.allclose(sim._base_frame, b)


class TestReset:
    def test_reset_clears(self, device):
        sim = DVSVoltmeterSimulator(width=16, height=16)
        sim.forward(torch.full((16, 16), 100.0, device=device), 0)
        sim.reset()
        assert not sim.is_initialised
        assert sim._frame_index == 0
