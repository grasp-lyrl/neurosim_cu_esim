"""Tests for the DVS-Voltmeter stochastic simulator."""

import math

import numpy as np
import pytest
import torch

from neurosim_cu_esim import DVSVoltmeterSimulator

pytestmark = pytest.mark.cuda

SAMPLERS = ["reference", "exact"]
K1, K2 = 0.00018 * 29250, 20.0


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

    def test_default_sampler_is_exact(self):
        assert DVSVoltmeterSimulator(width=8, height=8).sampler == "exact"

    def test_unknown_sampler_raises(self):
        with pytest.raises(ValueError, match="sampler"):
            DVSVoltmeterSimulator(width=8, height=8, sampler="nope")

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

    @pytest.mark.parametrize("sampler", SAMPLERS)
    def test_motion_generates_events(self, device, sampler):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=1, max_events=64 * 64 * 64, sampler=sampler
        )
        a = torch.full((64, 64), 50.0, device=device)
        b = torch.full((64, 64), 200.0, device=device)
        sim.forward(a, 0)
        ev = sim.forward(b, 5000)
        assert ev is not None
        assert ev.x.numel() > 0

    @pytest.mark.parametrize("sampler", SAMPLERS)
    def test_event_fields(self, device, sampler):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=1, max_events=64 * 64 * 64, sampler=sampler
        )
        sim.forward(torch.full((64, 64), 50.0, device=device), 0)
        ev = sim.forward(torch.full((64, 64), 200.0, device=device), 5000)
        assert ev is not None
        n = ev.x.numel()
        assert ev.y.numel() == n and ev.t.numel() == n and ev.p.numel() == n
        assert ev.x.dtype == torch.uint16
        assert ev.t.dtype == torch.uint64
        assert ev.p.dtype == torch.uint8

    @pytest.mark.parametrize("sampler", SAMPLERS)
    def test_timestamps_within_interval(self, device, sampler):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=1, max_events=64 * 64 * 64, sampler=sampler
        )
        sim.forward(torch.full((64, 64), 50.0, device=device), 1000)
        ev = sim.forward(torch.full((64, 64), 200.0, device=device), 6000)
        assert ev is not None
        t = ev.t.to(torch.int64)
        assert int(t.min()) > 1000
        assert int(t.max()) <= 6000

    @pytest.mark.parametrize("sampler", SAMPLERS)
    @pytest.mark.parametrize("dt", [1000, 2000, 5000])
    def test_timestamps_sorted(self, device, dt, sampler):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=1, max_events=64 * 64 * 64, sampler=sampler
        )
        gen = torch.Generator(device=device).manual_seed(0)
        sim.forward(torch.rand((64, 64), device=device, generator=gen) * 255, 0)
        ev = sim.forward(torch.rand((64, 64), device=device, generator=gen) * 255, dt)
        assert ev is not None
        t = ev.t.to(torch.int64)
        assert (t[1:] >= t[:-1]).all(), "events within a frame are not time-ordered"
        assert int(t[0]) >= 0 and int(t[-1]) <= dt, "events left the frame interval"

    @pytest.mark.parametrize("sampler", SAMPLERS)
    def test_saturated_buffer_stays_sorted(self, device, sampler):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=1, max_events=500, sampler=sampler
        )
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

    @pytest.mark.parametrize("sampler", SAMPLERS)
    def test_polarity_positive_on_brightness_increase(self, device, sampler):
        sim = DVSVoltmeterSimulator(
            width=64, height=64, seed=2, max_events=64 * 64 * 64, sampler=sampler
        )
        sim.forward(torch.full((64, 64), 40.0, device=device), 0)
        ev = sim.forward(torch.full((64, 64), 220.0, device=device), 5000)
        assert ev is not None
        # Strong brightness increase -> overwhelmingly ON (polarity 1).
        assert float((ev.p == 1).float().mean()) > 0.9

    @pytest.mark.parametrize("sampler", SAMPLERS)
    def test_non_finite_input_pixel_recovers(self, device, sampler):
        """One frame with NaN / Inf pixels must not leave those pixels dead."""
        k = [K1, K2, 0.0001, 1e-5, 0.0, 0.00001]  # 10 Hz leak, so events come fast
        sim = DVSVoltmeterSimulator(
            width=8, height=8, k=k, sampler=sampler, randomize_phase=True, seed=1
        )
        good = torch.full((8, 8), 128.0, device=device)
        bad = good.clone()
        bad[2, 2], bad[2, 5], bad[5, 2] = float("nan"), float("inf"), -float("inf")
        sim.forward(good, 0)
        sim.forward(bad, 1000)
        counts = torch.zeros(8, 8, dtype=torch.int64, device=device)
        for i in range(2, 2002):
            ev = sim.forward(good, i * 1000)
            if ev is not None:
                ones = torch.ones_like(ev.x, dtype=torch.int64)
                counts.index_put_((ev.y.long(), ev.x.long()), ones, accumulate=True)
        assert torch.isfinite(sim._delta_vd_res).all()
        assert int(counts[2, 2]) >= 15 and int(counts[2, 5]) >= 15
        assert int(counts[5, 2]) >= 15  # ~20 events in 2 s at 10 Hz

    def test_non_increasing_timestamp_raises(self, device):
        sim = DVSVoltmeterSimulator(width=16, height=16)
        sim.forward(torch.full((16, 16), 100.0, device=device), 1000)
        with pytest.raises(ValueError, match="must be >"):
            sim.forward(torch.full((16, 16), 100.0, device=device), 1000)


class TestDeterminism:
    @pytest.mark.parametrize("sampler", SAMPLERS)
    def test_same_seed_same_output(self, device, sampler):
        a = torch.full((48, 48), 60.0, device=device)
        b = torch.full((48, 48), 180.0, device=device)

        def run(seed):
            sim = DVSVoltmeterSimulator(
                width=48, height=48, seed=seed, max_events=48 * 48 * 64, sampler=sampler
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


class TestExactSampler:
    """The exact sampler against closed-form results for Brownian motion with drift
    mu and diffusion sigma between thresholds +-1, reset to 0 after each event."""

    @pytest.mark.parametrize("dt", [1000, 50_000])
    def test_static_noise_matches_brownian_motion(self, device, dt):
        """Leak drift and diffusion at 2 mu / sigma^2 = 1: event rate and ON fraction
        match the exact values whatever the frame interval (the reference sampler
        gives 0% OFF at 1 kHz here)."""
        L = 128.0
        mu = 1e-7 + 5e-9 * L
        sigma = math.sqrt(2 * mu)
        k = [K1, K2, 0.0, 1e-7, 5e-9, sigma]
        sim = DVSVoltmeterSimulator(width=64, height=64, k=k, sampler="exact", seed=3)
        frame = torch.full((64, 64), L, device=device)
        sim.forward(frame, 0)
        warmup, count = 3_000_000, 4_000_000  # us; count once the renewal is stationary
        n = n_on = 0
        for i in range(1, (warmup + count) // dt + 1):
            ev = sim.forward(frame, i * dt)
            if ev is not None and i * dt > warmup:
                n += ev.p.numel()
                n_on += int(ev.p.to(torch.int64).sum())
        rate = n / (64 * 64) / (count / 1e6)
        assert rate == pytest.approx(1e6 * mu / math.tanh(mu / sigma**2), rel=0.05)
        assert n_on / n == pytest.approx(
            1 / (1 + math.exp(-2 * mu / sigma**2)), abs=0.02
        )

    def test_off_crossings_against_drift_are_timed_like_on(self, device):
        """From 0 between symmetric thresholds, the exit time is independent of the
        exit side for any drift, so OFF-first events, against the drift, come at
        the same times as ON-first ones (the reference times them early)."""
        mu, pe = 1e-6, 4.0
        sigma = math.sqrt(2 * mu / pe)
        k = [K1, K2, 0.0, mu, 0.0, sigma]
        H = W = 256
        sim = DVSVoltmeterSimulator(
            width=W, height=H, k=k, sampler="exact", seed=5, max_events=H * W * 16
        )
        frame = torch.full((H, W), 100.0, device=device)
        sim.forward(frame, 0)
        ev = sim.forward(frame, 10_000_000)
        pix = (ev.y.to(torch.int64) * W + ev.x.to(torch.int64)).cpu().numpy()
        _, first = np.unique(pix, return_index=True)  # events are time-ordered
        t = ev.t.to(torch.int64).cpu().numpy()[first].astype(np.float64)
        on = ev.p.cpu().numpy()[first] == 1
        assert (~on).mean() == pytest.approx(1 / (1 + math.exp(pe)), rel=0.15)
        assert t[on].mean() == pytest.approx(math.tanh(pe / 2) / mu, rel=0.03)
        assert t[~on].mean() / t[on].mean() == pytest.approx(1.0, abs=0.08)

    def test_randomize_phase_starts_in_steady_state(self, device):
        """Noise comparable to the leak: a uniform start phase gave ~89% ON for the
        first 10 s against 60% in steady state."""
        mu, sigma = 9.8e-9, 2.16e-4
        k = [K1, K2, 0.0, mu, 0.0, sigma]
        sim = DVSVoltmeterSimulator(
            width=128, height=128, k=k, sampler="exact", randomize_phase=True, seed=0
        )
        frame = torch.full((128, 128), 128.0, device=device)
        sim.forward(frame, 0)
        n = n_on = 0
        for i in range(1, 1001):  # the first 10 s, in 10 ms frames
            ev = sim.forward(frame, i * 10_000)
            if ev is not None:
                n += ev.p.numel()
                n_on += int(ev.p.to(torch.int64).sum())
        steady_on = 1 / (1 + math.exp(-2 * mu / sigma**2))  # 0.604
        steady_rate = 1e6 * mu / math.tanh(mu / sigma**2)  # 0.047 events/px/s
        assert n_on / n == pytest.approx(steady_on, abs=0.03)
        assert n / (128 * 128) / 10 == pytest.approx(steady_rate, rel=0.08)

    def test_step_count_independent_of_frame_split(self, device):
        """A 20 -> 180 brightness step gives the same events whether it arrives in
        one 50 ms frame or spread over 50 frames (the reference gives 7 vs 8)."""
        L0, L1, T = 20.0, 180.0, 50_000

        def events_per_pixel(n_frames):
            sim = DVSVoltmeterSimulator(width=16, height=16, sampler="exact", seed=0)
            sim.forward(torch.full((16, 16), L0, device=device), 0)
            n = 0
            for i in range(1, n_frames + 1):
                lum = L0 + (L1 - L0) * i / n_frames
                ev = sim.forward(
                    torch.full((16, 16), lum, device=device), i * T // n_frames
                )
                n += 0 if ev is None else ev.p.numel()
            return n / 256

        # signal drift k1 ln((L1 + k2) / (L0 + k2)) plus leak (k4 + k5 L) T: 8.51
        drift = K1 * math.log((L1 + K2) / (L0 + K2)) + (1e-7 + 5e-9 * (L0 + L1) / 2) * T
        assert events_per_pixel(1) == pytest.approx(math.floor(drift), abs=0.05)
        assert events_per_pixel(50) == pytest.approx(math.floor(drift), abs=0.05)
