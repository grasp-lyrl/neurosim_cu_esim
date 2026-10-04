"""The extension runs on its inputs' GPU and on the caller's CUDA stream."""

import pytest
import torch

from neurosim_cu_esim import DVSVoltmeterSimulator, EventSimulator

pytestmark = pytest.mark.cuda

H, W = 48, 64
MODES = ["single", "multi", "exact", "reference"]  # ESIM modes, Voltmeter samplers


def frames(mode):
    """Twenty random frames, scaled for ESIM (0, 1] or the Voltmeter (0, 255)."""
    gen = torch.Generator().manual_seed(0)
    scale = 1.0 if mode in ("single", "multi") else 255.0
    return [
        (torch.rand(H, W, generator=gen) * scale).clamp(min=1e-4) for _ in range(20)
    ]


def run(mode, device, busy_cycles=0):
    """Events from the frames at 1 kHz. With ``busy_cycles``, each frame is copied
    on the current stream behind a busy kernel, so it is not ready until that ends."""
    if mode in ("single", "multi"):
        # Room for every event: which ones a full buffer drops is not fixed.
        sim = EventSimulator(
            width=W, height=H, mode=mode, max_events=W * H * 32, device=device
        )
    else:
        sim = DVSVoltmeterSimulator(
            width=W, height=H, sampler=mode, randomize_phase=True, seed=3, device=device
        )
    on_device = [f.to(device) for f in frames(mode)]
    torch.cuda.synchronize(device)
    out = []
    for i, frame in enumerate(on_device):
        if busy_cycles:
            torch.cuda._sleep(busy_cycles)
        ev = sim.forward(frame.clone(), 1000 * (i + 1))
        if ev is not None:
            out.append([v.to(torch.int64).cpu() for v in ev])
    return out


def assert_same_events(a, b):
    """Same events in every frame (the order within a microsecond is not fixed)."""
    assert len(a) == len(b) > 0
    for (xa, ya, ta, pa), (xb, yb, tb, pb) in zip(a, b):
        key_a = ((ta * H + ya) * W + xa) * 2 + pa
        key_b = ((tb * H + yb) * W + xb) * 2 + pb
        assert torch.equal(key_a.sort().values, key_b.sort().values)


@pytest.mark.parametrize("mode", MODES)
def test_runs_on_the_callers_stream(mode, device):
    """Inside ``torch.cuda.stream(s)`` the kernels queue on s behind the frame's
    copy; launched on another stream they would read the frame before it exists."""
    expected = run(mode, device)
    with torch.cuda.stream(torch.cuda.Stream()):
        got = run(mode, device, busy_cycles=20_000_000)  # ~10 ms per frame
    assert_same_events(expected, got)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", MODES)
def test_runs_on_a_gpu_that_is_not_current(mode):
    """A simulator on cuda:1 while cuda:0 is current gives the same events as one
    on cuda:0 (it used to be an illegal memory access)."""
    with torch.cuda.device(0):
        assert_same_events(run(mode, "cuda:0"), run(mode, "cuda:1"))
