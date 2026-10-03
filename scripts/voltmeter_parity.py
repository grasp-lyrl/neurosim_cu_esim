"""Parity check: neurosim_cu_esim DVSVoltmeterSimulator vs. the original
DVS-Voltmeter reference (pure PyTorch).

The two use different RNGs (cuRAND Philox vs. torch), so exact event-by-event
match is impossible. Instead we compare *statistics* over a frame sequence on
identical input: total event count, ON/OFF split, mean timestamp, and the final
per-pixel residual-voltage state. It runs our ``sampler="reference"``, the port
of upstream's algorithm; the default ``"exact"`` sampler differs from upstream on
purpose.

The reference source is fetched into a temp dir at runtime (no vendoring).
"""

import os
import sys
import tempfile
import urllib.request

import numpy as np
import torch

REF_BASE = "https://raw.githubusercontent.com/Lynn0306/DVS-Voltmeter/main"
REF_FILES = {
    "simulator.py": f"{REF_BASE}/src/simulator.py",
    "simulator_utils.py": f"{REF_BASE}/src/simulator_utils.py",
}

DVS346_K = [0.00018 * 29250, 20.0, 0.0001, 1e-7, 5e-9, 0.00001]


def load_reference():
    """Download the reference simulator into a temp package and import it."""
    d = tempfile.mkdtemp(prefix="dvsvolt_ref_")
    pkg = os.path.join(d, "ref")
    os.makedirs(pkg, exist_ok=True)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for name, url in REF_FILES.items():
        txt = urllib.request.urlopen(url).read().decode()
        # the reference uses "from .simulator_utils import ..."; keep as a package
        with open(os.path.join(pkg, name), "w") as f:
            f.write(txt)
    sys.path.insert(0, d)
    from ref.simulator import EventSim  # type: ignore

    class _Cfg:
        class SENSOR:
            K = DVS346_K

    return EventSim, _Cfg


def make_frames(h, w, n, seed=0):
    """A simple moving-gradient sequence in linear 0-255 intensity."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(20, 200, size=(h, w)).astype(np.float64)
    frames = []
    for i in range(n):
        shift = np.sin(i * 0.3) * 40.0
        f = np.clip(base + shift + rng.normal(0, 2.0, size=(h, w)), 1.0, 255.0)
        frames.append(f)
    return frames


def run_reference(frames, timestamps):
    EventSim, Cfg = load_reference()
    sim = EventSim(cfg=Cfg, output_folder=tempfile.mkdtemp(), video_name="x")
    all_ev = []
    for f, t in zip(frames, timestamps):
        ev = sim.generate_events(f.astype(np.float32), t)
        if ev is not None:
            all_ev.append(ev)  # [N,4] = (t, x, y, p)
    res = sim.delta_vd_res.cpu().numpy() if hasattr(sim, "delta_vd_res") else None
    if all_ev:
        return np.concatenate(all_ev, axis=0), res
    return np.zeros((0, 4), dtype=np.int32), res


def run_ours(frames, timestamps, h, w, seed=0, dtype=torch.float32):
    from neurosim_cu_esim import DVSVoltmeterSimulator

    sim = DVSVoltmeterSimulator(
        width=w,
        height=h,
        camera_type="DVS346",
        sampler="reference",
        max_events=h * w * 64,
        seed=seed,
        device="cuda",
    )
    ts_all, p_all = [], []
    n = 0
    for f, t in zip(frames, timestamps):
        ev = sim.forward(torch.from_numpy(f).to(dtype).cuda(), t)
        if ev is not None:
            ts_all.append(ev.t.cpu().numpy().astype(np.int64))
            p_all.append(ev.p.cpu().numpy().astype(np.int64))
            n += ev.x.numel()
    res = sim._delta_vd_res.cpu().numpy() if sim._delta_vd_res is not None else None
    t = np.concatenate(ts_all) if ts_all else np.zeros(0, np.int64)
    p = np.concatenate(p_all) if p_all else np.zeros(0, np.int64)
    return n, t, p, res


def summarize(name, n, t, p):
    on = int(np.sum(p == 1))
    off = int(np.sum(p == 0))
    mt = float(np.mean(t)) if len(t) else 0.0
    print(
        f"  {name:10s}: events={n:8d}  on={on:8d}  off={off:8d}  "
        f"on_frac={on / max(1, n):.3f}  mean_t={mt:.1f}"
    )
    return dict(n=n, on=on, off=off, mean_t=mt)


def main():
    h, w = 64, 64
    n_frames = 20
    dt = 5000  # us between frames (200 fps)
    timestamps = [i * dt for i in range(n_frames)]
    frames = make_frames(h, w, n_frames)

    print(f"Sequence: {n_frames} frames @ {h}x{w}, dt={dt}us\n")

    print("Reference (PyTorch):")
    ref_ev, ref_res = run_reference(frames, timestamps)
    rt = ref_ev[:, 0].astype(np.int64)
    rp = ref_ev[:, 3].astype(np.int64)
    ref = summarize("reference", ref_ev.shape[0], rt, rp)

    print("\nOurs (CUDA, float32):")
    n, t, p, our_res = run_ours(frames, timestamps, h, w, seed=0, dtype=torch.float32)
    ours = summarize("ours-f32", n, t, p)

    print("\n=== Comparison ===")
    if ref["n"] > 0:
        ratio = ours["n"] / ref["n"]
        print(f"  event-count ratio (ours/ref): {ratio:.3f}")
        print(
            f"  on-fraction  ref={ref['on'] / max(1, ref['n']):.3f}  "
            f"ours={ours['on'] / max(1, ours['n']):.3f}"
        )
        print(f"  mean-t       ref={ref['mean_t']:.1f}  ours={ours['mean_t']:.1f}")
    if ref_res is not None and our_res is not None:
        print(
            f"  residual L1 mean |ref-ours| = "
            f"{np.mean(np.abs(ref_res - our_res)):.4e}  "
            f"(ref std={np.std(ref_res):.4e})"
        )


if __name__ == "__main__":
    main()
