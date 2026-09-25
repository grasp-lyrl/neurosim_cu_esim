"""Run the GracaDVSSimulator on the repo's own example stimulus (the moving
textured patch from benchmark_esim.py) and save the full event stream.

Reuses precompute_frame_bank() from benchmark_esim so the input is byte-for-byte
the same moving-texture-on-white sequence the repo uses for its example GIF.
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_esim import precompute_frame_bank  # noqa: E402

from neurosim_cu_esim import GracaDVSSimulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=1000)
    ap.add_argument("--velocity-px-s", type=float, default=500.0)
    ap.add_argument("--texture-seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--dt-us", type=float, default=50.0)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--out", type=str, default="events_example.npz")
    args = ap.parse_args()

    bank = precompute_frame_bank(
        width=args.width, height=args.height, fps=args.fps,
        velocity_px_s=args.velocity_px_s,
        texture_width=args.width, texture_height=args.height,
        texture_seed=args.texture_seed, precompute_frames=args.frames,
        device="cuda",
    )
    sim = GracaDVSSimulator(
        width=args.width, height=args.height, add_noise=args.noise,
        contrast_threshold=0.25,
        max_events=args.width * args.height * 16, seed=0, device="cuda",
    )

    step_us = int(round(1e6 / args.fps))
    xs, ys, ts, ps = [], [], [], []
    sim.forward(bank[0] * 1e-12, 0)                       # init
    for i in range(1, args.frames):
        ev = sim.forward(bank[i] * 1e-12, i * step_us)
        if ev is None:
            continue
        xs.append(ev.x.cpu().numpy()); ys.append(ev.y.cpu().numpy())
        ts.append(ev.t.cpu().numpy()); ps.append(ev.p.cpu().numpy())

    x = np.concatenate(xs); y = np.concatenate(ys)
    t = np.concatenate(ts).astype(np.uint64); p = np.concatenate(ps)
    np.savez(args.out, x=x, y=y, t=t, p=p, H=args.height, W=args.width,
             fps=args.fps, frames=args.frames, velocity=args.velocity_px_s,
             texture_seed=args.texture_seed)
    print(f"saved {args.out}: {x.size} events  ON={int((p==1).sum())} "
          f"OFF={int((p==0).sum())}  over {args.frames} frames "
          f"(t up to {int(t.max())} us)")


if __name__ == "__main__":
    main()
