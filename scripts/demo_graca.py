"""Minimal array-level demo of the Graca physical DVS model.

Sweeps a soft vertical edge across a frame and runs GracaDVSSimulator over the
whole sensor, reporting per-frame event counts and ON/OFF balance. Optionally
dumps an aggregated event-count image to .npz. No video deps required.

    python scripts/demo_graca.py [--noise] [--height H --width W --frames N]
"""
import argparse
import numpy as np
import torch

from neurosim_cu_esim import GracaDVSSimulator


def make_bar(h, w, pos, lo=0.05, hi=1.0, half=8.0, soft=2.0):
    """A bright vertical bar (width 2*half) centred at column `pos` on a dark
    background. Its leading edge brightens pixels (ON) and trailing edge darkens
    them (OFF), so a sweep exercises both polarities."""
    xs = np.arange(w)[None, :].repeat(h, 0).astype(np.float32)
    left = 1.0 / (1.0 + np.exp(-(xs - (pos - half)) / soft))
    right = 1.0 / (1.0 + np.exp(-(xs - (pos + half)) / soft))
    return lo + (hi - lo) * (left - right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--width", type=int, default=240)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--dt-us", type=float, default=20.0)
    ap.add_argument("--frame-us", type=int, default=1000, help="inter-frame us")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    dev = "cuda"
    sim = GracaDVSSimulator(
        width=args.width, height=args.height,
        add_noise=args.noise,
        full_well_saturation_threshold=1e-12, dark_current=1e-15,
        contrast_threshold=0.3, refractory_us=100.0,
        max_events=args.width * args.height * 16, seed=0, device=dev,
    )

    counts = np.zeros((args.height, args.width), np.int64)
    tot_on = tot_off = 0
    for i in range(args.frames):
        pos = (i / max(1, args.frames - 1)) * (args.width - 1)
        # make_bar returns [0.05, 1.0], scale it to physical photocurrent [0, 1e-12]
        frame_np = make_bar(args.height, args.width, pos) * 1e-12
        frame = torch.from_numpy(frame_np).to(dev)
        ts = i * args.frame_us
        ev = sim.forward(frame, ts) if i > 0 else (sim.forward(frame, ts), None)[1]
        if ev is None:
            continue
        on = int((ev.p == 1).sum().item())
        off = int((ev.p == 0).sum().item())
        tot_on += on
        tot_off += off
        ys = ev.y.to(torch.int64).cpu().numpy()
        xs = ev.x.to(torch.int64).cpu().numpy()
        np.add.at(counts, (ys, xs), 1)
        print(f"frame {i:3d} t={ts:7d}us  events={on+off:7d}  ON={on:6d} OFF={off:6d}")

    print(f"\nTOTAL  ON={tot_on}  OFF={tot_off}  events={tot_on+tot_off}")
    print(f"active pixels: {(counts>0).sum()} / {counts.size}")
    if args.out:
        np.savez(args.out, counts=counts)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
