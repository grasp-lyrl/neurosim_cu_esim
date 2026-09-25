"""Run an event simulator on a real video file.

Decodes an input video to grayscale frames, runs an event simulator over the
sequence, and renders a side-by-side MP4: the input frame (left) and the events
generated in that frame interval (right; red = ON / +, blue = OFF / -).

Modes:
  * ``voltmeter`` — DVS-Voltmeter stochastic model; frames fed as linear 0-255.
  * ``single`` / ``multi`` — ESIM log-contrast model; frames clamped to >=1.

Usage:
    python scripts/simulate_on_video.py --input in.mp4 --output out.mp4 \
        --width 480 --mode voltmeter --camera-type DVS346
    python scripts/simulate_on_video.py --input in.mp4 --output out.mp4 \
        --width 480 --mode single
"""

import argparse
import subprocess

import cv2
import numpy as np
import torch

from neurosim_cu_esim import EventSimulator, DVSVoltmeterSimulator, GracaDVSSimulator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output MP4 path")
    p.add_argument("--width", type=int, default=480, help="Resize width (keeps aspect)")
    p.add_argument(
        "--mode",
        default="voltmeter",
        choices=["single", "multi", "voltmeter", "graca"],
        help="Event model: 'voltmeter' (DVS-Voltmeter stochastic), "
        "'graca' (Graca & Delbruck physically-realistic model), or the ESIM "
        "log-contrast model in 'single'/'multi' mode.",
    )
    p.add_argument("--camera-type", default="DVS346", choices=["DVS346", "DVS240"])
    p.add_argument(
        "--leak-scale",
        type=float,
        default=1.0,
        help="voltmeter: scale leakage drift k4/k5 (1.0=faithful, 0=signal-only). "
        "Lower to reduce background ON flashing on bright/overexposed regions.",
    )
    p.add_argument(
        "--contrast",
        type=float,
        default=0.35,
        help="single/multi: log-contrast threshold (both polarities).",
    )
    p.add_argument(
        "--randomize-phase",
        action="store_true",
        help="voltmeter: random per-pixel initial leakage phase, so static "
        "backgrounds give a desynchronised noise sparkle instead of full-frame "
        "flashes.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument(
        "--out-fps",
        type=float,
        default=0.0,
        help="Output playback fps (0 = input fps; lower it to slow-mo the result)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.input}")
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = args.width
    h = int(round(src_h * w / src_w))
    h -= h % 2
    print(f"input: {src_w}x{src_h} @ {in_fps:.2f} fps -> processing at {w}x{h}")

    if args.mode == "voltmeter":
        sim = DVSVoltmeterSimulator(
            width=w,
            height=h,
            camera_type=args.camera_type,
            leak_scale=args.leak_scale,
            randomize_phase=args.randomize_phase,
            max_events=w * h * 16,
            seed=args.seed,
            device="cuda",
        )
    elif args.mode == "graca":
        sim = GracaDVSSimulator(
            width=w,
            height=h,
            max_events=w * h * 16,
            device="cuda",
        )
    else:
        # ESIM log-contrast model. multi can emit many events/pixel/frame.
        max_events = w * h * 16 if args.mode == "multi" else None
        sim = EventSimulator(
            width=w,
            height=h,
            mode=args.mode,
            contrast_threshold_neg=args.contrast,
            contrast_threshold_pos=args.contrast,
            max_events=max_events,
            device="cuda",
        )

    out_fps = args.out_fps if args.out_fps > 0 else in_fps
    # High-quality H.264 via an ffmpeg pipe (-crf 14, near-lossless) instead of
    # OpenCV's heavily-compressed default writer.
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w * 2}x{h}",
            "-r",
            f"{out_fps}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            args.output,
        ],
        stdin=subprocess.PIPE,
    )

    dt_us = 1.0e6 / in_fps
    i = 0
    total_ev = total_on = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if args.max_frames and i >= args.max_frames:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        gray_f = torch.from_numpy(gray).float().cuda()  # linear 0-255
        if args.mode not in ["voltmeter", "graca"]:
            gray_f = gray_f.clamp_(min=1.0)  # ESIM takes log(); avoid log(0)
        elif args.mode == "graca":
            gray_f = (gray_f / 255.0) * 1e-12  # Graca wants absolute photocurrent in Amperes
        ts = int(round(i * dt_us))

        ev = sim.forward(gray_f, ts)  # first call inits and returns None

        # Left: the clean input frame. Right: events on a white background.
        left = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        right = np.full((h, w, 3), 255, dtype=np.uint8)
        if ev is not None:
            x = ev.x.to(torch.int64).cpu().numpy()
            y = ev.y.to(torch.int64).cpu().numpy()
            p = ev.p.to(torch.int64).cpu().numpy()
            on = p == 1
            right[y[on], x[on]] = (0, 0, 255)  # red (BGR) = ON / +
            right[y[~on], x[~on]] = (255, 0, 0)  # blue = OFF / -
            total_ev += x.size
            total_on += int(on.sum())

        ffmpeg.stdin.write(np.hstack([left, right]).tobytes())
        i += 1

    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    on_frac = total_on / max(1, total_ev)
    print(
        f"processed {i} frames, {total_ev} events "
        f"(on={on_frac:.2f}, off={1 - on_frac:.2f})"
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
