"""Benchmark suite for neurosim_cu_esim EventSimulator.

Metrics reported:
- Forward calls per second (kHz)
- Events generated per second (Mev/s)
- Average events per forward call
- Mean and peak GPU utilization (%) during benchmark window
- Mean forward latency (microseconds, CUDA-event timing)

The benchmark uses a precomputed bank of textured VGA frames on a white
background (moving texture patch), then repeatedly calls EventSimulator.forward
for a fixed number of iterations.
"""

import json
import time
import argparse
import threading
import subprocess
from pathlib import Path
from collections.abc import Callable
from datetime import datetime, timezone
from dataclasses import asdict, dataclass

import torch
import numpy as np

from neurosim_cu_esim import EventSimulator


@dataclass
class TrialResult:
    backend: str
    trial_index: int
    elapsed_s: float
    calls: int
    total_events: int
    calls_per_sec: float
    calls_khz: float
    events_per_sec: float
    events_mev_per_sec: float
    events_per_call: float
    avg_forward_latency_us: float
    gpu_util_mean_pct: float | None
    gpu_util_peak_pct: float | None


class GpuUtilSampler:
    """Poll GPU utilization from nvidia-smi in a background thread."""

    def __init__(self, gpu_index: int, interval_s: float = 0.1) -> None:
        self.gpu_index = gpu_index
        self.interval_s = max(0.05, interval_s)
        self.values: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> float | None:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                str(self.gpu_index),
            ]
            out = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, text=True, timeout=2.0
            )
            line = out.strip().splitlines()[0].strip()
            return float(line)
        except Exception:
            return None

    def _loop(self) -> None:
        while self._running:
            val = self._poll_once()
            if val is not None:
                self.values.append(val)
            time.sleep(self.interval_s)

    def start(self) -> None:
        self.values.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float | None, float | None]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.values:
            return None, None
        arr = np.asarray(self.values, dtype=np.float32)
        return float(arr.mean()), float(arr.max())


def make_random_texture(
    width: int, height: int, seed: int, smooth_passes: int = 3
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = rng.random((height, width), dtype=np.float32)
    for _ in range(max(0, smooth_passes)):
        tex = (
            tex
            + np.roll(tex, 1, axis=0)
            + np.roll(tex, -1, axis=0)
            + np.roll(tex, 1, axis=1)
            + np.roll(tex, -1, axis=1)
        ) / 5.0

    tex = (tex - tex.min()) / max(1e-6, float(tex.max() - tex.min()))
    tex = 0.1 + 0.8 * tex
    return tex.astype(np.float32)


def polarity_rgb(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Convert boolean polarity maps to an RGB visualization."""
    rgb = np.ones((pos.shape[0], pos.shape[1], 3), dtype=np.float32)
    only_pos = pos & ~neg
    only_neg = neg & ~pos
    both = pos & neg

    rgb[only_pos] = np.array([1.0, 0.2, 0.2], dtype=np.float32)  # red
    rgb[only_neg] = np.array([0.2, 0.2, 1.0], dtype=np.float32)  # blue
    rgb[both] = np.array([1.0, 0.0, 1.0], dtype=np.float32)  # magenta overlap
    return rgb


def render_texture_patch_frame(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    texture: np.ndarray,
    white_value: float = 1.0,
) -> np.ndarray:
    frame = np.full((height, width), white_value, dtype=np.float32)
    tex_h, tex_w = texture.shape

    cx = int(round(center_x))
    cy = int(round(center_y))

    left = cx - (tex_w // 2)
    top = cy - (tex_h // 2)
    right = left + tex_w
    bottom = top + tex_h

    dst_x0 = max(0, left)
    dst_y0 = max(0, top)
    dst_x1 = min(width, right)
    dst_y1 = min(height, bottom)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        src_x0 = dst_x0 - left
        src_y0 = dst_y0 - top
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        frame[dst_y0:dst_y1, dst_x0:dst_x1] = texture[src_y0:src_y1, src_x0:src_x1]

    return frame


def precompute_frame_bank(
    width: int,
    height: int,
    fps: int,
    velocity_px_s: float,
    texture_width: int,
    texture_height: int,
    texture_seed: int,
    precompute_frames: int,
    device: str,
) -> list[torch.Tensor]:
    """Precompute a cyclic frame bank on GPU for throughput benchmarks."""
    if velocity_px_s <= 0:
        raise ValueError("velocity_px_s must be > 0")

    texture = make_random_texture(texture_width, texture_height, texture_seed)

    start = np.array([0.0, float(height - 1)], dtype=np.float64)
    end = np.array([float(width - 1), 0.0], dtype=np.float64)
    path = end - start
    path_len = float(np.linalg.norm(path))
    direction = path / path_len

    bank: list[torch.Tensor] = []
    for i in range(precompute_frames):
        t_s = i / float(fps)
        d = min(velocity_px_s * t_s, path_len)
        pos = start + direction * d
        frame_np = render_texture_patch_frame(
            width=width,
            height=height,
            center_x=pos[0],
            center_y=pos[1],
            texture=texture,
        )
        bank.append(torch.from_numpy(frame_np).to(device=device, non_blocking=False))

    return bank


def stream_sanity_frames(
    frame_bank: list[torch.Tensor],
    width: int,
    height: int,
    fps_timestamp: int,
    bin_ms: float,
    sim: EventSimulator,
    max_frames: int,
):
    """Yield visualization frames for sanity-check animation.

    Yields tuples of:
        (simulated_gray_frame, aggregated_event_rgb, bin_start_us, events_in_bin)
    """
    bin_us = int(round(bin_ms * 1000.0))
    timestamp_step_us = int(round(1e6 / fps_timestamp))

    pos = np.zeros((height, width), dtype=bool)
    neg = np.zeros((height, width), dtype=bool)
    current_bin_start_us = 0
    current_bin_events = 0

    if not frame_bank:
        return

    sim.init(frame_bank[0])
    latest_frame = frame_bank[0].detach().cpu().numpy()

    for i in range(1, max_frames):
        timestamp_us = i * timestamp_step_us
        frame = frame_bank[i]
        latest_frame = frame.detach().cpu().numpy()

        while timestamp_us >= current_bin_start_us + bin_us:
            rgb = polarity_rgb(pos, neg)
            yield latest_frame, rgb, current_bin_start_us, current_bin_events
            pos.fill(False)
            neg.fill(False)
            current_bin_events = 0
            current_bin_start_us += bin_us

        events = sim(frame, timestamp_us)
        if events is not None:
            x = events.x.detach().cpu().numpy().astype(np.int32)
            y = events.y.detach().cpu().numpy().astype(np.int32)
            p = events.p.detach().cpu().numpy().astype(np.uint8)
            pos_mask = p == 1
            neg_mask = ~pos_mask
            if np.any(pos_mask):
                pos[y[pos_mask], x[pos_mask]] = True
            if np.any(neg_mask):
                neg[y[neg_mask], x[neg_mask]] = True
            current_bin_events += x.size

    rgb = polarity_rgb(pos, neg)
    yield latest_frame, rgb, current_bin_start_us, current_bin_events


def animate_sanity_check(
    frame_bank: list[torch.Tensor],
    width: int,
    height: int,
    fps_timestamp: int,
    bin_ms: float,
    device: str,
    contrast_threshold_neg: float,
    contrast_threshold_pos: float,
    save_path: str,
    dpi: int,
    max_frames: int,
    show: bool,
) -> None:
    """Create an MP4 sanity-check animation (frame + 20 ms events)."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    vis_fps = 1000.0 / bin_ms

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_left, ax_right = axes

    frame_artist = ax_left.imshow(
        np.ones((height, width), dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0
    )
    event_artist = ax_right.imshow(
        np.ones((height, width, 3), dtype=np.float32), vmin=0.0, vmax=1.0
    )

    ax_left.set_title("Simulated input frame")
    ax_right.set_title("Aggregated events (red=+, blue=-)")
    ax_left.axis("off")
    ax_right.axis("off")

    stats = {"bins": 0, "events": 0}

    sim = EventSimulator(
        width=width,
        height=height,
        contrast_threshold_neg=contrast_threshold_neg,
        contrast_threshold_pos=contrast_threshold_pos,
        device=device,
    )

    def update(payload):
        frame, event_rgb, bin_start_us, events_in_bin = payload
        stats["bins"] += 1
        stats["events"] += events_in_bin

        frame_artist.set_data(frame)
        event_artist.set_data(event_rgb)

        t0_ms = bin_start_us / 1000.0
        t1_ms = t0_ms + bin_ms
        ax_left.set_title(f"Simulated frame at ~{t1_ms:.1f} ms")
        ax_right.set_title(
            f"Events {t0_ms:.1f}-{t1_ms:.1f} ms | bin events: {events_in_bin}"
        )
        return frame_artist, event_artist

    frame_stream = stream_sanity_frames(
        frame_bank=frame_bank,
        width=width,
        height=height,
        fps_timestamp=fps_timestamp,
        bin_ms=bin_ms,
        sim=sim,
        max_frames=max_frames,
    )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_stream,
        interval=1000.0 / vis_fps,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )

    writer = animation.FFMpegWriter(fps=vis_fps, codec="libx264", bitrate=24000)
    ani.save(save_path, writer=writer, dpi=dpi)
    print(f"Saved sanity-check MP4: {save_path}")
    print(f"Visualization rate: {vis_fps:.1f} Hz ({bin_ms:.1f} ms bins)")
    print(f"Total visualization bins: {stats['bins']}")
    print(f"Total events emitted: {stats['events']}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_single_trial(
    backend: str,
    trial_index: int,
    width: int,
    height: int,
    fps_timestamp: int,
    warmup_calls: int,
    latency_calls: int,
    frame_bank: list[torch.Tensor],
    runner_factory: Callable[
        [],
        tuple[
            Callable[[torch.Tensor, int], None],
            Callable[[torch.Tensor, int], int],
        ],
    ],
    gpu_index: int,
    gpu_util_poll_interval_s: float,
) -> TrialResult:
    init_fn, step_fn = runner_factory()

    timestamp_step_us = int(round(1e6 / fps_timestamp))
    ts = 0

    init_fn(frame_bank[0], ts)
    ts += timestamp_step_us

    for i in range(warmup_calls):
        step_fn(frame_bank[i % len(frame_bank)], ts)
        ts += timestamp_step_us

    torch.cuda.synchronize()

    sampler = GpuUtilSampler(gpu_index=gpu_index, interval_s=gpu_util_poll_interval_s)
    sampler.start()

    calls = max(0, latency_calls)
    total_events = 0

    start_wall = time.perf_counter()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for i in range(calls):
        frame = frame_bank[i % len(frame_bank)]
        total_events += step_fn(frame, ts)
        ts += timestamp_step_us
    end_evt.record()

    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_wall
    total_ms = float(start_evt.elapsed_time(end_evt)) if calls > 0 else 0.0

    gpu_mean, gpu_peak = sampler.stop()

    calls_per_sec = calls / elapsed_s if elapsed_s > 0 else 0.0
    events_per_sec = total_events / elapsed_s if elapsed_s > 0 else 0.0
    avg_latency_us = (total_ms * 1000.0) / calls if calls > 0 else 0.0

    return TrialResult(
        backend=backend,
        trial_index=trial_index,
        elapsed_s=elapsed_s,
        calls=calls,
        total_events=total_events,
        calls_per_sec=calls_per_sec,
        calls_khz=calls_per_sec / 1000.0,
        events_per_sec=events_per_sec,
        events_mev_per_sec=events_per_sec / 1e6,
        events_per_call=(total_events / calls) if calls > 0 else 0.0,
        avg_forward_latency_us=avg_latency_us,
        gpu_util_mean_pct=gpu_mean,
        gpu_util_peak_pct=gpu_peak,
    )


def summarize_trials(trials: list[TrialResult]) -> dict[str, float | None]:
    def arr(name: str) -> np.ndarray:
        return np.asarray([getattr(t, name) for t in trials], dtype=np.float64)

    summary: dict[str, float | None] = {
        "trials": float(len(trials)),
        "calls_khz_mean": float(arr("calls_khz").mean()),
        "calls_khz_std": float(arr("calls_khz").std(ddof=0)),
        "events_mev_per_sec_mean": float(arr("events_mev_per_sec").mean()),
        "events_mev_per_sec_std": float(arr("events_mev_per_sec").std(ddof=0)),
        "events_per_call_mean": float(arr("events_per_call").mean()),
        "latency_us_mean": float(arr("avg_forward_latency_us").mean()),
        "latency_us_std": float(arr("avg_forward_latency_us").std(ddof=0)),
    }

    gpu_means = [t.gpu_util_mean_pct for t in trials if t.gpu_util_mean_pct is not None]
    gpu_peaks = [t.gpu_util_peak_pct for t in trials if t.gpu_util_peak_pct is not None]
    summary["gpu_util_mean_pct"] = float(np.mean(gpu_means)) if gpu_means else None
    summary["gpu_util_peak_pct"] = float(np.max(gpu_peaks)) if gpu_peaks else None
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--width", type=int, default=640, help="Frame width (default: VGA 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Frame height (default: VGA 480)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch CUDA device")

    parser.add_argument(
        "--trials", type=int, default=3, help="Number of throughput trials"
    )
    parser.add_argument(
        "--warmup-calls",
        type=int,
        default=2000,
        help="Warmup forward calls before timing",
    )
    parser.add_argument(
        "--latency-calls",
        type=int,
        default=1000000,
        help="Number of forward calls per trial (also used for latency timing)",
    )

    parser.add_argument(
        "--fps-timestamp",
        type=int,
        default=1000,
        help="Timestamp rate (Hz) used for event timestamps",
    )
    parser.add_argument(
        "--velocity-px-s",
        type=float,
        default=500.0,
        help="Texture patch motion speed in px/s",
    )
    parser.add_argument(
        "--texture-width", type=int, default=640, help="Moving texture patch width"
    )
    parser.add_argument(
        "--texture-height", type=int, default=480, help="Moving texture patch height"
    )
    parser.add_argument("--texture-seed", type=int, default=7, help="Texture RNG seed")
    parser.add_argument(
        "--precompute-frames",
        type=int,
        default=2000,
        help="Number of precomputed frames in frame bank",
    )

    parser.add_argument("--contrast-threshold-neg", type=float, default=0.25)
    parser.add_argument("--contrast-threshold-pos", type=float, default=0.25)

    parser.add_argument(
        "--gpu-util-poll-interval-s",
        type=float,
        default=0.1,
        help="nvidia-smi polling interval",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmarks/esim_benchmark_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--sanity-video",
        type=str,
        default=None,
        help="Optional MP4 path for frame+event sanity-check animation",
    )
    parser.add_argument(
        "--sanity-bin-ms",
        type=float,
        default=20.0,
        help="Event aggregation window in ms for sanity-check animation",
    )
    parser.add_argument(
        "--sanity-frames",
        type=int,
        default=0,
        help="Number of frames for sanity animation (0 = full frame bank)",
    )
    parser.add_argument(
        "--sanity-dpi", type=int, default=300, help="Sanity MP4 render DPI"
    )
    parser.add_argument(
        "--sanity-show", action="store_true", help="Display sanity animation window"
    )
    parser.add_argument(
        "--compare-vid2e",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also benchmark thirdparty cu_rpg_vid2e_esim reference backend "
            "(default: enabled)."
        ),
    )

    return parser.parse_args()


def make_neurosim_runner_factory(
    width: int,
    height: int,
    contrast_threshold_neg: float,
    contrast_threshold_pos: float,
    device: str,
) -> Callable[
    [],
    tuple[
        Callable[[torch.Tensor, int], None],
        Callable[[torch.Tensor, int], int],
    ],
]:
    def factory() -> tuple[
        Callable[[torch.Tensor, int], None],
        Callable[[torch.Tensor, int], int],
    ]:
        sim = EventSimulator(
            width=width,
            height=height,
            contrast_threshold_neg=contrast_threshold_neg,
            contrast_threshold_pos=contrast_threshold_pos,
            device=device,
        )

        def init_fn(first_frame: torch.Tensor, _first_ts: int) -> None:
            sim.init(first_frame)

        def step_fn(frame: torch.Tensor, ts: int) -> int:
            events = sim(frame, ts)
            if events is None:
                return 0
            return int(events.x.numel())

        return init_fn, step_fn

    return factory


def make_vid2e_runner_factory(
    width: int,
    height: int,
    contrast_threshold_neg: float,
    contrast_threshold_pos: float,
    device: str,
) -> Callable[
    [],
    tuple[
        Callable[[torch.Tensor, int], None],
        Callable[[torch.Tensor, int], int],
    ],
]:
    from thirdparty.cu_rpg_vid2e_esim import EventSimulatorVID2E_ESIM  # noqa: E402

    def factory() -> tuple[
        Callable[[torch.Tensor, int], None],
        Callable[[torch.Tensor, int], int],
    ]:
        sim = EventSimulatorVID2E_ESIM(
            W=width,
            H=height,
            start_time=0,
            first_image=None,
            contrast_threshold_neg=contrast_threshold_neg,
            contrast_threshold_pos=contrast_threshold_pos,
            device=device,
        )

        def init_fn(first_frame: torch.Tensor, _first_ts: int) -> None:
            sim.reset(first_image=first_frame)

        def step_fn(frame: torch.Tensor, ts: int) -> int:
            events = sim.image_callback(frame, ts)
            if events is None:
                return 0
            return int(events[0].numel())

        return init_fn, step_fn

    return factory


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not args.device.startswith("cuda"):
        raise ValueError("Benchmark is CUDA-only; use --device cuda")

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    dev = torch.device(args.device)
    gpu_index = dev.index if dev.index is not None else torch.cuda.current_device()

    print("Preparing frame bank...")
    frame_bank = precompute_frame_bank(
        width=args.width,
        height=args.height,
        fps=args.fps_timestamp,
        velocity_px_s=args.velocity_px_s,
        texture_width=args.texture_width,
        texture_height=args.texture_height,
        texture_seed=args.texture_seed,
        precompute_frames=args.precompute_frames,
        device=args.device,
    )

    frame_bytes = args.width * args.height * 4
    approx_gb = (frame_bytes * len(frame_bank)) / (1024**3)
    print(f"Frame bank: {len(frame_bank)} frames, ~{approx_gb:.2f} GiB on GPU")

    if args.sanity_video is not None:
        max_frames = args.sanity_frames if args.sanity_frames > 0 else len(frame_bank)
        max_frames = min(max_frames, len(frame_bank))
        if max_frames < 2:
            raise ValueError("Sanity animation requires at least 2 frames in the bank.")
        print(
            f"Rendering sanity animation with {max_frames} frames "
            f"({args.sanity_bin_ms:.1f} ms bins)..."
        )
        animate_sanity_check(
            frame_bank=frame_bank[:max_frames],
            width=args.width,
            height=args.height,
            fps_timestamp=args.fps_timestamp,
            bin_ms=args.sanity_bin_ms,
            device=args.device,
            contrast_threshold_neg=args.contrast_threshold_neg,
            contrast_threshold_pos=args.contrast_threshold_pos,
            save_path=args.sanity_video,
            dpi=args.sanity_dpi,
            max_frames=max_frames,
            show=args.sanity_show,
        )

    backend_factories: dict[
        str,
        Callable[
            [],
            tuple[
                Callable[[torch.Tensor, int], None],
                Callable[[torch.Tensor, int], int],
            ],
        ],
    ] = {
        "neurosim_cu_esim": make_neurosim_runner_factory(
            width=args.width,
            height=args.height,
            contrast_threshold_neg=args.contrast_threshold_neg,
            contrast_threshold_pos=args.contrast_threshold_pos,
            device=args.device,
        )
    }

    if args.compare_vid2e:
        try:
            backend_factories["cu_rpg_vid2e_esim"] = make_vid2e_runner_factory(
                width=args.width,
                height=args.height,
                contrast_threshold_neg=args.contrast_threshold_neg,
                contrast_threshold_pos=args.contrast_threshold_pos,
                device=args.device,
            )
        except Exception as exc:
            print(
                "Warning: could not enable cu_rpg_vid2e_esim comparison: "
                f"{type(exc).__name__}: {exc}"
            )

    backend_results: dict[str, dict[str, object]] = {}

    for backend_name, backend_factory in backend_factories.items():
        print(f"\n=== Running backend: {backend_name} ===")
        trials: list[TrialResult] = []

        for i in range(args.trials):
            result = run_single_trial(
                backend=backend_name,
                trial_index=i + 1,
                width=args.width,
                height=args.height,
                fps_timestamp=args.fps_timestamp,
                warmup_calls=args.warmup_calls,
                latency_calls=args.latency_calls,
                frame_bank=frame_bank,
                runner_factory=backend_factory,
                gpu_index=gpu_index,
                gpu_util_poll_interval_s=args.gpu_util_poll_interval_s,
            )
            trials.append(result)

            print(
                f"Trial {result.trial_index}: "
                f"{result.calls_khz:.2f} kHz, "
                f"{result.events_mev_per_sec:.2f} Mev/s, "
                f"latency {result.avg_forward_latency_us:.2f} us, "
                f"GPU util mean/peak "
                f"{result.gpu_util_mean_pct if result.gpu_util_mean_pct is not None else 'n/a'}"
                f"/"
                f"{result.gpu_util_peak_pct if result.gpu_util_peak_pct is not None else 'n/a'} %"
            )

        summary = summarize_trials(trials)
        backend_results[backend_name] = {
            "trials": [asdict(t) for t in trials],
            "summary": summary,
        }

        print(f"\n=== Summary ({backend_name}) ===")
        print(
            f"Calls/sec: {summary['calls_khz_mean']:.2f} ± "
            f"{summary['calls_khz_std']:.2f} kHz"
        )
        print(
            f"Events/sec: {summary['events_mev_per_sec_mean']:.2f} ± "
            f"{summary['events_mev_per_sec_std']:.2f} Mev/s"
        )
        print(f"Events/call: {summary['events_per_call_mean']:.2f}")
        print(
            f"Forward latency: {summary['latency_us_mean']:.2f} ± "
            f"{summary['latency_us_std']:.2f} us"
        )
        print(
            f"GPU util mean/peak: "
            f"{summary['gpu_util_mean_pct'] if summary['gpu_util_mean_pct'] is not None else 'n/a'}"
            f"/"
            f"{summary['gpu_util_peak_pct'] if summary['gpu_util_peak_pct'] is not None else 'n/a'} %"
        )

    primary_backend = "neurosim_cu_esim"
    primary = backend_results[primary_backend]
    primary_summary = primary["summary"]

    comparisons: dict[str, dict[str, float]] = {}
    for backend_name, result in backend_results.items():
        if backend_name == primary_backend:
            continue
        summary = result["summary"]
        comparisons[backend_name] = {
            "calls_khz_speedup_vs_backend": (
                primary_summary["calls_khz_mean"] / summary["calls_khz_mean"]
            ),
            "events_mev_per_sec_speedup_vs_backend": (
                primary_summary["events_mev_per_sec_mean"]
                / summary["events_mev_per_sec_mean"]
            ),
            "latency_speedup_vs_backend": (
                summary["latency_us_mean"] / primary_summary["latency_us_mean"]
            ),
        }

    if comparisons:
        print("\n=== Relative speedup (higher is better) ===")
        for backend_name, cmp in comparisons.items():
            print(
                f"vs {backend_name}: "
                f"calls {cmp['calls_khz_speedup_vs_backend']:.2f}x, "
                f"events/sec {cmp['events_mev_per_sec_speedup_vs_backend']:.2f}x, "
                f"latency {cmp['latency_speedup_vs_backend']:.2f}x"
            )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "device": {
            "torch_device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(gpu_index),
            "cuda_runtime": torch.version.cuda,
            "torch_version": torch.__version__,
            "gpu_index": gpu_index,
        },
        "trials": primary["trials"],
        "summary": primary_summary,
        "backends": backend_results,
        "comparisons": comparisons,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved benchmark report: {out_path}")


if __name__ == "__main__":
    main()
