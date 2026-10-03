# neurosim_cu_esim

High-performance CUDA implementations of event camera simulation algorithms written for the [Neurosim simulator](https://github.com/grasp-lyrl/neurosim).
Algorithms are implemented as fused CUDA kernels with warp-level aggregation.

✨ EventSimulator mode achieves **~11× better throughput** and **~10× lower latency** than [rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e) esim CUDA implementation. Check [Quickstart](#quick-start)

🚀 VoltmeterSimulator mode achieves **~240×–650× faster** than [Lin et al., *DVS-Voltmeter*, ECCV 2022](https://github.com/Lynn0306/DVS-Voltmeter) (~20,000 calls/s vs. 31 calls/s CPU / 84 calls/s GPU). Check [DVS-Voltmeter](#dvs-voltmeter-stochastic-model)


<p align="center">
    <img src="assets/example.gif" alt="Example output — moving texture stimulus and generated events (20 ms aggregation)" style="width:90%;" />
    <br/>
    <em>Example output: input frame (left) with events aggregated for 20 ms (right). See <a href="#benchmarking">Benchmarking</a> for reproduction.</em>
</p>

<p align="center">
    <img src="assets/voltmeter_real_video_randphase.gif" alt="Example event simulation using DVSVoltmeterSimulator on a real video" style="width:90%;" />
    <br/>
    <em>Example event simulation using DVSVoltmeterSimulator on a real video</em>
</p>

**Forward latency at 640×480 on an RTX 4090:** `single` ~21 µs · `multi` ~27 µs · `voltmeter` ~29 µs — all **much faster** than the corresponding reference implementations. See [Benchmarking](#benchmarking) for details.

## Contents

- [neurosim\_cu\_esim](#neurosim_cu_esim)
  - [Contents](#contents)
  - [How it works (Single event mode)](#how-it-works-single-event-mode)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Quick start](#quick-start)
    - [Multi-event mode](#multi-event-mode)
  - [DVS-Voltmeter (stochastic model)](#dvs-voltmeter-stochastic-model)
  - [API reference](#api-reference)
    - [`EventSimulator(width, height, ...)`](#eventsimulatorwidth-height-)
    - [`EventSimulator.forward(image, timestamp_us) -> Events | None`](#eventsimulatorforwardimage-timestamp_us---events--none)
    - [`EventSimulator.init(first_image)` / `EventSimulator.reset(first_image=None)`](#eventsimulatorinitfirst_image--eventsimulatorresetfirst_imagenone)
    - [Runtime threshold](#runtime-threshold)
    - [Diagnostics](#diagnostics)
  - [Benchmarking](#benchmarking)
    - [**Sanity animation** (frame + aggregated events MP4):](#sanity-animation-frame--aggregated-events-mp4)
    - [**Run on a real video**](#run-on-a-real-video)
  - [Running tests](#running-tests)
  - [Linting](#linting)
  - [Citation](#citation)
  - [Issues](#issues)
  - [License](#license)

## How it works (Single event mode)

| Step | Description |
|------|-------------|
| 1 | Maintain per-pixel upper/lower bounds in **log-intensity** space. |
| 2 | For each new frame, compute `log(pixel)` and compare against bounds. |
| 3 | If the log-intensity exceeds the upper bound → **positive event** (polarity 1). |
| 4 | If it drops below the lower bound → **negative event** (polarity 0). |
| 5 | On event: reset bounds around current value. No event: tighten bounds. |

All five steps execute in a single kernel launch.

## Requirements

| Dependency | Minimum version |
|------------|-----------------|
| Python | 3.9 |
| PyTorch | 2.0 |
| CUDA toolkit | 11.8 |

## Installation

The CUDA kernel is compiled from source at install time against your installed PyTorch. Make sure `nvcc` is on your `PATH` and compatible with the CUDA version your PyTorch was built against:

```bash
python -c "import torch; print(torch.version.cuda)"  # torch's CUDA
nvcc --version                                       # toolkit CUDA
```

Then install:

```bash
git clone https://github.com/grasp-lyrl/neurosim_cu_esim.git
cd neurosim_cu_esim
pip install .

# Editable install with dev dependencies (pytest, ruff)
pip install -e ".[dev]"
```

## Quick start

> **Input format.** All simulators take **linear-intensity** frames, positive values. `EventSimulator` is scale-invariant — any positive range works (`[0, 1]`, `[0, 255]`, ...); the kernel applies `log()` internally. `DVSVoltmeterSimulator` expects **0–255** natively (its `k` params are calibrated to that scale), or pass `input_normalized=True` to feed it `[0, 1]` directly.

```python
import torch
from neurosim_cu_esim import EventSimulator

# Create the simulator
sim = EventSimulator(
    width=640,
    height=480,
    contrast_threshold_neg=0.3,   # log-intensity units
    contrast_threshold_pos=0.3,
    device="cuda",
)

# Feed the first frame (initialises internal state, returns None)
first_frame = torch.rand(480, 640, device="cuda").clamp(min=1e-4)
sim.init(first_frame)

# Process subsequent frames
for i in range(1, 100):
    frame = torch.rand(480, 640, device="cuda").clamp(min=1e-4)
    events = sim(frame, timestamp_us=i * 1000)

    if events is not None:
        print(f"Frame {i}: {events.x.numel()} events")
        # events.x  — column coords  (uint16, CUDA)
        # events.y  — row coords     (uint16, CUDA)
        # events.t  — timestamps µs  (uint64, CUDA)
        # events.p  — polarity 0/1   (uint8,  CUDA)
```

**For a complete example with a moving texture stimulus, see:** [Benchmark with animation](#benchmarking).

### Multi-event mode

By default the simulator emits **at most one event per pixel per frame**
(`mode="single"`), which is the right choice for high-fps inputs where each frame step spans at most one contrast threshold.

Set `mode="multi"` to emit **as many events as the
log-contrast change warrants**, with their timestamps spread *equally* across
the interval between the previous and current frame (the last event lands
exactly on the current `timestamp_us`):

```python
sim = EventSimulator(
    width=640, height=480,
    mode="multi",
    max_events=640 * 480 * 32,   # a frame step can emit many events/pixel
)
```

In `"multi"` mode size `max_events` for the expected per-frame burst; events
beyond the cap are dropped (with a warning).

## DVS-Voltmeter (stochastic model)

A separate, **stochastic** event simulator based on [Lin et al., *DVS-Voltmeter*, ECCV 2022](https://github.com/Lynn0306/DVS-Voltmeter). Each pixel's sensor voltage is modelled as **Brownian motion with drift** (paper Eq. 10/11), and events are sampled at threshold crossings. A calibrated **leakage current** produces background ON events.

Input is **linear intensity** (0–255 — the scale the `k` params are calibrated to), not log:

```python
from neurosim_cu_esim import DVSVoltmeterSimulator

sim = DVSVoltmeterSimulator(
    width=640, height=480,
    camera_type="DVS346",     # or "DVS240"; or pass k=[k1..k6] explicitly
    sampler="exact",          # default; or "reference" (see below)
    randomize_phase=True,     # random per-pixel leakage phase
                              # (avoids synchronised background flashes)
    leak_scale=1.0,           # 1.0=faithful; lower to suppress bg ON events
    seed=0, device="cuda",
)
events = sim(frame_0_255_float.cuda(), timestamp_us)
```

**`sampler`.** `"exact"` (the default) carries each pixel's voltage across frames and advances it with the exact law of the model's SDE, testing for crossings with a Brownian bridge. Its event rates, ON/OFF split and timing match closed-form Brownian motion at any frame interval from 100 µs to 1 s. It integrates the signal drift as `k1·ln((L1+k2)/(L0+k2))`, so a brightness step gives the same events however many frames it is split across. It is also faster than `"reference"` on the benchmark below (~1.5% on the 4090, ~7% on the 4070 Laptop), because it draws its random numbers statelessly, one Philox block per step. `"reference"` reproduces the paper's Algorithm 1 and the upstream code's statistics; `scripts/voltmeter_parity.py` checks it against upstream. It throws away the voltage's random walk at every frame boundary, so the noise shrinks as the frame rate grows. At 1 kHz, timestamp jitter is 5–20× below what the model itself predicts, and noise strong enough to give 27% OFF events gives none. It also times crossings against the drift (OFF during brightening or leakage) about 2× too early.

## API reference

### `EventSimulator(width, height, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | `int` | — | Sensor width in pixels |
| `height` | `int` | — | Sensor height in pixels |
| `contrast_threshold_neg` | `float` | `0.35` | Negative contrast threshold (log scale) |
| `contrast_threshold_pos` | `float` | `0.35` | Positive contrast threshold (log scale) |
| `max_events` | `int \| None` | `W × H` | Cap on events per frame |
| `mode` | `str` | `"single"` | `"single"` = ≤1 event/pixel/frame; `"multi"` = many events/pixel with timestamps spread across the inter-frame interval |
| `device` | `str` | `"cuda"` | CUDA device |

### `EventSimulator.forward(image, timestamp_us) -> Events | None`

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `torch.Tensor` | Grayscale `(H, W)` frame, positive values |
| `timestamp_us` | `int` | Frame timestamp in microseconds |

Returns a named tuple `Events(x, y, t, p)` or `None` if zero events.

### `EventSimulator.init(first_image)` / `EventSimulator.reset(first_image=None)`

Initialize or reset internal state.

### Runtime threshold

```python
sim.set_contrast_thresholds(neg=0.2, pos=0.5)
```

### Diagnostics

```python
print(sim.state)                  # internal intensity bounds
print(sim.buffer_memory_bytes)    # GPU memory used by output buffers
```

## Benchmarking

```bash
python3 scripts/benchmark_esim.py                                     # ESIM single (default)
python3 scripts/benchmark_esim.py --mode multi                        # ESIM multi
python3 scripts/benchmark_esim.py --mode voltmeter --randomize-phase  # DVS-Voltmeter
```

**Reported metrics:** calls/sec (kHz), events/sec (Mev/s), events/call, mean forward latency (CUDA event timing), mean/peak GPU utilization (`nvidia-smi` polling). Saved to `benchmarks/esim_benchmark_results.json`.

**Throughput on an RTX 4090** (640×480, 1000 fps timestamps, fp32, 3 trials × 1 M forwards; voltmeter row with the default `exact` sampler, `reference` measured ~1.5% slower back to back):

| mode | calls/s | latency | events/call | events/sec |
|------|--------:|--------:|------------:|-----------:|
| `single` — ESIM, ≤1 event/pixel/frame (default) | 48.7 kHz | 21 µs | 18 017 | 878 Mev/s |
| `multi` — ESIM, many events/pixel | 37.2 kHz | 27 µs | 21 668 | 806 Mev/s |
| `voltmeter` — DVS-Voltmeter stochastic | 34.0 kHz | 29 µs | 18 364 | 625 Mev/s |

**Throughput on an RTX 4070 Laptop** (640×480, 1000 fps timestamps, fp32, 3 trials × 200 k forwards; voltmeter row with the default `exact` sampler, `reference` measured ~7% slower back to back):

| mode | calls/s | latency | events/call | events/sec |
|------|--------:|--------:|------------:|-----------:|
| `single` — ESIM, ≤1 event/pixel/frame (default) | 38.3 kHz | 26 µs | 18 017 | 690 Mev/s |
| `multi` — ESIM, many events/pixel | 28.1 kHz | 36 µs | 21 668 | 608 Mev/s |
| `voltmeter` — DVS-Voltmeter stochastic | 20.1 kHz | 50 µs | 18 364 | 370 Mev/s |

The voltmeter's latency is ~1.1–1.4× ESIM's on the 4090 and ~1.4–1.9× on the 4070 (per-pixel RNG + threshold-crossing sampling), but it still runs at tens of kHz at VGA — far above the reference PyTorch implementation (~84 Hz GPU-patched, ~31 Hz CPU).

### **Sanity animation** (frame + aggregated events MP4):

Save the video of event simulation on the moving texture stimulus.

```bash
python3 scripts/benchmark_esim.py --sanity-video sanity.mp4 \
    --mode voltmeter --randomize-phase
```

### **Run on a real video**

Run any mode; side-by-side frame | events MP4, on a real video.

```bash
python3 scripts/simulate_on_video.py --input in.mp4 --output out.mp4 \
    --mode voltmeter --randomize-phase
```

## Running tests

```bash
pip install -e ".[dev]"
conda run -n esim pytest
```

## Linting

```bash
ruff check .
ruff format --check .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{das2026neurosim,
      title={Neurosim: A Fast Simulator for Neuromorphic Robot Perception}, 
      author={Richeek Das and Pratik Chaudhari},
      year={2026},
      eprint={2602.15018},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.15018}, 
}
```

and checkout [Neurosim](https://github.com/grasp-lyrl/neurosim) for the full simulator codebase.

## Issues

Please report any bugs or feature requests on GitHub issues. Pull requests are also welcome!

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
