# neurosim_cu_esim

A high-performance CUDA implementation of the frame-differencing event generation algorithm written for the [Neurosim simulator](https://github.com/grasp-lyrl/neurosim). 
It is implemented as a single fused CUDA kernel with warp-level aggregation, making it **~10× faster** than reference Pytorch/CUDA implementations.

<p align="center">
    <img src="assets/example.gif" alt="Example output — moving texture stimulus and generated events (20 ms aggregation)" style="width:80%;" />
    <br/>
    <em>Example output: input frame (left) with events aggregated for 20 ms (right). See <a href="#benchmarking">Benchmarking</a> for reproduction.</em>
</p>

---

**Performance metrics on an RTX 4090 for a 640×480 moving texture stimulus:**

| Metric | Value |
|--------|-------|
| Calls/sec | 47.23 kHz |
| Events/call | 18016.82 |
| Events/sec | 850.01 Mev/s |
| Forward latency | 21.20 µs |
| Peak GPU util | 36% |

✨ neurosim_cu_esim achieves **~11× better throughput** and **~10× lower latency** than [rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e) esim CUDA implementation.

> **Calls/sec** measures how many frames can be processed per second.
> **Events/call** measures the total number of events generated per frame. Both quantities are very data specific.

## How it works

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

PyPI package coming soon!

## Installation

```bash
# Clone and install
git clone https://github.com/grasp-lyrl/neurosim_cu_esim.git
cd neurosim_cu_esim
pip install .

# Editable install with dev dependencies (pytest, ruff)
pip install -e ".[dev]"
```

> **Note:** Make sure `nvcc` is on your `PATH` and its version is compatible with the CUDA version that your PyTorch was built against. You can check with:
> ```bash
> python -c "import torch; print(torch.version.cuda)"
> nvcc --version
> ```

## Quick start

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

### Runtime threshold

```python
sim.set_contrast_thresholds(neg=0.2, pos=0.5)
```

### Diagnostics

```python
print(sim.state)                  # internal intensity bounds
print(sim.buffer_memory_bytes)    # GPU memory used by output buffers
```
## API reference

### `EventSimulator(width, height, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | `int` | — | Sensor width in pixels |
| `height` | `int` | — | Sensor height in pixels |
| `contrast_threshold_neg` | `float` | `0.35` | Negative contrast threshold (log scale) |
| `contrast_threshold_pos` | `float` | `0.35` | Positive contrast threshold (log scale) |
| `max_events` | `int \| None` | `W × H` | Cap on events per frame |
| `device` | `str` | `"cuda"` | CUDA device |

### `EventSimulator.forward(image, timestamp_us) -> Events | None`

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `torch.Tensor` | Grayscale `(H, W)` frame, positive values |
| `timestamp_us` | `int` | Frame timestamp in microseconds |

Returns a named tuple `Events(x, y, t, p)` or `None` if zero events.

### `EventSimulator.init(first_image)` / `EventSimulator.reset(first_image=None)`

Initialize or reset internal state.

## Benchmarking

Run the benchmark suite to measure throughput and utilization:

```bash
python3 scripts/benchmark_esim.py
```

Optional sanity-check animation (frame + 20 ms aggregated events):

```bash
python3 scripts/benchmark_esim.py --sanity-video benchmarks/sanity_20ms.mp4
```

Reported metrics include:

- forward calls per second (kHz)
- events generated per second (Mev/s)
- events per call
- average forward latency (CUDA event timing)
- mean/peak GPU utilization (via `nvidia-smi` polling)

Results are saved to:

```text
benchmarks/esim_benchmark_results.json
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
