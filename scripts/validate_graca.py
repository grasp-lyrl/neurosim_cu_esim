"""Single-pixel parity check for the CUDA Graca model vs the Python reference.

Replays the paper Fig. 5 pulse (10 fA -> 1 pA -> 10 fA) on a small array at the
same 10 us grid as the single-pixel reference (so n_sub = 1 per frame, i.e. the
CUDA kernel performs exactly one IIR step per forward — a 1:1 comparison), reads
the internal Vpr/Vsf out of sim.state each step, and compares to ref_trace.npz
(produced by model_final.simulate_full, float64).

Run on a CUDA machine:  python scripts/validate_graca.py [ref_trace.npz]
"""
import sys
import numpy as np
import torch

from neurosim_cu_esim import GracaDVSSimulator

S_VPR, S_VSF, S_VPRSF = 0, 3, 4  # indices into the packed state (see kernel GracaState)

ref_path = sys.argv[1] if len(sys.argv) > 1 else "ref_trace.npz"
ref = np.load(ref_path)
t, Ipd, Vpr_ref, Vsf_ref = ref["t"], ref["Ipd"], ref["Vpr"], ref["Vsf"]
N = len(t)

FULL_WELL_SATURATION_THRESHOLD = 1e-12
L = Ipd                       # 10 fA -> 10e-15,  1 pA -> 1e-12

H, W = 4, 4
sim = GracaDVSSimulator(
    width=W, height=H,
    Cpd=71.54e-15, Cfb=0.87e-15, Cpr=23.72e-15, Csf=581e-15, Ipr=3e-9, Isf=10e-12,
    full_well_saturation_threshold=FULL_WELL_SATURATION_THRESHOLD, dark_current=1e-15,
    contrast_threshold=0.3, refractory_us=100.0,
    add_noise=True, device="cuda",
)

Vpr_cuda = np.zeros(N)
Vsf_cuda = np.zeros(N)
n_on = n_off = 0
for n in range(N):
    frame = torch.full((H, W), float(L[n]), device="cuda", dtype=torch.float32)
    ts = int(round(t[n] * 1e6))         # microseconds (0, 10, 20, ...)
    if n == 0:
        sim.init(frame)
        sim._prev_time = ts
        continue
    ev = sim.forward(frame, ts)
    if ev is not None:
        n_on += int((ev.p == 1).sum().item())
        n_off += int((ev.p == 0).sum().item())
    st = sim.state
    Vpr_cuda[n] = float(st[S_VPRSF, 0, 0].item())
    Vsf_cuda[n] = float(st[S_VSF, 0, 0].item())

rmse_pr = np.sqrt(np.mean((Vpr_cuda - Vpr_ref) ** 2))
rmse_sf = np.sqrt(np.mean((Vsf_cuda - Vsf_ref) ** 2))
rel_pr = rmse_pr / Vpr_ref.max()
rel_sf = rmse_sf / Vsf_ref.max()
print(f"Vpr: peak ref={Vpr_ref.max():.5f}  cuda={Vpr_cuda.max():.5f}  "
      f"RMSE={rmse_pr*1e3:.4f} mV ({rel_pr*100:.3f}% of peak)")
print(f"Vsf: peak ref={Vsf_ref.max():.5f}  cuda={Vsf_cuda.max():.5f}  "
      f"RMSE={rmse_sf*1e3:.4f} mV ({rel_sf*100:.3f}% of peak)")
print(f"events on this pixel-stream: ON={n_on//(H*W)} OFF={n_off//(H*W)} (per pixel)")
np.savez("cuda_trace.npz", t=t, Vpr=Vpr_cuda, Vsf=Vsf_cuda)
print("saved cuda_trace.npz")

ok = rel_pr < 0.03 and rel_sf < 0.03
print("PARITY", "OK" if ok else "CHECK (>3% of peak)")
