"""
CPU mirror of the single-pixel arithmetic in csrc/evsim_graca_kernel.cu, used to
verify the stochastic first-passage-time (FPT) event-generation logic WITHOUT a GPU
(the .cu cannot be compiled in every environment).

It reproduces, exactly, the kernel's per-sub-step recurrences:
  * bilinear2 / bilinear1 coefficient transforms,
  * Zm signal accumulation + Asf source follower,
  * Zm/Zout/Zoutsf shot-noise filters,
  * the parallel noise-only SF that isolates the Vsf-noise increment,
  * the MSI (mean-square Vsf-noise increment) self-calibration carried across frames,
  * the FPT detector  P=exp(-2 (theta-dp)(theta-d)/MSI)  with two Bernoulli trials.

Under constant illumination it compares, vs a fine-dt ground truth, the noise-event
rate of the plain v2e detector (underestimates as dt grows) and the FPT detector
(recovers the truth) -- the kernel-level analogue of the single-pixel
`stochastic_demo.py`. Run: python scripts/graca_fpt_cpu_check.py
"""
import numpy as np

Q = 1.602176634e-19


def bilinear2(nb1, nb0, da2, da1, K0):
    B0 = nb1*K0 + nb0; B1 = 2.0*nb0; B2 = -nb1*K0 + nb0
    A0 = da2*K0*K0 + da1*K0 + 1.0
    A1 = -2.0*da2*K0*K0 + 2.0
    A2 = da2*K0*K0 - da1*K0 + 1.0
    return B0/A0, B1/A0, B2/A0, A1/A0, A2/A0


def bilinear1(nb0, da1, K0):
    A0 = da1*K0 + 1.0
    return nb0/A0, nb0/A0, (1.0 - da1*K0)/A0


def run(intensity, dt_us, n_frames, frame_us, theta, p, rng, stochastic, add_noise=True):
    """Single constant-illumination pixel; returns events per second."""
    Cpd, Cfb, Cpr, Csf, Ipr, Isf = p['Cpd'], p['Cfb'], p['Cpr'], p['Csf'], p['Ipr'], p['Isf']
    kfb, ksf, VA, UT = p['kappa_fb'], p['kappa_sf'], p['VA'], p['UT']
    full_well_saturation_threshold, dark_current = p['full_well_saturation_threshold'], p['dark_current']
    refr = p['refractory_us']

    Ipd = min(max(intensity, dark_current), full_well_saturation_threshold)
    Ts = dt_us*1e-6; K0 = 2.0/Ts
    gs_fb = Ipd/UT; gm_fb = kfb*Ipd/UT; gm_amp = kfb*Ipr/UT
    Rout = VA/(2*Ipr); Aloop = gm_amp*Rout*gm_fb/gs_fb; Rin = 1/gs_fb
    da2 = (Rout*(Cpd*Cfb+Cpr*Cfb+Cpr*Cpd))/(gs_fb*(Aloop+1))
    da1 = (Rin*(Cpd+(1+gm_amp*Rout)*Cfb)+Rout*(Cpr+(1-gm_fb*Rin)*Cfb))/(Aloop+1)
    Zm_dc = (1/gm_fb)*(Aloop/(Aloop+1)); wz_zm = -gm_amp/Cfb
    zb0, zb1, zb2, za1, za2 = bilinear2(Zm_dc/wz_zm, Zm_dc, da2, da1, K0)
    gs_sf = Isf/UT; tau_sf = Csf/gs_sf
    sb0, sb1, sa1 = bilinear1(ksf, tau_sf, K0)
    Zout_dc = Rout/(Aloop+1); wz_zo = -gs_fb/(Cpd+Cfb)
    ob0, ob1, ob2, oa1, oa2 = bilinear2(Zout_dc/wz_zo, Zout_dc, da2, da1, K0)
    nsb0, nsb1, nsa1 = bilinear1(1/gs_sf, tau_sf, K0)
    std_ipd = np.sqrt(4*Q*Ipd/(2*Ts)); std_ipr = np.sqrt(4*Q*Ipr/(2*Ts))
    std_isf = np.sqrt(4*Q*Isf/(2*Ts))

    # state
    Vpr=dvpr1=dvpr2=xpr1=xpr2=0.0
    Vsf=vsfc=vprsf=Vref=tse=0.0
    zmx1=zmx2=zmy1=zmy2=zox1=zox2=zoy1=zoy2=zsx1=zsy1=0.0
    vsfn=vprsfn=0.0; msi_stored=0.0; vsf_noise_prev=0.0
    use_fpt = stochastic and add_noise
    n_ev = 0
    n_sub = int(round(frame_us/dt_us))
    for f in range(n_frames):
        vsf_prev = Vsf
        msi_sum = 0.0; msi_cnt = 0
        A = 0.0   # constant illumination -> no log-current step after frame 0
        for i in range(n_sub):
            xin = A if i == 0 else 0.0
            dvpr = zb0*xin + zb1*xpr1 + zb2*xpr2 - za1*dvpr1 - za2*dvpr2
            xpr2=xpr1; xpr1=xin; dvpr2=dvpr1; dvpr1=dvpr
            Vpr += dvpr
            vpr_total = Vpr
            if add_noise:
                w_ipd = std_ipd*rng.standard_normal(); w_ipr = std_ipr*rng.standard_normal()
                zmy = zb0*w_ipd + zb1*zmx1 + zb2*zmx2 - za1*zmy1 - za2*zmy2
                zmx2=zmx1; zmx1=w_ipd; zmy2=zmy1; zmy1=zmy
                zoy = ob0*w_ipr + ob1*zox1 + ob2*zox2 - oa1*zoy1 - oa2*zoy2
                zox2=zox1; zox1=w_ipr; zoy2=zoy1; zoy1=zoy
                vpr_total = Vpr + zmy + zoy
            vsfc = sb0*vpr_total + sb1*vprsf - sa1*vsfc   # clean Asf feedback
            vprsf = vpr_total
            vsf = vsfc
            if add_noise:
                w_isf = std_isf*rng.standard_normal()
                zsy = nsb0*w_isf + nsb1*zsx1 - nsa1*zsy1
                zsx1=w_isf; zsy1=zsy
                vsf = vsfc + zsy
                if use_fpt:
                    vpr_noise = vpr_total - Vpr
                    vsfn = sb0*vpr_noise + sb1*vprsfn - sa1*vsfn
                    vprsfn = vpr_noise
                    vsf_noise = vsfn + zsy
                    dvn = vsf_noise - vsf_noise_prev
                    msi_sum += dvn*dvn; msi_cnt += 1
                    vsf_noise_prev = vsf_noise
            Vsf = vsf
            tse += dt_us
            if tse >= refr:
                d = Vsf - Vref; dp = vsf_prev - Vref; pol = -1
                if d >= theta: pol = 1
                elif d <= -theta: pol = 0
                elif use_fpt and msi_stored > 0.0:
                    if dp < theta and rng.random() < np.exp(-2*(theta-dp)*(theta-d)/msi_stored):
                        pol = 1
                    elif dp > -theta and rng.random() < np.exp(-2*(dp+theta)*(d+theta)/msi_stored):
                        pol = 0
                if pol >= 0:
                    n_ev += 1; Vref = Vsf; tse = 0.0
            vsf_prev = Vsf
        if use_fpt:
            msi_stored = (msi_sum/msi_cnt) if msi_cnt > 0 else msi_stored
    total_s = n_frames*n_sub*dt_us*1e-6
    return n_ev/total_s


if __name__ == "__main__":
    p = dict(Cpd=71.54e-15, Cfb=0.87e-15, Cpr=23.72e-15, Csf=581e-15, Ipr=3e-9, Isf=10e-12,
             kappa_fb=0.7, kappa_sf=0.7, VA=3.0, UT=25.8e-3,
             full_well_saturation_threshold=1e-12, dark_current=10e-15, refractory_us=1.0)
    intensity = 1e-14                    # -> Ipd ~ 10 fA (with dark_current floor)
    TC = 0.07
    theta = TC*p['kappa_sf']*p['UT']/p['kappa_fb']
    sigma_hint = None

    # ground truth: fine sub-step
    FR = 60000.0   # frame interval (us); keeps n_sub>=30 even at dt=2ms
    truth = run(intensity, 5.0, 400, FR, theta, p, np.random.default_rng(1), stochastic=False)
    print(f"theta = {theta*1e3:.3f} mV ; ground-truth rate (dt=5us) = {truth:.2f} ev/s\n")
    print(" dt_us   dt/tau_noise   v2e(ev/s)   FPT(ev/s)")
    tau_noise = 12.1e3  # us (Vsf-noise correlation time at 10 fA)
    for dt in [5.0, 50.0, 200.0, 500.0, 1000.0, 2000.0]:
        r_na = run(intensity, dt, 400, FR, theta, p, np.random.default_rng(2), stochastic=False)
        r_fp = run(intensity, dt, 400, FR, theta, p, np.random.default_rng(2), stochastic=True)
        print(f" {dt:6.0f}   {dt/tau_noise:11.3f}   {r_na:8.2f}    {r_fp:8.2f}")
    print("\n(v2e underestimates as dt grows toward tau_noise; FPT stays near ground truth)")
