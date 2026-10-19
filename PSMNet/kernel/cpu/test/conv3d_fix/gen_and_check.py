#!/usr/bin/env python3
"""
Validate ops/common/conv3d_fix_kernel.hpp against a PyTorch reference.

Two independent checks:

  A. util::fix() over a dense float sweep, so the rounding/saturation
     primitive is pinned down before it is used inside a convolution.

  B. Whole-op comparison for a matrix of conv3d-fix / transposed-conv3d-fix
     configurations (axis-distinct kernels/strides/dilations/pads, FLOOR and
     CEIL, bias present and absent, RELU / RELU6 / NONE, saturation stress,
     and a thread-count determinism pair). Every comparison is bit-exact on
     the int8 output.

XIR conventions under test (see ops/common/conv3d_fix_kernel.hpp):
  tensors [N,H,W,D,C]; weights [OC,KH,KW,KD,IC]
  kernel/stride/dilation ordered {W,H,D}; pad ordered {l,r,t,b,near,far}

The transposed reference flips the kernel over (KH,KW,KD) because XIR stores
transposed-conv weights mirrored relative to torch.conv_transpose3d.

  usage: python3 gen_and_check.py [--keep]
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "test_conv3d_fix")
CASES = os.path.join(HERE, "cases")

# ---------------------------------------------------------------- fix() ------


def fix_np(v):
    """Bit-exact numpy transcription of vitis::ai::cpu_task::util::fix()."""
    v = np.asarray(v, dtype=np.float32)
    fl = np.floor(v).astype(np.float32)
    frac = (v - fl).astype(np.float32)
    # C round(): half away from zero
    away = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.float32)
    r = np.where((v < 0) & (frac == np.float32(0.5)), np.ceil(v).astype(np.float32), away)
    # the clamp is tested first in the C++ source, so it wins
    r = np.where(v > np.float32(127.0), np.float32(127.0), r)
    r = np.where(v < np.float32(-128.0), np.float32(-128.0), r)
    return r.astype(np.int32)


def check_fix():
    vals = [
        np.arange(-140.0, 140.0, 0.0625, dtype=np.float32),           # dense, hits every .5
        np.arange(-3.0, 3.0, 1.0 / 512, dtype=np.float32),            # fine grid near zero
        np.array([-128.5, -128.0, -127.5, 126.5, 127.0, 127.5, 128.0,
                  0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5,
                  -0.4999999, 0.4999999], dtype=np.float32),
    ]
    v = np.concatenate(vals).astype(np.float32)
    in_p = os.path.join(HERE, "_fix_in.f32")
    out_p = os.path.join(HERE, "_fix_out.i32")
    v.tofile(in_p)
    subprocess.run([EXE, "fix", in_p, out_p], check=True)
    got = np.fromfile(out_p, dtype=np.int32)
    exp = fix_np(v)
    os.remove(in_p)
    os.remove(out_p)
    bad = np.nonzero(got != exp)[0]
    print(f"[A] util::fix over {v.size} floats: "
          + ("OK" if bad.size == 0 else f"FAIL on {bad.size}"))
    if bad.size:
        for i in bad[:10]:
            print(f"      x={v[i]!r} cpp={got[i]} py={exp[i]}")
        return False
    return True


# ------------------------------------------------------------- reference -----

# axis order used everywhere below: torch dims (d1,d2,d3) == XIR (H,W,D)
def hwd(attr3):
    """{W,H,D} attribute -> (H,W,D) tuple for torch."""
    return (attr3[1], attr3[0], attr3[2])


def ref_acc_forward(x, w, stride, dilation, pad, out_shape):
    """Exact integer accumulator of conv3d-fix, as float64. -> [N,OH,OW,OD,OC]"""
    sH, sW, sD = hwd(stride)
    dH, dW, dD = hwd(dilation)
    pl, pr, pt, pb, pn, pf = pad
    N, H, W, D, IC = x.shape
    OC, KH, KW, KD, _ = w.shape
    OH, OW, OD = out_shape[1], out_shape[2], out_shape[3]

    # Pad to exactly the extent the last output position reads. Anything past
    # the real data (including the CEIL/SAME ragged tap) is zero, which is what
    # the kernel's bounds check produces.
    need = ((OH - 1) * sH + dH * (KH - 1) + 1,
            (OW - 1) * sW + dW * (KW - 1) + 1,
            (OD - 1) * sD + dD * (KD - 1) + 1)
    beg = (pt, pl, pn)
    end = tuple(max(0, need[i] - (beg[i] + s)) for i, s in enumerate((H, W, D)))

    xt = torch.from_numpy(x.transpose(0, 4, 1, 2, 3).astype(np.float64))
    # F.pad on 5D takes (d3_l,d3_r, d2_l,d2_r, d1_l,d1_r)
    xt = F.pad(xt, (beg[2], end[2], beg[1], end[1], beg[0], end[0]))
    wt = torch.from_numpy(w.transpose(0, 4, 1, 2, 3).astype(np.float64))
    a = F.conv3d(xt, wt, bias=None, stride=(sH, sW, sD), padding=0,
                 dilation=(dH, dW, dD)).numpy()
    assert a.shape == (N, OC, OH, OW, OD), (a.shape, (N, OC, OH, OW, OD))
    return a.transpose(0, 2, 3, 4, 1)


def ref_acc_transposed(x, w, stride, dilation, pad, out_shape):
    """Exact integer accumulator of transposed-conv3d-fix. -> [N,OH,OW,OD,OC]"""
    sH, sW, sD = hwd(stride)
    dH, dW, dD = hwd(dilation)
    pl, pr, pt, pb, pn, pf = pad
    N = x.shape[0]
    OC = w.shape[0]
    OH, OW, OD = out_shape[1], out_shape[2], out_shape[3]

    xt = torch.from_numpy(x.transpose(0, 4, 1, 2, 3).astype(np.float64))
    wf = w[:, ::-1, ::-1, ::-1, :].copy()          # XIR stores it mirrored
    wt = torch.from_numpy(wf.transpose(4, 0, 1, 2, 3).astype(np.float64))
    full = F.conv_transpose3d(xt, wt, bias=None, stride=(sH, sW, sD), padding=0,
                             output_padding=0, dilation=(dH, dW, dD)).numpy()
    FH, FW, FD = full.shape[2], full.shape[3], full.shape[4]
    # XIR crops pad from each side of the full transposed extent
    a = full[:, :, pt:FH - pb, pl:FW - pr, pn:FD - pf]
    assert a.shape == (N, OC, OH, OW, OD), (a.shape, (N, OC, OH, OW, OD))
    return a.transpose(0, 2, 3, 4, 1)


def epilogue(acc, bias, in_fp, w_fp, out_fp, bias_fp, nonlinear):
    """int32 accumulator -> int8, replicating the kernel's float32 epilogue."""
    shift_cut_scale = np.float32(2.0 ** (in_fp + w_fp - out_fp))
    six = np.float32(6.0 * (2.0 ** out_fp))
    filout = acc.astype(np.float32)
    if bias is not None:
        sbs = np.float32(2.0 ** (in_fp + w_fp - bias_fp))
        filout = (filout + bias.astype(np.float32) * sbs).astype(np.float32)
    v = (filout / shift_cut_scale).astype(np.float32)
    if nonlinear == "RELU":
        v = np.where(v < 0, np.float32(0.0), v).astype(np.float32)
    elif nonlinear == "LEAKYRELU":
        # kernel multiplies by the double literal 0.01, then narrows to float
        v = np.where(v < 0, (v.astype(np.float64) * 0.01).astype(np.float32), v)
    elif nonlinear == "RELU6":
        v = np.maximum(np.float32(0.0), np.minimum(v, six)).astype(np.float32)
    elif nonlinear in ("NONE", ""):
        pass
    else:
        raise AssertionError("unhandled nonlinear " + nonlinear)
    return fix_np(v).astype(np.int8), v


def tie_stats(v):
    """How many pre-rounding values land exactly on .5 (the DPU_ROUND path).

    shift_cut_scale is a power of two and the accumulator is an integer, so
    halves occur exactly rather than approximately; the negative ones take
    util::fix()'s ceil() branch instead of round-half-away.
    """
    v = np.asarray(v, dtype=np.float32)
    inrange = (v <= np.float32(127.0)) & (v >= np.float32(-128.0))
    half = ((v - np.floor(v)) == np.float32(0.5)) & inrange
    return int(np.count_nonzero(half & (v < 0))), int(np.count_nonzero(half & (v > 0)))



# ------------------------------------------------------------- case runner ---


def out_shape_forward(in_shape, kernel, stride, dilation, pad, pad_mode):
    """xir::cal_out_3d"""
    import math
    rnd = math.floor if pad_mode in ("FLOOR",) else math.ceil
    out = [in_shape[0], 0, 0, 0, None]
    # axis a in (H,W,D) -> tensor dim 1+a, attr idx, pad begin/end idx
    for a, (tdim, ai, pbeg, pend) in enumerate(
            ((1, 1, 2, 3), (2, 0, 0, 1), (3, 2, 4, 5))):
        i, k, s, d = in_shape[tdim], kernel[ai], stride[ai], dilation[ai]
        p = pad[pbeg] + pad[pend]
        if pad_mode in ("FLOOR", "CEIL"):
            o = rnd(1.0 * (i + p - ((k - 1) * d + 1) + s) / s)
        elif pad_mode == "SAME":
            o = math.ceil(1.0 * (i + p) / s)
        elif pad_mode == "VALID":
            o = math.ceil(1.0 * (i + p - (k - 1) * d) / s)
        else:
            raise AssertionError(pad_mode)
        out[1 + a] = int(o)
    return out


def out_shape_transposed(in_shape, kernel, stride, dilation, pad):
    out = [in_shape[0], 0, 0, 0, None]
    for a, (tdim, ai, pbeg, pend) in enumerate(
            ((1, 1, 2, 3), (2, 0, 0, 1), (3, 2, 4, 5))):
        i, k, s, d = in_shape[tdim], kernel[ai], stride[ai], dilation[ai]
        out[1 + a] = (i - 1) * s + d * (k - 1) + 1 - pad[pbeg] - pad[pend]
    return out


def run_case(name, cfg, rng):
    transposed = cfg["transposed"]
    kernel = cfg["kernel"]                 # {W,H,D}
    stride = cfg["stride"]
    dilation = cfg.get("dilation", [1, 1, 1])
    pad = cfg["pad"]
    pad_mode = cfg.get("pad_mode", "FLOOR")
    nonlinear = cfg.get("nonlinear", "NONE")
    threads = cfg.get("threads", 4)
    N, H, W, D, IC = cfg["in"]
    OC = cfg["oc"]
    in_fp, w_fp, out_fp, bias_fp = cfg["fp"]
    has_bias = cfg.get("bias", True)

    in_shape = [N, H, W, D, IC]
    KH, KW, KD = kernel[1], kernel[0], kernel[2]
    weight_shape = [OC, KH, KW, KD, IC]
    if transposed:
        out_shape = out_shape_transposed(in_shape, kernel, stride, dilation, pad)
    else:
        out_shape = out_shape_forward(in_shape, kernel, stride, dilation, pad, pad_mode)
    out_shape[4] = OC
    assert all(d > 0 for d in out_shape), (name, out_shape)

    lo, hi = cfg.get("range", (-64, 64))
    x = rng.integers(lo, hi, size=in_shape, dtype=np.int64).astype(np.int8)
    w = rng.integers(-32, 32, size=weight_shape, dtype=np.int64).astype(np.int8)
    b = rng.integers(-100, 100, size=(OC,), dtype=np.int64).astype(np.int8) if has_bias else None

    d = os.path.join(CASES, name)
    os.makedirs(d, exist_ok=True)
    x.tofile(os.path.join(d, "input.bin"))
    w.tofile(os.path.join(d, "weight.bin"))
    if b is not None:
        b.tofile(os.path.join(d, "bias.bin"))
    with open(os.path.join(d, "case.txt"), "w") as f:
        f.write(f"transposed {int(transposed)}\n")
        f.write("kernel " + " ".join(map(str, kernel)) + "\n")
        f.write("stride " + " ".join(map(str, stride)) + "\n")
        f.write("dilation " + " ".join(map(str, dilation)) + "\n")
        f.write("pad " + " ".join(map(str, pad)) + "\n")
        f.write(f"pad_mode {pad_mode}\ngroup 1\n")
        f.write("input_shape " + " ".join(map(str, in_shape)) + "\n")
        f.write("weight_shape " + " ".join(map(str, weight_shape)) + "\n")
        f.write("output_shape " + " ".join(map(str, out_shape)) + "\n")
        f.write(f"nonlinear {nonlinear}\n")
        f.write(f"input_fp {in_fp}\nweight_fp {w_fp}\noutput_fp {out_fp}\n")
        f.write(f"bias_fp {bias_fp}\nhas_bias {int(has_bias)}\nthreads {threads}\n")

    subprocess.run([EXE, "conv", d], check=True)
    got = np.fromfile(os.path.join(d, "got.bin"), dtype=np.int8).reshape(out_shape)

    ref = ref_acc_transposed if transposed else ref_acc_forward
    acc = ref(x, w, stride, dilation, pad, out_shape)
    exp, pre = epilogue(acc, b, in_fp, w_fp, out_fp, bias_fp, nonlinear)

    ok = np.array_equal(got, exp)
    sat = float(np.mean((exp == 127) | (exp == -128)))
    nz = float(np.mean(exp != 0))
    neg_half, pos_half = tie_stats(pre)
    tag = "OK  " if ok else "FAIL"
    print(f"[B] {tag} {name:34s} out={tuple(out_shape)} "
          f"sat={sat*100:5.1f}% nonzero={nz*100:5.1f}% "
          f"ties={neg_half}-/{pos_half}+")
    if not ok:
        diff = np.nonzero(got != exp)
        n_bad = len(diff[0])
        print(f"       {n_bad}/{exp.size} elements differ; first few:")
        for i in range(min(6, n_bad)):
            idx = tuple(int(c[i]) for c in diff)
            print(f"         {idx}: cpp={got[idx]} ref={exp[idx]} acc={acc[idx]:.0f}")
    return ok, got, neg_half


def main():
    keep = "--keep" in sys.argv
    if not os.path.exists(EXE):
        print(f"missing {EXE}; run `make` first", file=sys.stderr)
        return 1
    if os.path.isdir(CASES):
        shutil.rmtree(CASES)
    os.makedirs(CASES)
    torch.set_num_threads(1)
    rng = np.random.default_rng(20240117)
    ok = check_fix()

    # PSMNet exercises exactly two (stride,pad) shapes with 3x3x3/dil1/FLOOR
    # and RELU|NONE; the rest of the matrix guards the general contract.
    cases = {
        # --- the two configurations that appear in PSMNet_pruned_2_pt ---
        "fwd_psmnet_s1":       dict(transposed=False, kernel=[3,3,3], stride=[1,1,1],
                                    pad=[1,1,1,1,1,1], nonlinear="RELU",
                                    **{"in": [1,7,6,5,8]}, oc=6, fp=(6,6,4,7)),
        "fwd_psmnet_s2":       dict(transposed=False, kernel=[3,3,3], stride=[2,2,2],
                                    pad=[1,0,1,0,1,0], nonlinear="RELU",
                                    **{"in": [1,8,7,6,8]}, oc=5, fp=(6,6,4,7)),
        "tconv_psmnet_s2":     dict(transposed=True,  kernel=[3,3,3], stride=[2,2,2],
                                    pad=[1,0,1,0,1,0], nonlinear="NONE",
                                    **{"in": [1,5,4,3,8]}, oc=6, fp=(6,6,4,7)),
        # --- axis-mapping traps: distinct kernel/stride/dilation/pad per axis ---
        "fwd_axis_distinct_k":  dict(transposed=False, kernel=[2,3,4], stride=[1,1,1],
                                     pad=[1,0,2,1,0,3], nonlinear="NONE",
                                     **{"in": [2,7,6,9,5]}, oc=4, fp=(6,6,4,7)),
        "fwd_axis_distinct_sd": dict(transposed=False, kernel=[3,2,3], stride=[3,2,1],
                                     dilation=[1,2,3], pad=[2,1,1,2,3,0],
                                     nonlinear="RELU",
                                     **{"in": [1,9,11,13,6]}, oc=5, fp=(6,6,4,7)),
        "tconv_axis_distinct":  dict(transposed=True,  kernel=[2,3,4], stride=[2,3,1],
                                     dilation=[2,1,1], pad=[1,0,2,1,1,2],
                                     nonlinear="NONE",
                                     **{"in": [1,5,4,6,5]}, oc=4, fp=(6,6,4,7)),
        "tconv_stride1_nopad":  dict(transposed=True,  kernel=[3,2,3], stride=[1,1,1],
                                     pad=[0,0,0,0,0,0], nonlinear="RELU",
                                     **{"in": [1,6,5,4,7]}, oc=3, fp=(6,6,4,7)),
        # --- pad_mode CEIL: output needs a ragged tap past the padded input ---
        "fwd_ceil_ragged":      dict(transposed=False, kernel=[3,3,3], stride=[2,2,2],
                                     pad=[0,0,0,0,0,0], pad_mode="CEIL",
                                     nonlinear="NONE",
                                     **{"in": [1,8,8,8,6]}, oc=4, fp=(6,6,4,7)),
        "fwd_valid":            dict(transposed=False, kernel=[3,3,3], stride=[1,1,1],
                                     pad=[0,0,0,0,0,0], pad_mode="VALID",
                                     nonlinear="NONE",
                                     **{"in": [1,6,6,6,6]}, oc=4, fp=(6,6,4,7)),
        # --- no bias ---
        "fwd_no_bias":          dict(transposed=False, kernel=[3,3,3], stride=[1,1,1],
                                     pad=[1,1,1,1,1,1], nonlinear="RELU", bias=False,
                                     **{"in": [1,6,5,4,8]}, oc=5, fp=(6,6,4,7)),
        "tconv_no_bias":        dict(transposed=True,  kernel=[3,3,3], stride=[2,2,2],
                                     pad=[1,0,1,0,1,0], nonlinear="NONE", bias=False,
                                     **{"in": [1,4,4,4,6]}, oc=4, fp=(6,6,4,7)),
        # --- saturation stress: shift_cut small, values large -> heavy clipping ---
        "fwd_saturate":         dict(transposed=False, kernel=[3,3,3], stride=[1,1,1],
                                     pad=[1,1,1,1,1,1], nonlinear="NONE",
                                     **{"in": [1,5,5,5,8]}, oc=4, fp=(6,6,12,4),
                                     range=(-128, 128)),
        # moderate clipping: both rails reached, but most outputs still carry
        # arithmetic, so a wrong tap still shows up as a mismatch
        "fwd_partial_sat":      dict(transposed=False, kernel=[3,3,3], stride=[1,1,1],
                                     pad=[1,1,1,1,1,1], nonlinear="NONE",
                                     **{"in": [1,6,6,6,8]}, oc=4, fp=(6,6,6,9)),
        "tconv_partial_sat":    dict(transposed=True,  kernel=[3,3,3], stride=[2,2,2],
                                     pad=[1,0,1,0,1,0], nonlinear="RELU",
                                     **{"in": [1,5,5,5,8]}, oc=4, fp=(6,6,6,9)),
        # --- activations ---
        "fwd_relu6":            dict(transposed=False, kernel=[3,3,3], stride=[1,1,1],
                                     pad=[1,1,1,1,1,1], nonlinear="RELU6",
                                     **{"in": [1,5,5,5,8]}, oc=4, fp=(6,6,2,5)),
        "fwd_leakyrelu":        dict(transposed=False, kernel=[3,3,3], stride=[1,1,1],
                                     pad=[1,1,1,1,1,1], nonlinear="LEAKYRELU",
                                     **{"in": [1,5,5,5,8]}, oc=4, fp=(6,6,4,7)),
        # --- batch > 1, single-threaded ---
        "fwd_batch2_1thread":   dict(transposed=False, kernel=[3,3,3], stride=[1,1,1],
                                     pad=[1,1,1,1,1,1], nonlinear="RELU", threads=1,
                                     **{"in": [3,5,4,4,6]}, oc=4, fp=(6,6,4,7)),
    }

    outs = {}
    neg_ties = 0
    for name, cfg in cases.items():
        good, got, nh = run_case(name, cfg, rng)
        ok = ok and good
        neg_ties += nh
        outs[name] = got

    # --- thread-count determinism: same case, 1 vs 7 workers, byte-identical ---
    det = dict(transposed=False, kernel=[3,3,3], stride=[1,1,1], pad=[1,1,1,1,1,1],
               nonlinear="RELU", **{"in": [2,11,9,7,8]}, oc=6, fp=(6,6,4,7))
    a_ok, a, _ = run_case("det_threads_1", dict(det, threads=1),
                          np.random.default_rng(7))
    b_ok, b, _ = run_case("det_threads_7", dict(det, threads=7),
                          np.random.default_rng(7))
    same = np.array_equal(a, b)
    print(f"[C] {'OK  ' if same else 'FAIL'} 1 vs 7 threads bit-identical")
    ok = ok and a_ok and b_ok and same

    # The negative-half tie rule is the one place util::fix() departs from
    # round(); make sure the convolution cases actually reached it.
    print(f"[D] {'OK  ' if neg_ties else 'WEAK'} negative .5 ties hit inside "
          f"conv: {neg_ties}")
    ok = ok and neg_ties > 0

    if not keep:
        shutil.rmtree(CASES)
    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
