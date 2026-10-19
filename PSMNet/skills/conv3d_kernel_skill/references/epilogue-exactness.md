# Bit-exact integer epilogue

Real code: host `epilogue_plan()` + `set_bias()` in
`kernel/fpga/conv3d_fix_fpga_host.hpp`; kernel `epi_lane()` in
`conv3d_fix_hls.cpp`; CPU reference epilogue in
`kernel/cpu/ops/common/conv3d_fix_kernel.hpp` (`run_rows`).

## What the CPU computes (the reference you must match)
```cpp
float filout = static_cast<float>(acc);                       // int64 acc
if (bias) filout += static_cast<float>(bias[oc]) * shift_bias_scale;   // 2^b
out = util::fix(nonlinear_fn(filout / shift_cut_scale));      // / 2^n
```
`util::fix` (kernel/cpu/ops/common/util.hpp): clamp to [−128, 127], round to
nearest with ties toward +∞ (`ceil` for negative x.5, `round` otherwise).
`shift_cut_scale = 2^(in_fp + w_fp − out_fp)`, `shift_bias_scale =
2^(in_fp + w_fp − bias_fp)` (`Param::set_fix_point / set_bias_fix_point`).

## What the kernel computes
```
host, per output channel:  bterm[oc] = floor(bias[oc]·2^b + 2^(n−1))
                           thr6[oc]  = ceil(six·2^n − bias[oc]·2^b)      (RELU6)
kernel, per output:        r = (acc + bterm[oc]) >> n                     (arith. shift)
                           RELU6: acc ≥ thr6 → six_q;  RELU/RELU6: r < 0 → 0
                           clamp r to [−128, 127]
```
Why it is exact: for integer `acc`, real `f` and `n ≥ 0`,
`floor((acc + f) / 2^n) == floor((acc + floor(f)) / 2^n)`. With
`f = bias·2^b + 2^(n−1)` the left side is "round half up of (acc + bias·2^b)/2^n",
i.e. `util::fix` of the exact value. RELU commutes with rounding
(`fix(max(v,0)) == max(fix(v),0)`).

## When the CPU's float maths stops being exact
`float(acc)` and the float add round once the value needs > 24 significant
bits: `|sum| ≥ 2^24 · min(1, 2^b)` or `|acc| ≥ 2^24`. The host proves that
any such value is ≥ 256 after `/2^n`, so both sides saturate to the same int8:
```cpp
const double lim1 = std::ldexp(1.0, 24 + std::min(0, b) - n);
const double lim2 = (std::ldexp(1.0, 24) - std::ldexp(1.0, 7 + std::max(0, b))) / std::ldexp(1.0, n);
if (lim1 < 256.0 || lim2 < 256.0) return fail(why, "float epilogue not provably exact in integers");
```
Plus: `shift_cut_scale` and `shift_bias_scale` must be powers of two
(`frexp(v) == 0.5`), `0 ≤ n ≤ 24`, `|bterm| < 2^31`. Any failed check → CPU
path for that layer.

## Test it
The golden test must sweep several fix-point sets, not just one. conv3d's
test runs every PSMNet shape with `{in, w, out, bias}` =
`{4,4,1,4}`, `{5,6,2,14}` (b < 0), `{3,3,6,9}` (n = 0) and `{6,7,3,6}` with
full-range inputs (≈ 90 % saturated outputs).

## History (what did NOT work)
1. Float epilogue in the kernel (16 float lanes): 37.7k LUT.
2. Integer epilogue with 64-bit variable shifts per lane: 60k LUT (32 barrel
   shifters). The per-channel `bterm` folding above brought it to ~9k LUT.
