# Kernel design patterns (from conv3d_fix)

Source of truth: `kernel/fpga/conv3d_fix_hls.h` (constants, cfg/tab layout),
`conv3d_fix_hls.cpp` (kernel), `conv3d_fix_fpga_host.hpp` (`make_plan`,
`build_tables`, `pack_weights`).

## 1. Host thinks, FPGA computes
Everything that is index math, done once per layer, is computed on the host
and passed in three buffers. The kernel never divides, never handles padding
or stride logic, never sees floats.

| buffer | type | contents (conv3d) |
|---|---|---|
| `cfg[32]` | int32 | shapes, `NOCT`, `NRB`, `NTILES`, `LW`, `GF_BYTES`, `W_ROWS`, `OUT_ROWS`, `EPI_N`, `SIX_Q`, `NONLINEAR`; `cfg[31] = CFG_STATUS` written back by the kernel |
| `tab[]` | int32 | `tapH[OH][KH]`, `tapW[OW][KW]` (input row/col or -1), `phase[NT]`, `pos[NT][16]`, `obyte[NT][16]`, `bterm[OC]`, `thr6[OC]` |
| `weight` | packed w128 rows | one row = 16 oc lanes × RED bytes in the order the MAC loop consumes them |

The layout lives in ONE header included by the kernel and the host
(`conv3d_fix_hls.h`: `enum CfgIdx`, the `tab[]` comment block, the weight
packing formula). Change it in one place only.

## 2. Job decomposition
Pick the smallest unit whose working set fits on chip and whose output is
contiguous in DDR. conv3d: one **output line** `(n, oh, ow)` = all `OD × OC`
outputs (contiguous in `[N][OH][OW][OD][OC]`).
- `njobs = N · OH · OW`
- inputs per job: one staged input line per valid `(kh, kw)` (≤ `KH·KW`)
- all accumulators of the job stay on chip: `NT · NOCT ≤ MAX_TN` (16)
- the output line is assembled in a 16-bank BRAM (`lbuf`, `OD·OC ≤ LINE_MAX`)
  and written with full 16 B bursts by a separate writer stage.

## 3. Flattened reduction (odd channel counts waste nothing)
Along D, output position `o` reads `L` consecutive input positions
`d_start(o) … d_start(o)+L−1`, and the tensor is `[..][D][IC]`, so those
`L·IC` bytes are **contiguous**. Reduce over that byte range in chunks of
`RED` bytes instead of padding IC:

```
IC = 26, L = 3 (3 taps):   L·IC = 78 bytes
padded per tap (old 16x16): 3 × ceil(26/16)·16 = 96 lanes   (81 % useful)
flattened, RED = 16:        NRB = ceil(78/16) = 5 → 80 lanes (97.5 % useful)
```
`NRB = ceil(L · IC / RED)`. Weight bytes for flat index `f = rb·RED + r`:
`m = f / IC` (slot), `c = f % IC` (channel), `kd = pattern[phase][m]`, zero
when `m ≥ L` or the slot has no tap. Bytes past `L·IC` in the last chunk read
real input data but multiply a zero weight.

## 4. Transposed convolution by phases (no zero-inserted input)
For stride `s`, output positions fall into `s` "phases": which kernel taps hit
real input samples depends only on `o mod s`. The host:
1. computes, per output `o`, the virtual taps `(k, i)` (no range check; out of
   range reads land in the staged line's zero guard bands `GF`/`GB`),
2. builds the tap pattern of length `L` (`kd` or −1 per slot) and assigns a
   phase id per distinct pattern,
3. groups outputs of one phase into tiles of 16 (`tile_phase[]`, `pos[][]`,
   `obyte[][]`), packs one weight set per phase (holes = zero weights).
Same datapath as a regular conv; only the tables differ.

## 5. Two int8 MACs per DSP48E2
```
packed = (b << 18) + a            // a, b: int8 inputs of positions t and t+8  (27-bit A port)
p      = packed * w               // w: int8 weight                             (18-bit B port)
       = (b·w)·2^18 + (a·w)
ps     = Σ_{GROUP} p              // sum before unpacking
lo     = ps[17:0] as signed       // Σ a·w   exact while |Σ a·w| < 2^17
hi     = (ps >> 18) + (lo < 0)    // Σ b·w   (+1 = borrow from the negative low field)
```
- bound: `|a·w| ≤ 2^14` ⇒ `GROUP · 2^14 < 2^17` ⇒ `GROUP ≤ 7` (code: 6 for
  RED % 6 == 0, else 4).
- `#pragma HLS BIND_OP variable=p op=mul impl=dsp` on the product.
- DSPs ≈ `OC_PAR × RED` (16×16 = 256 MAC DSPs for RED=16 → 320 total routed;
  16×24 = 384 → 446–448 total). MAC/cycle = `2 × OC_PAR × RED`.
- Do NOT try `BIND_OP op=add impl=dsp` for the group sums: it put every add in
  its own DSP (+320 DSP), it did not fuse with the multiplier.

## 6. Memory plan
| data | storage | why |
|---|---|---|
| weights | URAM, even rows `wm0` / odd rows `wm1` (`ram_2p`) | 2 rows/cycle → a 16-row block refills in 8 cycles |
| current/next weight block | registers `wcur[16]`, `wnxt[16]` | `wnxt` is a shift register (2 rows in per cycle), copied to `wcur` at block end — no 192-bit muxes |
| staged input line | LUTRAM `ram_1wnr`, 4 banks × ping-pong (`ln0`, `ln1`) | any 3 consecutive 16 B words hit 3 different banks; 1 write + 2 reads per cycle |
| accumulators | LUTRAM `accA/accB[16][MAX_TN·8]` | whole job on chip; see the RAW distance rule |
| output line | BRAM `lbuf[16][LINE_MAX/16]` (byte banks) | 16 rotated bytes per cycle, emitted as 16 B rows |
Weight block reuse: one 16×RED block serves 8 cycles × 2 positions = 16
outputs while the next block is prefetched.

## 7. Five-stage DATAFLOW
```
reader ─raw_s(64)→ aligner ─data_s(32)→ mac_engine ─accA/accB(4)→ epilogue ─chunk_s(4)→ writer
   └──────────── hdr_s(2·MAX_PAIRS+4): per job #lines, (kh,kw) list ──┘
```
FIFO depth rule:
- A FIFO between two stages that both run at 1 item/cycle needs only a few
  entries (SRL): `accA/accB`, `chunk_s` = 4.
- A FIFO that must absorb a burst while the consumer is busy elsewhere must
  hold that burst — or the consumer must not wait on anything the producer
  emits after the burst (this is the deadlock in lessons.md #1).
- Header/control FIFOs must hold every header the producer can be ahead by
  (conv3d: 2 jobs × (1 + MAX_PAIRS) words → depth 102).
- Then PROVE it with the threaded simulation (real depths, all stages
  concurrent). Sequential C-sim with unbounded FIFOs cannot.

## 8. Limits and fallback
- `MAX_*` constants in the shared header size every on-chip array.
- Kernel: a cheap all-in-one config check at entry → `cfg[CFG_STATUS] =
  ST_BAD_CFG; return;` (never start the dataflow on a bad config).
- Host: `FpgaKernel::supported(param, &why)` = `make_plan()` returning false
  with a reason for every limit → the op glue falls back to the CPU kernel.
- Weight limit: rows = `nphase · KH · KW · NRB · NOCT · 16 ≤ W_ROWS`
  (blocks ≤ `W_ROWS / 16`). PSMNet's largest (IC 64, OC 26, RED 16):
  `NRB = ceil(3·64/16) = 12` → 1·9·12·2·16 = 3456 rows ≤ 8192.
