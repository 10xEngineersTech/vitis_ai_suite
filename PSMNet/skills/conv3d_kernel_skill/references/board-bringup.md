# Link, deploy and validate on the KV260

Real project: `dpu_sfm` (Vitis 2022.2 IDE: `dpu_kernels`, `dpu_system_hw_link`,
`dpu_system`), platform `kv260_custom_platform`, PetaLinux 2022.2, XRT 2.14.

## 1. Link (one xclbin for DPU + all PL kernels)
The KV260 holds one xclbin; the DPU runtime keeps it loaded. Your kernel must
be linked into the same xclbin as `DPUCZDX8G`.

Command line (the IDE writes the exact command to
`dpu_system_hw_link/Hardware/dpu.xclbin.sh`):
```sh
v++ --target hw --link --config dpu-link.cfg --config system.cfg -o dpu.xclbin \
    ../../dpu_kernels/Hardware/DPUCZDX8G.xo myop_fix.xo
```
IDE: add the kernel to the binary container; put `[clock]`, `[connectivity]`,
`[vivado]` lines in the container's config-settings box (Hardware
configuration). Always check the GENERATED `Hardware/dpu-link.cfg`: the IDE
box and the generated file have diverged (a link once lost all DPU `sp=`
lines). `TODO: verify` why — check `dpu_system_hw_link.prj` `configSettings`.

## 2. After the link — must all pass
```sh
scripts/check_xclbin.sh Hardware/dpu.xclbin <bitstream.bit.bin> Hardware/
```
- `WNS >= 0` ("All user specified timing constraints are met.")
- every M_AXI of your kernel is 128 bit
- every master has an `sp=`; your kernel's clock id is the 200 MHz one (`id=2`)
- register offsets match the host driver's `enum Reg`
- per-kernel routed utilization (conv3d RED=16: 40,999 LUT, 9,845 LUT-as-mem,
  28,935 FF, 22 BRAM, 4 URAM, 320 DSP; whole design 72 % LUT, 62.5 % BRAM)

## 3. Bitstream for the board
`v++` does not create the `.bit.bin`:
```sh
cd Hardware/dpu.build/link/int        # system.bit of THIS link
echo 'all:{system.bit}' > bootgen.bif
bootgen -w -arch zynqmp -process_bitstream bin -image bootgen.bif   # -> system.bit.bin
```

## 4. Deploy (board)
```sh
APP=/lib/firmware/xilinx/<app>                  # the app /etc/vart.conf points into
BAK=/home/petalinux/fw_backup_<app>; mkdir -p $BAK; cp -a $APP/. $BAK/   # backups OUTSIDE /lib/firmware
sudo cp dpu.xclbin $APP/dpu.xclbin
sudo cp system.bit.bin $APP/<existing name>.bit.bin
ls $APP          # exactly one *.bit.bin, one *.dtbo, dpu.xclbin, shell.json
sudo xmutil unloadapp; sudo xmutil loadapp <app>      # or reboot
cat /etc/vart.conf                                     # firmware: $APP/dpu.xclbin
sudo xbutil examine -r all -d 0000:00:00.0 | grep -E "UUID|<your cu>|DPUCZDX8G"
xclbinutil --info --input $APP/dpu.xclbin | grep "UUID (xclbin)"   # same uuid
```
No device-tree change: kernels get no DT nodes in the flat-shell flow; zocl
reads IP_LAYOUT from the xclbin and the platform overlay already routes 32
interrupts (`axi_intc_0`) to `zyxclmm_drm`.

Op library:
```sh
sudo cp libvart_op_imp_<op>.so /usr/lib/          # replaces the CPU-only one
strings /usr/lib/libvart_op_imp_<op>.so | grep -c VAI_<OP>_FPGA_VERIFY   # 1 = new build
ldd /usr/lib/libvart_op_imp_<op>.so | grep "not found"                    # nothing
```

## 5. Validate
```sh
sudo rm -f /tmp/DPU_*                     # stale locks after root runs
DEBUG_<OP>_FPGA=1 VAI_<OP>_FPGA_VERIFY=1 ./app ... 2>&1 | grep -E ": FPGA|: CPU|VERIFY"
```
Every layer must say `FPGA` (or `CPU (<reason>)` on purpose) and
`mismatches 0`. Then run without VERIFY for timing:
`DEEPHI_PROFILING=1` → `CPU_OP_EXEC : <us>` per op.

## 6. Mismatch → cause
Start from the FIRST mismatching layer in graph order (later layers inherit
its error). Group the mismatching outputs by coordinate:

| pattern | likely cause |
|---|---|
| only layers with `NOCT = 1` (OC ≤ 16) and > 1 reduction block | accumulator RAW hazard (`inter false` / distance too small) |
| only transposed layers (`[T]`) | phase table / per-phase weight packing |
| only the last D tile (`od ≥ 16·(NT−1)`) | partial tile: invalid positions not masked (`obyte = −1`) |
| only small/large `oh`, `ow`, `od` | padding: tap tables (−1) or guard bands `GF/GB` |
| only certain `oc` groups / lanes | weight packing order vs MAC loop order |
| mostly off-by-1 | epilogue rounding (`bterm`, shift, clamp order) |
| random scattered, varies run to run | timing violation (WNS < 0 bitstream) or a cache sync missing |
| all zeros / constant | wrong buffer address, CU never started, `CFG_STATUS` ignored |

## 7. End-to-end
Compare the final model output with a run where the op is forced to the CPU
(`VAI_<OP>_FPGA=0`) — not with an older build. "Small, localized noise with the
overall structure intact" in PSMNet meant: wide layers exact, a few narrow
bottleneck layers wrong (the NOCT = 1 hazard). `VAI_<OP>_FPGA_VERIFY=2`
substitutes the CPU result per layer: if that image is clean, the kernel is
the culprit.
