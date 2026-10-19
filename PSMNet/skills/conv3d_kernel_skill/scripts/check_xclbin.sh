#!/bin/bash
# Post-link checks for a KV260 xclbin that holds the DPU + custom kernel(s).
#
#   check_xclbin.sh <dpu.xclbin> [system.bit.bin] [link Hardware dir]
#
# Prints: uuid, kernels, per-kernel CU address / arg offsets / memory banks /
# AXI port widths, the clock ids from the embedded v++ command line, whether
# the .bit.bin is the bitstream inside this xclbin, and (with the link dir)
# WNS and the per-kernel routed utilization.
set -uo pipefail
X="${XCLBINUTIL:-/tools/X22/Vitis/2022.2/bin/xclbinutil}"
BG="${BOOTGEN:-/tools/X22/Vivado/2022.2/bin/bootgen}"
XB="$1"; BIN="${2:-}"; LINK="${3:-}"
INFO="$(mktemp)"; trap 'rm -rf "$INFO" "$INFO.d"' EXIT
"$X" --info --input "$XB" > "$INFO" 2>/dev/null

echo "== xclbin"
grep -E "UUID \(xclbin\)|Kernels:" "$INFO"
echo "== CUs (base address, args -> register offset / memory)"
awk '/^Kernel: /{k=$2} /Instance:/{print k, $2} /Base Address/{print "   base", $3}
     /Argument:/{a=$2} /Register Offset/{o=$3} /Memory:/{if(a!="")print "     ", a, o, "->", $2; a=""}' "$INFO"
echo "== AXI master widths (expect 128 on KV260 HP ports)"
awk '/^Kernel: /{k=$2} /^Ports$/{inp=1} /^Instance:/{inp=0}
     inp && /Port: +M_AXI/{p=$2} inp && /Data Width/{if(p!=""){print "  ", k, p, $3, $4; p=""}}' "$INFO"
echo "== clock ids from the link command line (kv260_custom_platform: 1=100 2=200 3=400 MHz)"
grep -oE "\-\-clock\.(id|freqHz) [^ ]+" "$INFO" | sort -u
echo "== sp= memory mappings on the link command line (every master should have one)"
grep -oE "\-\-connectivity\.sp [^ ]+" "$INFO" | sort -u

if [ -n "$BIN" ]; then
  echo "== bitstream match"
  mkdir -p "$INFO.d"
  "$X" --input "$XB" --dump-section BITSTREAM:RAW:"$INFO.d/x.bit" --force >/dev/null 2>&1
  (cd "$INFO.d" && echo 'all:{x.bit}' > b.bif && "$BG" -w -arch zynqmp -process_bitstream bin -image b.bif >/dev/null 2>&1)
  if cmp -s "$INFO.d/x.bit.bin" "$BIN"; then echo "   OK: $BIN is the bitstream inside $XB"
  else echo "   MISMATCH: $BIN does not come from this xclbin's link"; fi
fi

if [ -n "$LINK" ]; then
  R="$LINK/dpu.build/link/vivado/vpl/prj/prj.runs/impl_1"
  echo "== timing (WNS must be >= 0)"
  grep -E "^ +-?[0-9]+\.[0-9]+ +-?[0-9]+\.[0-9]+ +[0-9]+ +[0-9]+ +-?[0-9]" \
       "$R/system_wrapper_timing_summary_routed.rpt" | head -1 | awk '{print "   WNS", $1, " TNS", $2, " failing", $3}'
  echo "== routed utilization per kernel"
  grep -E "^\| +(Name|[A-Za-z0-9_]+) +\|" "$R/kernel_util_routed.rpt" | head -6
fi
