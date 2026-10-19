#!/bin/bash
# Strip the HLS dataflow debug metadata (<module> tree, <FIFOInformation>) of
# conv3d_fix from an xclbin's EMBEDDED_METADATA. It is only used by hardware
# emulation / debug tools, but it makes the section ~100 KB, which crashes
# dfx-mgrd 2022.2 on the KV260 ("malloc(): invalid next size") right after
# `xmutil loadapp`. Bitstream, register map and connectivity are untouched;
# xclbinutil assigns the result a new uuid.
#
#   ./trim_xclbin_metadata.sh in.xclbin out.xclbin
set -euo pipefail
IN="$1"; OUT="$2"
XCLBINUTIL="${XCLBINUTIL:-/tools/X22/Vitis/2022.2/bin/xclbinutil}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
"$XCLBINUTIL" --input "$IN" --dump-section EMBEDDED_METADATA:RAW:"$TMP/em.xml" --force >/dev/null
python3 - "$TMP/em.xml" "$TMP/em_trim.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
t = ET.parse(sys.argv[1])
n = 0
for k in t.getroot().iter('kernel'):
    for ch in list(k):
        if ch.tag in ('module', 'FIFOInformation'):
            k.remove(ch); n += 1
t.write(sys.argv[2], encoding='UTF-8', xml_declaration=True)
print(f"removed {n} debug element(s)")
PY
"$XCLBINUTIL" --input "$IN" --replace-section EMBEDDED_METADATA:RAW:"$TMP/em_trim.xml" \
              --output "$OUT" --force >/dev/null
"$XCLBINUTIL" --info --input "$OUT" | grep -E "UUID \(xclbin\)|Kernels:"
echo "EMBEDDED_METADATA: $(stat -c %s "$TMP/em.xml") -> $(stat -c %s "$TMP/em_trim.xml") bytes"
