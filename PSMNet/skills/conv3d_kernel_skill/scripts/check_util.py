#!/usr/bin/env python3
"""Fail if the kernel uses more than LIMIT % of any KV260 (XCK26) resource.

usage:  check_util.py <report> [limit_percent=40]
  <report>  conv3d_fix_prj/sol1/syn/report/csynth.xml          (HLS estimate)
         or conv3d_fix_prj/sol1/impl/report/verilog/export_syn.rpt
                                                     (Vivado synthesis, real)
HLS estimates are pessimistic for LUTs; judge by the Vivado number (make hls
runs it), and re-check after v++ link (report_utilization of the full design).
"""
import re
import sys
import xml.etree.ElementTree as ET

TOTAL = {"LUT": 117120, "FF": 234240, "BRAM_18K": 288, "URAM": 64, "DSP": 1248}

path = sys.argv[1]
limit = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0

if path.endswith(".xml"):
    root = ET.parse(path).getroot()
    used = {r.tag: int(r.text) for r in root.find(".//AreaEstimates/Resources")}
    src = "HLS estimate"
else:  # Vivado export_syn.rpt: "LUT:   45711" ...
    txt = open(path).read()
    used = {}
    for key, name in (("LUT", "LUT"), ("FF", "FF"), ("DSP", "DSP"),
                      ("BRAM", "BRAM_18K"), ("URAM", "URAM")):
        m = re.search(r"^%s:\s+(\d+)" % key, txt, re.M)
        used[name] = int(m.group(1)) if m else 0
    src = "Vivado synthesis"

ok = True
print(f"{src}: {path}")
print(f"{'resource':10s}{'used':>9s}{'device':>9s}{'%':>8s}  (limit {limit:.0f}%)")
for k, total in TOTAL.items():
    u = used.get(k, 0)
    pct = 100.0 * u / total
    flag = "" if pct <= limit else "   <-- OVER BUDGET"
    ok &= pct <= limit
    print(f"{k:10s}{u:9d}{total:9d}{pct:7.1f}%{flag}")
sys.exit(0 if ok else 1)
