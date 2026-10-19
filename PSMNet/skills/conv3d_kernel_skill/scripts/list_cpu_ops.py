#!/usr/bin/env python3
"""List the ops a compiled xmodel runs on the CPU (the candidates for a PL
kernel), with shapes and the attributes a kernel needs.

Run inside the Vitis-AI docker that compiled the model (needs `import xir`):

    source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch
    python list_cpu_ops.py compiled/PSMNet_pruned_2_pt/PSMNet_pruned_2_pt.xmodel

Prints one line per CPU op type with its count, then every op of the chosen
type (default: the non-trivial ones) with input/weight/output shapes, fix
points and attributes such as kernel/stride/pad/dilation/nonlinear.
"""
import collections
import sys

import xir

SKIP = {"const", "const-fix", "data", "data-fix", "download", "upload"}
ATTRS = ("kernel", "stride", "pad", "dilation", "nonlinear", "pad_mode", "group",
         "order", "shape", "axis", "mode")


def main(path, only_type=None):
    g = xir.Graph.deserialize(path)
    kids = g.get_root_subgraph().toposort_child_subgraph()
    cpu_ops = []
    for sg in kids:
        dev = sg.get_attr("device") if sg.has_attr("device") else "?"
        if dev != "CPU":
            continue
        cpu_ops += [o for o in sg.get_ops() if o.get_type() not in SKIP]
    print("CPU op types:", dict(collections.Counter(o.get_type() for o in cpu_ops)))
    for o in cpu_ops:
        if only_type and o.get_type() != only_type:
            continue
        ins = []
        for name, ops in o.get_input_ops().items():
            for i in ops:
                t = i.get_output_tensor()
                fp = t.get_attr("fix_point") if t.has_attr("fix_point") else "-"
                ins.append(f"{name}{t.dims}@fix{fp}")
        t = o.get_output_tensor()
        fp = t.get_attr("fix_point") if t.has_attr("fix_point") else "-"
        a = {k: o.get_attr(k) for k in ATTRS if o.has_attr(k)}
        print(f"{o.get_type():24s} {o.get_name()[-60:]}\n"
              f"    in={ins}\n    out={t.dims}@fix{fp} {t.dtype} {a}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
