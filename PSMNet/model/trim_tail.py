#!/usr/bin/env python3
"""Cut the CPU transpose/reshape tail off the quantized PSMNet kernel-2 xmodel.

The quantizer exports kernel 2 as

    resize (trilinear) -> fix [1,576,960,192,1]      <- the real network output
      -> transpose [0,4,3,1,2] -> reshape [1,192,576,960]

The compiler leaves the transpose/reshape on the CPU, where they took 187 s
per frame and ~1 GB of RAM (float32 copies). Cutting the graph after the fix
op gives an int8 [1,576,960,192,1] = [H][W][D] output, which psmnet.cpp reads
as-is.

Run inside the Vitis-AI 3.0 docker (needs the xir python module), from
PSMNet/model (see README.md, step 3):

    python trim_tail.py pt_psmnet_sceneflow_576_960_0.68_696G_3.0/quantized/PSMNet_2_int.xmodel \\
                        PSMNet_2_int_notail.xmodel
    # optional: --cut <op name>   (default: the fix op feeding the transpose)
"""
import argparse

import xir


def ancestors(op, seen):
    if op.get_name() in seen:
        return
    seen.add(op.get_name())
    for ins in op.get_input_ops().values():
        for i in ins:
            ancestors(i, seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--cut", help="name of the op that becomes the output")
    a = ap.parse_args()

    g = xir.Graph.deserialize(a.src)
    ops = g.toposort()

    if a.cut:
        cut = g.get_op(a.cut)
        assert cut is not None, f"no op named {a.cut}"
    else:
        tr = [o for o in ops if o.get_type() == "transpose"]
        assert len(tr) == 1, f"expected one transpose, found {len(tr)}"
        cut = tr[0].get_input_ops()["input"][0]
    assert cut.get_type() == "fix", f"cut op is a {cut.get_type()}, not fix"

    keep = set()
    ancestors(cut, keep)
    drop = [o for o in ops if o.get_name() not in keep]
    print(f"cut after: {cut.get_name()}  {cut.get_output_tensor().dims} "
          f"fix_point {cut.get_attr('fix_point')}")
    for o in reversed(drop):  # consumers first
        print(f"  remove {o.get_type():10s} {o.get_name()}")
        g.remove_op(o)

    tails = g.get_tail_ops()
    assert [t.get_name() for t in tails] == [cut.get_name()], \
        f"unexpected outputs: {[t.get_name() for t in tails]}"
    g.serialize(a.dst)
    print(f"wrote {a.dst}: {g.get_op_num()} ops (was {len(ops)}), "
          f"output {cut.get_output_tensor().dims}")


if __name__ == "__main__":
    main()
