# Vitis HLS 2022.2:  vitis_hls -f run_hls.tcl     (or: make hls)
# KV260 platform clocks are fixed at 100/200/400 MHz -> 200 MHz (5 ns).
set red [expr {[info exists ::env(RED)] ? $::env(RED) : 24}]
open_project -reset conv3d_fix_prj
set_top conv3d_fix
add_files conv3d_fix_hls.cpp -cflags "-I. -DCONV3D_RED=$red"
add_files -tb test_conv3d_fix.cpp -cflags "-I. -I../cpu/ops/common -I../cpu/test/conv3d_fix/stub -DCONV3D_RED=$red"
open_solution -reset sol1 -flow_target vitis
set_part xck26-sfvc784-2LV-c
create_clock -period 5 -name default
set_clock_uncertainty 15%
config_interface -m_axi_alignment_byte_size 64 -m_axi_max_widen_bitwidth 128
csynth_design
export_design -format xo -output conv3d_fix.xo
if {[info exists ::env(VIVADO_SYN)]} { export_design -flow syn -rtl verilog -format ip_catalog }
exit
