# Vitis HLS 2022.2: vitis_hls -f run_hls.tcl   (real: kernel/fpga/run_hls.tcl)
# VIVADO_SYN=1 also runs Vivado synthesis for real LUT/FF/DSP/BRAM numbers.
open_project -reset myop_fix_prj
set_top myop_fix
add_files myop_fix_hls.cpp -cflags "-I."
add_files -tb test_myop_fix.cpp -cflags "-I. -I<cpu_task>/ops/common -I<cpu_task>/test/conv3d_fix/stub"
open_solution -reset sol1 -flow_target vitis
set_part xck26-sfvc784-2LV-c
create_clock -period 5 -name default          ;# 200 MHz = platform clock id 2
set_clock_uncertainty 15%
config_interface -m_axi_alignment_byte_size 64 -m_axi_max_widen_bitwidth 128
csynth_design
export_design -format xo -output myop_fix.xo
if {[info exists ::env(VIVADO_SYN)]} { export_design -flow syn -rtl verilog -format ip_catalog }
exit
