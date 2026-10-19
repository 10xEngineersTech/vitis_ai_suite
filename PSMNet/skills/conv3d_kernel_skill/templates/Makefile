# Template (real: kernel/fpga/Makefile). Paths are the KV260 dev host's.
XILINX_HLS ?= /tools/X22/Vitis_HLS/2022.2
CPU_TASK   ?= ../src/vai_library/cpu_task
CXX        ?= g++
CXXFLAGS   ?= -std=c++17 -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label
INC = -I. -I$(CPU_TASK)/ops/common -I$(CPU_TASK)/test/conv3d_fix/stub -I$(XILINX_HLS)/include
SRC = test_myop_fix.cpp myop_fix_hls.cpp

# 1. C simulation vs the CPU golden kernel (sequential, unbounded FIFOs)
check:
	$(CXX) $(CXXFLAGS) $(INC) -o test_myop_fix $(SRC) -pthread && ./test_myop_fix

# 2. Threaded simulation: one thread per DATAFLOW stage, real FIFO depths;
#    a stall prints DEADLOCK and exits 3. sim_threaded/ must come first in -I.
check-threaded:
	$(CXX) $(CXXFLAGS) -DMYOP_THREADED_SIM -Isim_threaded $(INC) -o test_myop_fix_thr $(SRC) -pthread
	CONV3D_SIM_STALL_MS=3000 ./test_myop_fix_thr

# 3. HLS + .xo + Vivado synthesis + budget check (~25 min)
hls:
	VIVADO_SYN=1 $(XILINX_HLS)/bin/vitis_hls -f run_hls.tcl
	python3 check_util.py myop_fix_prj/sol1/impl/report/verilog/export_syn.rpt 40

# Never delete the exported .xo in `clean` (a clean once destroyed a build).
clean:
	rm -f test_myop_fix test_myop_fix_thr

.PHONY: check check-threaded hls clean
