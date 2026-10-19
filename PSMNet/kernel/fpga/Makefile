# C simulation of the HLS kernel against the CPU golden kernel (plain g++).
#   make check            RED=24 (default)
#   make check RED=16
XILINX_HLS ?= /tools/X22/Vitis_HLS/2022.2
CPU_TASK   ?= ../cpu
RED        ?= 24
CXX        ?= g++
CXXFLAGS   ?= -std=c++17 -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label

INC = -I. -I$(CPU_TASK)/ops/common -I$(CPU_TASK)/test/conv3d_fix/stub \
      -I$(XILINX_HLS)/include

test_conv3d_fix: test_conv3d_fix.cpp conv3d_fix_hls.cpp conv3d_fix_hls.h \
                 conv3d_fix_fpga_host.hpp
	$(CXX) $(CXXFLAGS) -DCONV3D_RED=$(RED) $(INC) -o $@ \
	    test_conv3d_fix.cpp conv3d_fix_hls.cpp -pthread

check: test_conv3d_fix
	./test_conv3d_fix

# Same test with every DATAFLOW stage in its own thread and bounded FIFOs of
# the real depth (sim_threaded/hls_stream.h): a stalled stage prints DEADLOCK.
test_conv3d_fix_thr: test_conv3d_fix.cpp conv3d_fix_hls.cpp conv3d_fix_hls.h \
                     conv3d_fix_fpga_host.hpp sim_threaded/hls_stream.h
	$(CXX) $(CXXFLAGS) -DCONV3D_RED=$(RED) -DCONV3D_THREADED_SIM -Isim_threaded $(INC) -o $@ \
	    test_conv3d_fix.cpp conv3d_fix_hls.cpp -pthread

check-threaded: test_conv3d_fix_thr
	./test_conv3d_fix_thr

# HLS synthesis + .xo export + Vivado synthesis for real numbers (~25 min)
hls:
	RED=$(RED) VIVADO_SYN=1 $(XILINX_HLS)/bin/vitis_hls -f run_hls.tcl
	python3 check_util.py conv3d_fix_prj/sol1/impl/report/verilog/export_syn.rpt 40

clean:
	rm -rf test_conv3d_fix test_conv3d_fix_thr conv3d_fix_prj *.log conv3d_fix.xo

.PHONY: check check-threaded hls clean
