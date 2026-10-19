# CPU conv3d-fix / transposed-conv3d-fix (golden reference + fallback).
#
#   make            aarch64 op libraries (cpu_task plugins) + host test binary
#   make lib        aarch64/libvart_op_imp_{conv3d-fix,transposed-conv3d-fix}.so
#   make test       host test binary (no VART/XIR/glog needed, glog stub in test/conv3d_fix/stub)
#   make check      run the PyTorch goldens (needs torch + numpy)
#   make clean
#
# Override the toolchain / sysroot on the command line if they live elsewhere:
#   make SYSROOT=/path/to/sysroots/cortexa72-cortexa53-xilinx-linux AARCH64_CXX=/path/to/aarch64-linux-gnu-g++

SYSROOT     ?= /mnt/ssd2/abdul_raouf_khalid/petalinux_sdk_2022.2/sysroots/cortexa72-cortexa53-xilinx-linux
AARCH64_CXX ?= /tools/X22/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
OUT         ?= aarch64

# Same flags as test/conv3d_fix/build_aarch64.sh. The in-tree include/ comes
# first so its vart/op_imp.h wins over the sysroot's; --no-undefined turns a
# missing symbol into a link error here instead of a dlopen failure on the board.
LIB_FLAGS = --sysroot=$(SYSROOT) -O2 -std=c++17 -fPIC -shared \
            -Wall -Wextra -Wno-unused-parameter \
            -I include -I $(SYSROOT)/usr/include \
            -Wl,--no-undefined -L $(SYSROOT)/usr/lib \
            -lvitis_ai_library-runner_helper -lvart-runner -lxir -lglog -lpthread

COMMON = ops/common/conv3d_fix_kernel.hpp ops/common/conv3d_fix_op.hpp ops/common/util.hpp include/vart/op_imp.h

all: lib test

lib: $(OUT)/libvart_op_imp_conv3d-fix.so $(OUT)/libvart_op_imp_transposed-conv3d-fix.so

$(OUT)/libvart_op_imp_conv3d-fix.so: ops/conv3d-fix/conv3d-fix.cpp $(COMMON)
	mkdir -p $(OUT)
	$(AARCH64_CXX) -o $@ $< $(LIB_FLAGS)

$(OUT)/libvart_op_imp_transposed-conv3d-fix.so: ops/transposed-conv3d-fix/transposed-conv3d-fix.cpp $(COMMON)
	mkdir -p $(OUT)
	$(AARCH64_CXX) -o $@ $< $(LIB_FLAGS)

test:
	$(MAKE) -C test/conv3d_fix all

check:
	$(MAKE) -C test/conv3d_fix check

clean:
	rm -rf $(OUT)
	$(MAKE) -C test/conv3d_fix clean

.PHONY: all lib test check clean
