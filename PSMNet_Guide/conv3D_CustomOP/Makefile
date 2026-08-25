SYSROOT ?= /<CrossCompile path>/petalinux_sdk_2022.2/sysroots/cortexa72-cortexa53-xilinx-linux
CXX ?= /tools/X22/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++

ROOT := ..
OUT := aarch64

CXXFLAGS := --sysroot=$(SYSROOT) -O2 -std=c++17 -fPIC -shared \
            -Wall -Wextra -Wno-unused-parameter \
            -I$(ROOT)/include -I$(SYSROOT)/usr/include \
            -Wl,--no-undefined -L$(SYSROOT)/usr/lib \
            -lvitis_ai_library-runner_helper -lvart-runner -lxir -lglog -lpthread

TARGETS := $(OUT)/libvart_op_imp_conv3d-fix.so $(OUT)/libvart_op_imp_transposed-conv3d-fix.so

all: $(OUT) $(TARGETS)

$(OUT):
	mkdir -p $(OUT)

$(OUT)/libvart_op_imp_conv3d-fix.so: conv3d-fix/conv3d-fix.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(OUT)/libvart_op_imp_transposed-conv3d-fix.so: transposed-conv3d-fix/transposed-conv3d-fix.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf $(OUT)

.PHONY: all clean