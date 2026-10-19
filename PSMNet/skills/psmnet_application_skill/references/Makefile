# Template: cross build for the KV260 with the PetaLinux 2022.2 SDK
# (pattern of application/Makefile).   make SDK=/path/to/petalinux_sdk_2022.2
SDK ?= /mnt/ssd2/abdul_raouf_khalid/petalinux_sdk_2022.2
SYSROOT ?= $(SDK)/sysroots/cortexa72-cortexa53-xilinx-linux
override CXX := $(SDK)/sysroots/x86_64-petalinux-linux/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-g++

CXX_FLAGS += --sysroot=$(SYSROOT) -I$(SYSROOT)/usr/include -I$(SYSROOT)/usr/include/opencv4
CXX_FLAGS += -std=c++17 -O3 -g -Wall

LD_FLAGS += -L$(SYSROOT)/usr/lib
LD_FLAGS += -lglog -lvitis_ai_library-model_config -lprotobuf
LD_FLAGS += -lvitis_ai_library-graph_runner -lvitis_ai_library-dpu_task
LD_FLAGS += -lvart-runner -lxir
LD_FLAGS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
LD_FLAGS += -pthread

all: demo_mynet

%.o: %.cpp mynet.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ -I. $<

demo_mynet: demo_mynet.o mynet.o
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(LD_FLAGS)

clean:
	rm -f *.o demo_mynet

.PHONY: all clean
