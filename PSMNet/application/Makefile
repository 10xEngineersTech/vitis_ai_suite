#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# KV260 (Zynq UltraScale+, DPUCZDX8G) cross build with the PetaLinux 2022.2 SDK.
# Override on the command line if the SDK lives elsewhere:
#   make SDK=/path/to/petalinux_sdk_2022.2
SDK ?= /mnt/ssd2/abdul_raouf_khalid/petalinux_sdk_2022.2
SYSROOT ?= $(SDK)/sysroots/cortexa72-cortexa53-xilinx-linux
override CXX := $(SDK)/sysroots/x86_64-petalinux-linux/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-g++
override LD := $(CXX)

CXX_FLAGS += --sysroot=$(SYSROOT)
CXX_FLAGS += -I$(SYSROOT)/usr/include
CXX_FLAGS += -I$(SYSROOT)/usr/include/opencv4
CXX_FLAGS += -std=c++17
CXX_FLAGS += -O3
CXX_FLAGS += -g
CXX_FLAGS += -Wall

# Vitis AI 3.0 libs ship in <sysroot>/usr/lib in the PetaLinux 2022.2 SDK.
LD_FLAGS += -L$(SYSROOT)/usr/lib
LD_FLAGS += -lglog
LD_FLAGS += -lvitis_ai_library-model_config
LD_FLAGS += -lprotobuf
LD_FLAGS += -lvitis_ai_library-graph_runner
LD_FLAGS += -lvitis_ai_library-dpu_task
LD_FLAGS += -lvart-runner
LD_FLAGS += -lxir
LD_FLAGS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_video
LD_FLAGS += -pthread

OBJS = psmnet.o resize.o cost_volume.o lut_sfm.o

all: demo_psmnet demo_psmnet_live

demo_psmnet.o: demo_psmnet.cpp psmnet.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ -I. $<

demo_psmnet_live.o: demo_psmnet_live.cpp psmnet.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ -I. $<

psmnet.o: psmnet.cpp psmnet.hpp resize.hpp cost_volume.hpp lut_sfm.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ -I. $<

lut_sfm.o: lut_sfm.cpp lut_sfm.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ -I. $<

resize.o: resize.cpp resize.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ -I. $<

cost_volume.o: cost_volume.cpp cost_volume.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ -I. $<

demo_psmnet: demo_psmnet.o $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(LD_FLAGS)

demo_psmnet_live: demo_psmnet_live.o $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $+ $(LD_FLAGS)

clean:
	rm -f *.o demo_psmnet demo_psmnet_live

.PHONY: all clean
