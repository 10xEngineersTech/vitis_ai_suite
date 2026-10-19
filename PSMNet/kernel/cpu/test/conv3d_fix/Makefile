# Host-side validation build for the 3D fixed-point convolution kernel.
# Needs no VART/XIR: the glog stub lives in ./stub.
#
#   make            build the test binary
#   make check      build, then run the PyTorch goldens
#

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter
PYTHON ?= python3

COMMON := ../../ops/common/conv3d_fix_kernel.hpp ../../ops/common/util.hpp

all: test_conv3d_fix

test_conv3d_fix: test_conv3d_fix.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) -I stub -o $@ test_conv3d_fix.cpp -pthread

check: all
	$(PYTHON) gen_and_check.py

clean:
	rm -f test_conv3d_fix
	rm -rf cases

.PHONY: all check clean
