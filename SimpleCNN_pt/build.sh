#!/bin/bash
result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
    OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
    OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

CXX=${CXX:-g++}

for file in $(ls *.cpp); do
    filename=${file%.*}
    $CXX -std=c++17 -O2 -I. -o ${filename} ${file} \
        -lvart-runner -lvart-mem-manager -lvart-util -lxir \
        -lglog \
        ${OPENCV_FLAGS} \
        -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
        -pthread
done
