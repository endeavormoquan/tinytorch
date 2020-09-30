# TinyPyTorch

An tiny implementation of [PyTorch](https://github.com/pytorch/pytorch).

Many features will be cut:
* CUDA/HIP
* OpenMP and other accelerators
* Parallelazation
* Python support
* and so on

Some core conponent will be kept
* Tensor/Storage and its Ops
* Dispatch
* Computation Graph
* Autograd
* and so on

Notice:
As a learning project, to save time, a lot of code is copied from [PyTorch](https://github.com/pytorch/pytorch).

# Usage and Test
## GTEST

```
cd build
./bin/xxx.test
```

## Build c10 as a shared lib

Build and install this project
```
cd build
cmake .. && make && make install
```

The lib will be installed in directory torch.

Create another project, write a `test_tinytorch.cpp`, then copy the CMakeLists.txt below.
```
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(test)
set(CMAKE_PREFIX_PATH /home/dingqimin/workspace/tinypytorch/torch)

add_executable(test test_tinytorch.cpp)

include_directories(${CMAKE_PREFIX_PATH}/include)
target_link_libraries(test ${CMAKE_PREFIX_PATH}/lib/libc10.so)
set_property(TARGET test PROPERTY CXX_STANDARD 14)
```

```
mkdir build && cd build
cmake .. && make
./test
```