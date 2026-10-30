# glmbench (ggml MUL_MAT_ID baseline) and glmref (fp64 reference, needs the CPU backend) against a llama.cpp ggml tree:
#   cmake -S tools/glm53 -B $W/build-cpu -DGGML_SRC=$W/llama/ggml -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
# SYCL: build-sycl.sh (oneAPI container).
cmake_minimum_required(VERSION 3.16)
project(glmbench CXX C)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(${GGML_SRC} ggml)
add_executable(glmbench glmbench.cpp)
target_link_libraries(glmbench PRIVATE ggml ggml-base)
if(TARGET ggml-cpu)
  add_executable(glmref glmref.cpp)
  target_link_libraries(glmref PRIVATE ggml-base ggml-cpu)
endif()
