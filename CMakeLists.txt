cmake_minimum_required(VERSION 3.10)
project(BasicNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-O2)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

set(SOURCES
    src/adam_optimizer.cpp
    src/cross_entropy_loss.cpp
    src/fully_connected_layer.cpp
    src/identity.cpp
    src/main.cpp
    src/mean_squared_error_loss.cpp
    src/neural_network.cpp
    src/optimizer.cpp
    src/relu.cpp
    src/sigmoid.cpp
    src/vanilla_sgd_optimizer.cpp
)

add_executable(basic_nn ${SOURCES})
