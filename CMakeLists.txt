cmake_minimum_required(VERSION 3.10)
project(BasicNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-O3)

include_directories(include)


set(SOURCES
    src/main.cpp
    src/core/adam_optimizer.cpp
    src/core/cross_entropy_loss.cpp
    src/core/identity.cpp
    src/core/mean_squared_error_loss.cpp
    src/core/mlx_utils.cpp
    src/core/network.cpp
    src/core/optimizer.cpp
    src/core/relu.cpp
    src/core/sigmoid.cpp
    src/core/vanilla_sgd_optimizer.cpp
    src/mlp/fully_connected_layer.cpp
    src/mlp/neural_network.cpp
    src/transformer/bpe_tokenizer.cpp
    src/transformer/decoder.cpp
    src/transformer/encoder.cpp
    src/transformer/feed_forward.cpp
    src/transformer/input_layer.cpp
    src/transformer/layer_norm.cpp
    src/transformer/linear_layer.cpp
    src/transformer/multi_head_attention.cpp
    src/transformer/transformer.cpp
)

add_executable(basic_nn ${SOURCES})

find_library(MLX_LIB mlx PATHS /opt/homebrew/lib REQUIRED)
include_directories(/opt/homebrew/include)

target_link_libraries(basic_nn PRIVATE ${MLX_LIB})
