#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <mlx/mlx.h>
#include "mlp/mnist.cpp"
#include "transformer/translation.cpp"

using namespace std;
namespace mx = mlx::core;

int main() {

    // train_test_mnist();
    // infer_mnist();

    TrainingConfig cfg;
    // train_and_save_tokenizer(cfg);
    init_transformer_model(cfg);
    train_transformer_model(cfg, true);
    // infer_live_translation(cfg);

    return 0;
}
