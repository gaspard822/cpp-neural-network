#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <mlx/mlx.h>
#include "mlp/mnist.cpp"
#include "transformer/translation.cpp"

using namespace std;
namespace mx = mlx::core;


int main(int argc, char* argv[]) {
    string mode = (argc > 1) ? argv[1] : "translate";

    if (mode == "mnist-train") {
        train_test_mnist();
    } else if (mode == "mnist-infer") {
        infer_mnist();
    } else if (mode == "train-tokenizer") {
        TrainingConfig cfg;
        train_and_save_tokenizer(cfg);
    } else if (mode == "init-model") {
        TrainingConfig cfg;
        init_transformer_model(cfg);
    } else if (mode == "train") {
        TrainingConfig cfg;
        train_transformer_model(cfg, true);
    } else if (mode == "translate") {
        TrainingConfig cfg;
        infer_live_translation(cfg);
    } else {
        cout << "Usage: ./nn [mnist-train|mnist-infer|train-tokenizer|init-model|train|translate]" << endl;
    }

    return 0;
}
