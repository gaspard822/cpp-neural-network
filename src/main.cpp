#include <iostream>
#include <fstream>
#include "mlp/mnist.cpp"
#include "transformer/translation.cpp"

using namespace std;


int main() {

    // train_test_mnist();
    // infer_mnist();

    TrainingConfig cfg;
    infer_live_translation(cfg);

    return 0;
}