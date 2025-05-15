#include <iostream>
#include <fstream>
#include "neural_network.hpp"
#include "loss_function.hpp"
#include "fully_connected_layer.hpp"
#include "relu.hpp"
#include "identity.hpp"
#include "mean_squared_error_loss.hpp"
#include "cross_entropy_loss.hpp"
#include "mnist.cpp"

using namespace std;

int main() {

    // train_test_mnist();
    infer_mnist();

    return 0;
}