#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include "mlp/neural_network.hpp"
#include "mlp/fully_connected_layer.hpp"
#include "core/mlx_utils.hpp"
#include "core/relu.hpp"
#include "core/sigmoid.hpp"
#include "core/identity.hpp"
#include "core/mean_squared_error_loss.hpp"
#include "core/cross_entropy_loss.hpp"
#include "core/adam_optimizer.hpp"
#include "core/vanilla_sgd_optimizer.hpp"

using namespace std;
namespace mx = mlx::core;

MultiLayerPerceptronNetwork::MultiLayerPerceptronNetwork() {
    loss_function = nullptr;
    optimizer = nullptr;
}

MultiLayerPerceptronNetwork::MultiLayerPerceptronNetwork(LossFunction* loss, Optimizer* optim) : Network(loss, optim) {
    optimizer->set_network(this);
}

MultiLayerPerceptronNetwork::MultiLayerPerceptronNetwork(const string& loss, const string& optim) {
    if (loss == "MeanSquaredError") {
        loss_function = new MeanSquaredError();
    } else if (loss == "CrossEntropy") {
        loss_function = new CrossEntropy();
    } else {
        throw runtime_error("The loss function given to the network was not recognized");
    }

    if (optim == "VanillaSGD") {
        optimizer = new VanillaSGDOptimizer(this, 0.2);
    } else if (optim == "Adam") {
        optimizer = new AdamOptimizer(this);
    } else {
        throw runtime_error("The optimizer given to the network was not recognized");
    }
}

MultiLayerPerceptronNetwork::~MultiLayerPerceptronNetwork() {
    for (Layer* layer : layers) {
        delete layer;
    }
    if (loss_function) delete loss_function;
    if (optimizer) delete optimizer;
}

void MultiLayerPerceptronNetwork::add_layer(Layer* layer) {
    layers.push_back(layer);
}

mx::array MultiLayerPerceptronNetwork::forward(const mx::array& input) const {
    const mx::array* activation = &input;
    for (Layer* layer: layers) {
        layer->forward(*activation);
        activation = &layer->get_output();
    }
    return *activation;
}

void MultiLayerPerceptronNetwork::backward(const mx::array& y_true, const mx::array& y_pred) const {
    mx::array d_loss_buf = loss_function->derivative(y_true, y_pred);
    const mx::array* d_loss = &d_loss_buf;
    int num_layers = layers.size();
    for (int i = num_layers - 1; i >= 0; i--) {
        layers[i]->backward(*d_loss);
        d_loss = &layers[i]->get_d_input();
    }
    optimizer->update_parameters();
}

// If the argument batch_size is <= 0, then no mini-batching is done
void MultiLayerPerceptronNetwork::train(const mx::array& X_train, const mx::array& Y_train, int epochs, int batch_size,
                          const mx::array& X_val, const mx::array& Y_val, bool early_stopping) {
    int patience = 10;
    int epochs_without_improvement = 0;
    float best_val_loss = numeric_limits<float>::max();
    float current_error;
    int N = X_train.shape(0);
    bool has_val = X_val.ndim() >= 2 && X_val.shape(0) > 0;

    for (int i = 0; i < epochs; i++) {
        if (i % 10 == 0) {
            cout << "Epoch " << i << endl;
        }

        // Creating the batches
        mx::array X_batch = X_train;
        mx::array Y_batch = Y_train;
        if (batch_size > 0) {
            vector<int> indices(N);
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});
            indices.resize(batch_size);
            mx::array idx = mx::array(indices.data(), {batch_size}, mx::int32);
            X_batch = mx::take(X_train, idx, 0);
            Y_batch = mx::take(Y_train, idx, 0);
        }

        // Forward+Backward passes
        mx::array forward_X_batch = forward(X_batch);
        backward(Y_batch, forward_X_batch);

        if (has_val) {
            mx::array infer_X_val = infer(X_val);
            current_error = loss_function->compute(Y_val, infer_X_val);
            cout << "Current error: " << current_error << endl;

            if (early_stopping) {
                if (current_error < best_val_loss) {
                    best_val_loss = current_error;
                    epochs_without_improvement = 0;
                } else {
                    epochs_without_improvement += 1;
                    if (epochs_without_improvement >= patience) {
                        cout << "Early stopping at epoch " << i << endl;
                        break;
                    }
                }
            }
        }

        if (loss_function->get_type() == LossFunctionType::CROSSENTROPY && (i+1) % 10 == 0 && has_val) {
            mx::array inference = infer(X_val);
            int num_samples = X_val.shape(0);
            int correct = mx::sum(mx::equal(mx::argmax(inference, 1), mx::argmax(Y_val, 1))).item<int>();
            cout << "           Accuracy: " << 100.0 * (float) correct / (float) num_samples << "%" << endl;
        }
    }
}

mx::array MultiLayerPerceptronNetwork::infer(const mx::array& input) const {
    mx::array activation = input;
    for (Layer* layer: layers) {
        activation = layer->infer(activation);
    }
    return activation;
}

void MultiLayerPerceptronNetwork::save_model(const string& path) const {
    ofstream file(path);
    file << layers.size() << "\n";
    if (optimizer->get_type() == OptimizerType::ADAM) {
        file << "Adam\n";
    } else if (optimizer->get_type() == OptimizerType::VANILLA_SGD) {
        file << "VanillaSGD\n";
    } else {
        throw runtime_error("No optimizer is defined, the network can not be saved");
    }
    for (auto* layer : layers) {
        layer->save(file);
    }
    file << loss_function->get_loss_name() << "\n";
}

void MultiLayerPerceptronNetwork::load_model(const string& filename) {
    ifstream file(filename);
    int num_layers;
    file >> num_layers;
    string optimizer_type;
    file >> optimizer_type;
    if (optimizer_type == "Adam") {
        optimizer = new AdamOptimizer(this);
    } else if (optimizer_type == "VanillaSGD") {
        optimizer = new VanillaSGDOptimizer(this, 0.02);
    } else {
        throw runtime_error("The optimizer could not be recognized.");
    }

    layers.clear();

    for (int i = 0; i < num_layers; i++) {
        string layer_type;
        file >> layer_type;
        if (layer_type == "FullyConnectedLayer") {
            string act_name;
            file >> act_name;
            ActivationFunction* act = nullptr;
            if (act_name == "relu") act = new Relu();
            else if (act_name == "sigmoid") act = new Sigmoid();
            else if (act_name == "identity") act = new Identity();

            int input_size, output_size;
            file >> input_size >> output_size;

            FullyConnectedLayer* fc = new FullyConnectedLayer(act, input_size, output_size);
            fc->load(file);
            add_layer(fc);
        } else {
            throw runtime_error("The type of layer was not recognized during the model loading");
        }
    }

    string loss_function_name;
    file >> loss_function_name;
    loss_function = nullptr;
    if (loss_function_name == "mse") loss_function = new MeanSquaredError();
    else if (loss_function_name == "cross-entropy") loss_function = new CrossEntropy();
}

const vector<Layer*>& MultiLayerPerceptronNetwork::get_layers() const {
    return layers;
}

Optimizer* MultiLayerPerceptronNetwork::get_optimizer() const {
    return optimizer;
}

NetworkType MultiLayerPerceptronNetwork::get_type() const {
    return NetworkType::MULTI_LAYER_PERCEPTRON;
}

void MultiLayerPerceptronNetwork::set_optimizer(Optimizer* optim) {
    optimizer = optim;
}
