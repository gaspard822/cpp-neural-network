#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "neural_network.hpp"
#include "fully_connected_layer.hpp"
#include "relu.hpp"
#include "sigmoid.hpp"
#include "identity.hpp"
#include "mean_squared_error_loss.hpp"
#include "cross_entropy_loss.hpp"
#include "adam_optimizer.hpp"
#include "vanilla_sgd_optimizer.hpp"

using namespace std;

NeuralNetwork::NeuralNetwork() {
    loss_function = nullptr;
    optimizer = nullptr;
}

NeuralNetwork::NeuralNetwork(LossFunction* loss, Optimizer* optim) : loss_function(loss), optimizer(optim) {
    optimizer->set_network(this);
}

NeuralNetwork::NeuralNetwork(const string& loss, const string& optim) {
    if (loss == "MeanSquaredError") {
        loss_function = new MeanSquaredError();
    } else if (loss == "CrossEntropy") {
        loss_function = new CrossEntropy();
    } else {
        throw runtime_error("The loss function given to the network was not recognized");
    }

    if (optim == "VanillaSGD") {
        optimizer = new VanillaSGDOptimizer(this, 0.02);
    } else if (optim == "Adam") {
        optimizer = new AdamOptimizer(this);
    } else {
        throw runtime_error("The optimizer given to the network was not recognized");
    }
}

NeuralNetwork::~NeuralNetwork() {
    for (Layer* layer : layers) {
        delete layer;
    }
    if (loss_function) delete loss_function;
    if (optimizer) delete optimizer;
}

void NeuralNetwork::add_layer(Layer* layer) {
    layers.push_back(layer);
    optimizer->update_optimizer(layer);
}

MatrixXd NeuralNetwork::forward(const MatrixXd& input) {
    const MatrixXd* activation = &input;
    for (Layer* layer: layers) {
        layer->forward(*activation);
        activation = &layer->get_output();
    }
    return *activation;
}

void NeuralNetwork::backward(const MatrixXd& y_true, const MatrixXd& y_pred) {
    // First compute the derivative of the loss with respect to the loss function
    MatrixXd d_loss = loss_function->derivative(y_true, y_pred);
    // Propagate the gradients with respect to each layer back into the network and update the parameters accordingly
    int num_layers = layers.size();
    for (int i = num_layers - 1; i >= 0; i--) {
        d_loss = layers[i]->backward(d_loss);
        optimizer->update_parameters(i);
    }
}

// If the argument batch_size is <= 0, then no mini-batching is done
void NeuralNetwork::train(const MatrixXd& X_train, const MatrixXd& Y_train, int epochs, int batch_size,
                          const MatrixXd& X_val, const MatrixXd& Y_val, bool early_stopping, bool verbose) {
    int patience = 10;
    int epochs_without_improvement = 0;
    double best_val_loss = __DBL_MAX__;
    double current_error;
    chrono::time_point<chrono::high_resolution_clock> start, end;

    for (int i = 0; i < epochs; i++) {
        if (i % 10 == 0) {
            start = chrono::high_resolution_clock::now();
            cout << "Epoch " << i << endl;
        }

        // Creating the batches
        auto batch_time_start = chrono::high_resolution_clock::now();
        MatrixXd X_batch;
        MatrixXd Y_batch;
        if (batch_size > 0) {
            vector<int> indices(X_train.rows());
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});

            X_batch.resize(batch_size, X_train.cols());
            Y_batch.resize(batch_size, Y_train.cols());

            for (int i = 0; i < batch_size; ++i) {
                X_batch.row(i) = X_train.row(indices[i]);
                Y_batch.row(i) = Y_train.row(indices[i]);
            }
        } else {
            X_batch = X_train;
            Y_batch = Y_train;
        }
        auto batch_time_end = chrono::high_resolution_clock::now();

        // Forward+Backward passes
        auto forward_time_start = chrono::high_resolution_clock::now();
        MatrixXd forward_X_batch = forward(X_batch);
        auto forward_time_end = chrono::high_resolution_clock::now();
        auto backward_time_start = chrono::high_resolution_clock::now();
        backward(Y_batch, forward_X_batch);
        auto backward_time_end = chrono::high_resolution_clock::now();
        
        if (verbose) {
            cout << "Time for creating the batch: " << chrono::duration_cast<chrono::milliseconds>(batch_time_end - batch_time_start).count() << "ms" << endl;
            cout << "Time for forwarding the batch: " << chrono::duration_cast<chrono::milliseconds>(forward_time_end - forward_time_start).count() << "ms" << endl;
            cout << "Time for backwarding the batch: " << chrono::duration_cast<chrono::milliseconds>(backward_time_end - backward_time_start).count() << "ms" << endl;
        }

        // If some validation set is defined, compute and print error
        if (X_val.rows() > 0 && X_val.cols() > 0 && Y_val.rows() > 0 && Y_val.cols() > 0) {
            auto infer_time_start = chrono::high_resolution_clock::now();
            MatrixXd infer_X_val = infer(X_val);
            auto infer_time_end = chrono::high_resolution_clock::now();
            auto validation_error_time_start = chrono::high_resolution_clock::now();
            current_error = loss_function->compute(Y_val, infer_X_val);
            auto validation_error_time_end = chrono::high_resolution_clock::now();
            cout << "Current error: " << current_error << endl;

            if (verbose) {
                cout << "Time for inferring the validation set: " << chrono::duration_cast<chrono::milliseconds>(infer_time_end - infer_time_start).count() << "ms" << endl;
                cout << "Time for computing the validation error: " << chrono::duration_cast<chrono::milliseconds>(validation_error_time_end - validation_error_time_start).count() << "ms" << endl;
            }

            if (early_stopping) {
                if (current_error - best_val_loss < 0) {
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

        if ((i+1) % 10 == 0) {
            end = chrono::high_resolution_clock::now();
            cout << "           Execution Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        }

        if (loss_function->get_type() == LossFunctionType::CROSSENTROPY && (i+1) % 10 == 0
            && X_val.rows() > 0 && X_val.cols() > 0 && Y_val.rows() > 0 && Y_val.cols() > 0) {

            MatrixXd inference = infer(X_val);
            int num_samples = X_val.rows();
            VectorXd prediction(num_samples);
            VectorXd truth(num_samples);
            int correct_predictions = 0;
            for (int i = 0; i < num_samples; i++) {
                Index pred, label;
                inference.row(i).maxCoeff(&pred);
                Y_val.row(i).maxCoeff(&label);
                if (pred == label) correct_predictions += 1;
            }
            cout << "           Accuracy: " << 100.0 * (double) correct_predictions / (double) num_samples << "%" << endl;
            
        }

    }
}

MatrixXd NeuralNetwork::infer(const MatrixXd& input) const {
    MatrixXd activation = input;
    for (Layer* layer: layers) {
        activation = layer->infer(activation);
    }
    return activation;
}

void NeuralNetwork::save_model(const string& path) const {
    // In this function, we save:
    // 1. The number of layers
    // 2. The type of the optimizer
    // 3. For each layer, we save:
    //    a. The type of the layer
    //    b. The input and output dimensions
    //    c. The activation function
    //    d. The parameters and internal state (weights, bias, scale, shift, running mean, running variance,
    //       inverse of the squared variance)

    ofstream file(path);
    file << layers.size() << "\n";
    if (optimizer->get_type() == OptimizerType::ADAM) {
        file << "Adam" << "\n";
    } else if (optimizer->get_type() == OptimizerType::VANILLA_SGD) {
        file << "VanillaSGD" << "\n";
    } else {
        throw runtime_error("No optimizer is defined, the network can hence not be saved");
    }
    for (auto* layer : layers) {
        auto* fc = dynamic_cast<FullyConnectedLayer*>(layer);
        if (fc) {
            file << "FullyConnected\n";
            // fc->get_weights().cols() <=> input_size
            // fc->get_weights().cols() <=> output_size
            file << fc->get_weights().cols() << " " << fc->get_weights().rows() << "\n";
            file << fc->get_activation_name() << "\n";
            file << fc->get_weights() << "\n";
            file << fc->get_bias().transpose() << "\n";
            file << fc->get_gamma() << "\n";
            file << fc->get_beta() << "\n";
            file << fc->get_running_mean() << "\n";
            file << fc->get_running_variance() << "\n";
            file << fc->get_inv_sqrt_var_plus_epsilon() << "\n";
        }
    }
    file << loss_function->get_loss_name() << "\n";
}

void NeuralNetwork::load_model(const string& filename) {
    // This function should be called as follows: "NeuralNetwork nn; nn.load_model(path);"
    // Here, we restore a network previously saved with save_model()
    
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
        throw runtime_error("The optimizer could not be recognized upon loading of the network");
    }

    layers.clear();

    for (int i = 0; i < num_layers; i++) {
        string layer_type;
        file >> layer_type;
        if (layer_type == "FullyConnected") {
            int input_size, output_size;
            file >> input_size >> output_size;

            string act_name;
            file >> act_name;
            ActivationFunction* act = nullptr;
            if (act_name == "relu") act = new Relu();
            if (act_name == "sigmoid") act = new Sigmoid();
            if (act_name == "identity") act = new Identity();

            MatrixXd weights(output_size, input_size);
            for (int r = 0; r < output_size; r++) {
                for (int c = 0; c < input_size; c++) {
                    file >> weights(r, c);
                }
            }

            VectorXd bias(output_size);
            for (int j = 0; j < output_size; j++) {
                file >> bias(j);
            }

            RowVectorXd gamma(input_size);
            for (int j = 0; j < input_size; j++) {
                file >> gamma(j);
            }

            RowVectorXd beta(input_size);
            for (int j = 0; j < input_size; j++) {
                file >> beta(j);
            }

            RowVectorXd running_mean(input_size);
            for (int j = 0; j < input_size; j++) {
                file >> running_mean(j);
            }

            RowVectorXd running_variance(input_size);
            for (int j = 0; j < input_size; j++) {
                file >> running_variance(j);
            }

            RowVectorXd inv_sqrt_var_plus_epsilon(input_size);
            for (int j = 0; j < input_size; j++) {
                file >> inv_sqrt_var_plus_epsilon(j);
            }

            FullyConnectedLayer* fc = new FullyConnectedLayer(act, weights, bias, gamma, beta);
            fc->set_running_mean(running_mean);
            fc->set_running_variance(running_variance);
            fc->set_inv_sqrt_var_plus_epsilon(inv_sqrt_var_plus_epsilon);
            add_layer(fc);
        } else {
            throw runtime_error("The type of layer was not recognized during the model loading");
        }
    }

    string loss_function_name;
    file >> loss_function_name;
    loss_function = nullptr;
    if (loss_function_name == "mse") loss_function = new MeanSquaredError();
    if (loss_function_name == "cross-entropy") loss_function = new CrossEntropy();
}

const vector<Layer*>& NeuralNetwork::get_layers() const {
    return layers;
}

void NeuralNetwork::set_optimizer(Optimizer* optim) {
    optimizer = optim;
}