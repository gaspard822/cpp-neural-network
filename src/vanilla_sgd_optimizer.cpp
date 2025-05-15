#include "vanilla_sgd_optimizer.hpp"
#include "neural_network.hpp"

VanillaSGDOptimizer::VanillaSGDOptimizer(NeuralNetwork* new_nn, double stepsize) : Optimizer(new_nn), stepsize(stepsize) {}

void VanillaSGDOptimizer::update_optimizer(Layer* layer) {}

void VanillaSGDOptimizer::update_parameters(int layer_index) const {
    // Get the gradients and the parameters of the layer
    unique_ptr<Gradients> gradients = nn->get_layers()[layer_index]->get_gradients();
    unique_ptr<Gradients> parameters = nn->get_layers()[layer_index]->get_params();

    if (nn->get_layers()[layer_index]->get_type() == LayerType::FULLY_CONNECTED_LAYER) {
        // Get the gradients and the parameters of the fully connected layer
        FCGradients* grads = gradients.get()->as_fc_gradients();
        FCGradients* params = parameters.get()->as_fc_gradients();
        if (!grads || !params) throw runtime_error("The gradients or parameters of layer " + to_string(layer_index) + " are not stored as FCGradients");

        // Update the parameters
        params->d_weights = params->d_weights - stepsize * grads->d_weights;
        params->d_bias = params->d_bias - stepsize * grads->d_bias;
        params->d_gamma = params->d_gamma - stepsize * grads->d_gamma;
        params->d_beta = params->d_beta - stepsize * grads->d_beta;
    } else {
        throw runtime_error("The type of layer " + to_string(layer_index) + " was not recognized");
    }
}

OptimizerType VanillaSGDOptimizer::get_type() const {
    return OptimizerType::VANILLA_SGD;
}