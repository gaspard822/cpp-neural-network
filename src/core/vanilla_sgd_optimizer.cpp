#include "core/vanilla_sgd_optimizer.hpp"
#include "mlp/neural_network.hpp"
#include "transformer/transformer.hpp"

VanillaSGDOptimizer::VanillaSGDOptimizer(Network* new_nn, double stepsize) : Optimizer(new_nn), stepsize(stepsize) {}

void VanillaSGDOptimizer::update_optimizer() {}

void VanillaSGDOptimizer::update_parameters_mlp(MultiLayerPerceptronNetwork* mlp) const {
    vector<Layer*> layers = mlp->get_layers();
    int num_layers = layers.size();
    for (int layer_index = num_layers - 1; layer_index >= 0; layer_index--) {
        // Get the gradients and the parameters of the layer
        unique_ptr<Gradients> gradients = layers[layer_index]->get_gradients();
        unique_ptr<Gradients> parameters = layers[layer_index]->get_params();

        if (layers[layer_index]->get_type() == LayerType::FULLY_CONNECTED_LAYER) {
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
}

void VanillaSGDOptimizer::update_parameters_transformer(TransformerNetwork* transformer) const {
    // TODO: Implement
}

void VanillaSGDOptimizer::update_parameters() const {
    switch(network->get_type()) {
        case NetworkType::MULTI_LAYER_PERCEPTRON:
            update_parameters_mlp(static_cast<MultiLayerPerceptronNetwork*>(network));
            break;
        case NetworkType::TRANSFORMER:
            update_parameters_transformer(static_cast<TransformerNetwork*>(network));
            break;
        default:
            throw runtime_error("The network type was not recognized");
    }
}

OptimizerType VanillaSGDOptimizer::get_type() const {
    return OptimizerType::VANILLA_SGD;
}