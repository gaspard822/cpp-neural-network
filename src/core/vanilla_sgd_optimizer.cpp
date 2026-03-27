#include "core/vanilla_sgd_optimizer.hpp"
#include "mlp/neural_network.hpp"
#include "transformer/transformer.hpp"

VanillaSGDOptimizer::VanillaSGDOptimizer(Network* new_nn, double stepsize) : Optimizer(new_nn), stepsize(stepsize) {}

void VanillaSGDOptimizer::update_optimizer() {}

void VanillaSGDOptimizer::update_parameters() const {
    vector<Layer*> layers = network->get_layers();
    for (Layer* layer : layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            auto weights = p.value();
            auto gradients = p.grad();

            weights -= stepsize * gradients;
        }
    }
}

OptimizerType VanillaSGDOptimizer::get_type() const {
    return OptimizerType::VANILLA_SGD;
}

void VanillaSGDOptimizer::save(ofstream& file) const {}

void VanillaSGDOptimizer::load(ifstream& file) {}
