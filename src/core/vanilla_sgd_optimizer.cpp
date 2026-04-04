#include "core/vanilla_sgd_optimizer.hpp"
#include "mlp/neural_network.hpp"
#include "transformer/transformer.hpp"

using namespace std;
namespace mx = mlx::core;

VanillaSGDOptimizer::VanillaSGDOptimizer(Network* new_nn, float stepsize) : Optimizer(new_nn), stepsize(stepsize) {}

void VanillaSGDOptimizer::update_optimizer() {}

void VanillaSGDOptimizer::update_parameters() const {
    vector<Layer*> layers = network->get_layers();
    for (Layer* layer : layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            mx::array& weights = *p.value;
            mx::array& gradients = *p.grad;

            weights = weights - stepsize * gradients;
        }
    }
}

OptimizerType VanillaSGDOptimizer::get_type() const {
    return OptimizerType::VANILLA_SGD;
}

void VanillaSGDOptimizer::save(ofstream& file) const {}

void VanillaSGDOptimizer::load(ifstream& file) {}
