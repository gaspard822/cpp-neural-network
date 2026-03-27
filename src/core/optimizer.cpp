#include "core/optimizer.hpp"

Optimizer::Optimizer(Network* new_network) : network(new_network) {}

void Optimizer::set_network(Network* new_network) {
    network = new_network;
}
