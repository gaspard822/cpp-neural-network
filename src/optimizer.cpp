#include "optimizer.hpp"

Optimizer::Optimizer(NeuralNetwork* new_nn) : nn(new_nn) {}

void Optimizer::set_network(NeuralNetwork* new_nn) {
    nn = new_nn;
}