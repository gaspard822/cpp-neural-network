#include "core/network.hpp"

Network::Network(LossFunction* loss, Optimizer* optim) : loss_function(loss), optimizer(optim) {}