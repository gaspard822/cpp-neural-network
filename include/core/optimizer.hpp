#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <fstream>
#include "core/layer.hpp"

class Network;

enum class OptimizerType {
    ADAM,
    VANILLA_SGD
}; 

/**
 * Abstract base class for all optimizers. Defines the interface for updating parameters during training.
 */
class Optimizer {
    protected:
        // Pointer to the network that this optimizer updates
        Network* network;

    public:
        /**
         * Constructs an optimizer for the given neural network.
         * @param new_network Pointer to the network to be optimized
         */
        Optimizer(Network* new_network);

        virtual ~Optimizer() = default;

        /**
         * Prepares internal optimizer state for a new network.
         */
        virtual void update_optimizer() = 0;

        /**
         * Applies the optimizer's update rule to the parameters of the network.
         */
        virtual void update_parameters() const = 0;

        /**
         * Simple setter to set or replace the neural network associated with the optimizer.
         * @param new_nn Pointer to the new neural network
         */
        void set_network(Network* new_network);

        /**
         * Returns the type of the optimizer.
         * @return OptimizerType Enum value indicating the optimizer type
         */
        virtual OptimizerType get_type() const = 0;

        virtual void save(ofstream& file) const = 0;
        virtual void load(ifstream& file) = 0;
};

#endif