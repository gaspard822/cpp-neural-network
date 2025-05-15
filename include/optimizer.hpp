#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "layer.hpp"

class NeuralNetwork;

enum class OptimizerType {
    ADAM,
    VANILLA_SGD
}; 

/**
 * Abstract base class for all optimizers. Defines the interface for updating parameters during training.
 */
class Optimizer {
    protected:
        // Pointer to the neural network that this optimizer updates
        NeuralNetwork* nn;

    public:
        /**
         * Constructs an optimizer for the given neural network.
         * @param new_nn Pointer to the neural network to be optimized
         */
        Optimizer(NeuralNetwork* new_nn);

        virtual ~Optimizer() = default;

        /**
         * Prepares internal optimizer state for a new layer.
         * Called when adding a layer to the network in NeuralNetwork::add_layer(Layer* layer)
         * @param layer Pointer to the layer being added
         */
        virtual void update_optimizer(Layer* layer) = 0;

        /**
         * Applies the optimizer's update rule to the parameters of the layer at the given index.
         * @param layer_index Index of the layer to update
         */
        virtual void update_parameters(int layer_index) const = 0;

        /**
         * Simple setter to set or replace the neural network associated with the optimizer.
         * @param new_nn Pointer to the new neural network
         */
        void set_network(NeuralNetwork* new_nn);

        /**
         * Returns the type of the optimizer.
         * @return OptimizerType Enum value indicating the optimizer type
         */
        virtual OptimizerType get_type() const = 0;
};

#endif