#ifndef VANILLA_SGD_OPTIMIZER_HPP
#define VANILLA_SGD_OPTIMIZER_HPP

#include "optimizer.hpp"
#include "fully_connected_layer.hpp"

/**
 * Implementation of vanilla stochastic gradient descent. Uses a fixed learning rate.
 */
class VanillaSGDOptimizer : public Optimizer {
    private:
        // Learning rate for SGD updates
        double stepsize;
    public:
        /**
         * Constructs a vanilla SGD optimizer with a specified learning rate.
         * @param new_nn Pointer to the neural network to optimize
         * @param stepsize Learning rate
         */
        VanillaSGDOptimizer(NeuralNetwork* new_nn, double stepsize);

        ~VanillaSGDOptimizer() = default;
        
        /**
         * Implementation of the optimizer interface. In the case of vanilla SGD, doesn't actually do anything.
         * @param layer Pointer to the layer being added
         */
        void update_optimizer(Layer* layer) override;

        /**
         * Applies SGD update to the layer at the given index.
         * @param layer_index Index of the layer in the network
         */
        void update_parameters(int layer_index) const override;

        /**
         * Returns the type of optimizer (Vanilla SGD).
         * @return OptimizerType Enum value corresponding to the optimizer type
         */
        OptimizerType get_type() const override;
};

#endif