#ifndef VANILLA_SGD_OPTIMIZER_HPP
#define VANILLA_SGD_OPTIMIZER_HPP

#include "core/optimizer.hpp"
#include "mlp/fully_connected_layer.hpp"
#include "mlp/neural_network.hpp"
#include "transformer/transformer.hpp"


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
        VanillaSGDOptimizer(Network* new_network, double stepsize);

        ~VanillaSGDOptimizer() = default;
        
        /**
         * Implementation of the optimizer interface. In the case of vanilla SGD, doesn't actually do anything.
         */
        void update_optimizer() override;

        /**
         * Applies SGD update to the layer at the given index.
         * @param layer_index Index of the layer in the network
         */
        void update_parameters() const override;
        void update_parameters_mlp(MultiLayerPerceptronNetwork* mlp) const;
        void update_parameters_transformer(TransformerNetwork* transformer) const;

        /**
         * Returns the type of optimizer (Vanilla SGD).
         * @return OptimizerType Enum value corresponding to the optimizer type
         */
        OptimizerType get_type() const override;
};

#endif