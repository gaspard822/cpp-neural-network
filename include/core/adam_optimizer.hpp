#ifndef ADAM_OPTIMIZER_HPP
#define ADAM_OPTIMIZER_HPP

#include "core/optimizer.hpp"
#include "mlp/neural_network.hpp"
#include "transformer/transformer.hpp"

/**
 * The goal of this struct is to save the parameters of a layer, so that the Adam optimizer can store the first and 
 * second moment vectors and compute the new parameters accordingly.
 * As this struct is the structure that actually contains the parameter, they are not stored as references like in 
 * fully_connected_layer.hpp.
 */
struct OwnedFCGradients : public Gradients {
    MatrixXd d_weights;
    VectorXd d_bias;
    RowVectorXd d_gamma;
    RowVectorXd d_beta;

    OwnedFCGradients(MatrixXd dw, VectorXd dbi, RowVectorXd dg, RowVectorXd dbe);
    OwnedFCGradients* as_owned_fc_gradients() override { return this; }
};

/**
 * Implementation of the Adam optimization algorithm.
 */
class AdamOptimizer : public Optimizer {
    private:
        // Learning rate
        double stepsize;
        // Exponential decay rates for the moment estimates
        double b1, b2;
        // Small constant to prevent division by zero
        double epsilon;
        // First and second moment vectors for each layer
        vector<unique_ptr<OwnedFCGradients>> m, v;
        // Time step (incremented at each parameter update), mutable to allow const update
        mutable int t;

    public:
        /**
         * Constructs an Adam optimizer with custom hyperparameters.
         * @param new_network Pointer to the network to optimize
         * @param stepsize Learning rate
         * @param b1 First moment decay rate
         * @param b2 Second moment decay rate
         */
        AdamOptimizer(Network* new_network, double stepsize, double b1, double b2);

        /**
         * Constructs an Adam optimizer with default hyperparameters
         * stepsize=0.001, b1=0.9, b2=0.999.
         * @param new_network Pointer to the neural network to optimize
         */
        AdamOptimizer(Network* new_network);

        ~AdamOptimizer() = default;

        /**
         * Creates first and second moment vectors corresponding to the layers of the network.
         */
        void update_optimizer() override;
        void update_optimizer_mlp(MultiLayerPerceptronNetwork* mlp);
        void update_optimizer_transformer(TransformerNetwork* transformer);

        /**
         * Applies Adam parameter update to the layers of the network.
         * See Algorithm 1 in https://arxiv.org/pdf/1412.6980.
         */
        void update_parameters() const override;
        void update_parameters_mlp(MultiLayerPerceptronNetwork* mlp) const;
        void update_parameters_transformer(TransformerNetwork* transformer) const;

        /**
         * Returns the type of optimizer (Adam).
         * @return OptimizerType Enum value corresponding to the optimizer type
         */
        OptimizerType get_type() const override;
};

#endif