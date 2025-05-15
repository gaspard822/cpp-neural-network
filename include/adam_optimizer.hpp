#ifndef ADAM_OPTIMIZER_HPP
#define ADAM_OPTIMIZER_HPP

#include "optimizer.hpp"
#include "fully_connected_layer.hpp"

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
         * @param new_nn Pointer to the neural network to optimizer
         * @param stepsize Learning rate
         * @param b1 First moment decay rate
         * @param b2 Second moment decay rate
         */
        AdamOptimizer(NeuralNetwork* new_nn, double stepsize, double b1, double b2);

        /**
         * Constructs an Adam optimizer with default hyperparameters
         * stepsize=0.001, b1=0.9, b2=0.999.
         * @param new_nn Pointer to the neural network to optimizer
         */
        AdamOptimizer(NeuralNetwork* new_nn);

        ~AdamOptimizer() = default;

        /**
         * Given a pointer to a layer, pushes a first and second moment vector with corresponding dimensions onto m and v.
         * Called when adding a layer to the network in NeuralNetwork::add_layer(Layer* layer).
         * @param layer Pointer to the layer added to the network
         */
        void update_optimizer(Layer* layer) override;

        /**
         * Applies Adam parameter update to the layer at the given index.
         * See Algorithm 1 in https://arxiv.org/pdf/1412.6980.
         * @param layer_index Index of the layer in the network
         */
        void update_parameters(int layer_index) const override;

        /**
         * Returns the type of optimizer (Adam).
         * @return OptimizerType Enum value corresponding to the optimizer type
         */
        OptimizerType get_type() const override;
};

#endif