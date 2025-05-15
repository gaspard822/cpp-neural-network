#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include "layer.hpp"
#include "activation_function.hpp"

/**
 * Struct used to pass the gradients (and parameters as well) of a fully connected layer. Stored as references to avoid
 * data movement.
 */
struct FCGradients : public Gradients {
    MatrixXd& d_weights;
    VectorXd& d_bias;
    RowVectorXd& d_gamma;
    RowVectorXd& d_beta;

    FCGradients(MatrixXd& dw, VectorXd& dbi, RowVectorXd& dg, RowVectorXd& dbe);
    FCGradients* as_fc_gradients() override { return this; }
};

/**
 * Implements a fully connected layer with batch normalization and activation.
 */
class FullyConnectedLayer : public Layer {
    private:
        // The mathematical signification of each of these can be found in the "Technical details" section of the README
        // Some of these are stored during the forwarding in order to be used to compute gradients during the backpropagation
        RowVectorXd gamma, beta, d_gamma, d_beta, running_mean, running_variance, inv_sqrt_var_plus_epsilon;
        MatrixXd weights, d_weights, a_hat, a_bar, z;
        VectorXd bias, d_bias;
        double momentum;
        ActivationFunction* activation;
    
    public:
        /**
         * Constructs a fully connected layer with randomly initialized parameters, depending on the chosen activation
         * function (e.g. He initialization for ReLU).
         * @param activation Pointer to the activation function
         * @param input_size Number of input features
         * @param output_size Number of output features
         */
        FullyConnectedLayer(ActivationFunction* activation, int input_size, int output_size);

        /**
         * Constructs a fully connected layer with custom parameters.
         * @param activation Pointer to the activation function
         * @param init_weights Initial weights matrix
         * @param init_bias Initial bias vector
         * @param init_gamma Initial gamma for batch norm
         * @param init_beta Initial beta for batch norm
         */
        FullyConnectedLayer(ActivationFunction* activation,
                            const MatrixXd& init_weights,
                            const VectorXd& init_bias,
                            const RowVectorXd& init_gamma,
                            const RowVectorXd& init_beta);

        ~FullyConnectedLayer();

        /**
         * Does the forward pass with batch normalization and activation. See the "Technical details" section of the
         * README for context on the internal variable names used in this function.
         * @param input Input matrix (samples x features)
         */
        void forward(const MatrixXd& input) override;

        /**
        * Does the backward pass and computes input gradients. See the "Technical details" section of the
        * README for context on the internal variable names used in this function.
        * @param d_output Gradient from the following layer
        * @return MatrixXd Gradient with respect to the layer input
        */
        MatrixXd backward(const MatrixXd& d_output) override;

        /**
         * Performs inference without modifying the internal state of the fully connected layer.
         * @param layer_input Input of the layer (samples x features)
         * @return MatrixXd Output of the layer
         */
        MatrixXd infer(const MatrixXd& layer_input) const override;

        // Various straightforward getters
        unique_ptr<Gradients> get_gradients() override;
        unique_ptr<Gradients> get_params() override;
        const MatrixXd& get_weights() const;
        const VectorXd& get_bias() const;
        const RowVectorXd& get_gamma() const;
        const RowVectorXd& get_beta() const;
        const RowVectorXd& get_running_mean() const;
        const RowVectorXd& get_running_variance() const;
        const RowVectorXd& get_inv_sqrt_var_plus_epsilon() const;

        /**
         * Returns the name of the activation function used by the layer (used in
         * NeuralNetwork::save_model(const string& path) and NeuralNetwork::load_model(const string& path)).
         * @return string Name of the loss function
         */
        string get_activation_name() const override;

        /**
         * Returns the type of the layer (fully connected layer).
         * @return LayerType Enum value corresponding to the layer type
         */
        LayerType get_type() const override;

        // Various straightforward setters
        void set_running_mean(RowVectorXd new_mean);
        void set_running_variance(RowVectorXd new_mean);
        void set_inv_sqrt_var_plus_epsilon(RowVectorXd new_inv_sqrt_var_plus_epsilon);
};

#endif