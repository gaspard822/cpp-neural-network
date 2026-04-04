#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

#include "core/layer.hpp"
#include "core/activation_function.hpp"


/**
 * Implements a fully connected layer with batch normalization and activation.
 */
class FullyConnectedLayer : public Layer {
    private:
        // The mathematical signification of each of these can be found in the "Technical details" section of the README
        mlx::core::array gamma, beta, d_gamma, d_beta, running_mean, running_variance, inv_sqrt_var_plus_epsilon;
        mlx::core::array weights, d_weights, a_hat, a_bar, z;
        mlx::core::array bias, d_bias;
        std::vector<TrainableParameter> params;
        float momentum;
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

        ~FullyConnectedLayer();

        /**
         * Does the forward pass with batch normalization and activation. See the "Technical details" section of the
         * README for context on the internal variable names used in this function.
         * @param input Input matrix (samples x features)
         */
        void forward(const mlx::core::array& input) override;

        /**
        * Does the backward pass, computes and saves the gradient with respect to the layer input in d_input.
        * See the "Technical details" section of the README for context on the internal variable names used in this function.
        * @param d_output Gradient from the following layer
        */
        void backward(const mlx::core::array& d_output) override;

        /**
         * Performs inference without modifying the internal state of the fully connected layer.
         * @param layer_input Input of the layer (samples x features)
         * @return mlx::core::array Output of the layer
         */
        mlx::core::array infer(const mlx::core::array& layer_input) const override;

        // Various straightforward getters
        const std::vector<TrainableParameter>& get_parameters() const override;
        const mlx::core::array& get_output() const override;
        const mlx::core::array& get_d_input() const override;

        /**
         * Returns the name of the layer (used in save_model(const string& path) and load_model(const string& path)).
         * @return string Name of the loss function
         */
        std::string get_layer_name() const override;

        /**
         * Returns the name of the activation function used by the layer (used in save_model(const string& path) and
         * load_model(const string& path)).
         * @return string Name of the loss function
         */
        std::string get_activation_name() const override;

        /**
         * Returns the type of the layer (fully connected layer).
         * @return LayerType Enum value corresponding to the layer type
         */
        LayerType get_type() const override;

        void save(std::ofstream& file) const override;
        void load(std::ifstream& file) override;
};

#endif
