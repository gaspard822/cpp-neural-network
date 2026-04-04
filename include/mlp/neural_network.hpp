#ifndef MULTI_LAYER_PERCEPTRON_NETWORK_HPP
#define MULTI_LAYER_PERCEPTRON_NETWORK_HPP

#include <vector>
#include "core/network.hpp"
#include "core/layer.hpp"
#include "core/loss_function.hpp"
#include "core/optimizer.hpp"

using namespace std;

/**
 * Represents a feedforward neural network composed of sequential layers.
 */
class MultiLayerPerceptronNetwork : public Network {
    private:
        vector<Layer*> layers;
    
    public:
        /**
         * Constructs an empty neural network.
         */
        MultiLayerPerceptronNetwork();

        /**
         * Constructs a network with the given loss function and optimizer.
         * @param loss Pointer to the loss function
         * @param optim Pointer to the optimizer
         */
        MultiLayerPerceptronNetwork(LossFunction* loss, Optimizer* optim);

        /**
         * Constructs a network by selecting loss function and optimizer by name.
         * @param loss_function Name of the loss function ("MeanSquaredError" or "CrossEntropy")
         * @param optimizer Name of the optimizer ("VanillaSGD" or "Adam")
         */
        MultiLayerPerceptronNetwork(const string& loss_function, const string& optimizer);

        ~MultiLayerPerceptronNetwork();

        /**
         * Adds a new layer to the network and updates the optimizer accordingly.
         * @param layer Pointer to the layer to add
         */
        void add_layer(Layer* layer);

        /**
         * Does a full forward pass through the network.
         * @param input Input data matrix (samples x features)
         * @return mlx::core::array Output of the final layer
         */
        mlx::core::array forward(const mlx::core::array& input) const;

        /**
         * Does a full backward pass using the given target and prediction.
         * @param y_true Ground truth labels
         * @param y_pred Predicted outputs from the forward pass
         */
        void backward(const mlx::core::array& y_true, const mlx::core::array& y_pred) const;

        /**
         * Trains the network using the given training data and validation data.
         * @param X_train Input training data
         * @param Y_train Target training labels
         * @param epochs Number of training epochs
         * @param batch_size Size of each training batch
         * @param X_val Validation input data (pass mx::zeros({0}) if none)
         * @param Y_val Validation labels (pass mx::zeros({0}) if none)
         * @param early_stopping Whether to stop early based on validation performance
         * @param verbose Whether to print training progress
         */
        void train(const mlx::core::array& X_train, const mlx::core::array& Y_train, int epochs, int batch_size,
                   const mlx::core::array& X_val, const mlx::core::array& Y_val,
                   bool early_stopping = true);

        /**
         * Does inference on new data without modifying internal state.
         * @param input Input data matrix (samples x features)
         * @return mlx::core::array Network output
         */
        mlx::core::array infer(const mlx::core::array& input) const;

        /**
         * Saves the model's architecture and parameters to a .txt file.
         * @param path Filesystem path to save the model
         */
        void save_model(const string& path) const override;

        /**
         * Loads a model's architecture and parameters from a file.
         * @param path Filesystem path to the saved model
         */
        void load_model(const string& path) override;

        // Straightforward getter
        const vector<Layer*>& get_layers() const override;

        // Straightforward getter
        Optimizer* get_optimizer() const override;

        // Straightforward getter
        NetworkType get_type() const override;

        // Straightforward setter
        void set_optimizer(Optimizer* optim);
};

#endif