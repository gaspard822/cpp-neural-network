#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <Eigen/Dense>
#include "layer.hpp"
#include "loss_function.hpp"
#include "optimizer.hpp"

using namespace std;
using namespace Eigen;

/**
 * Represents a feedforward neural network composed of sequential layers.
 */
class NeuralNetwork {
    private:
        vector<Layer*> layers;
        LossFunction* loss_function;
        Optimizer* optimizer;
    
    public:
        /**
         * Constructs an empty neural network.
         */
        NeuralNetwork();

        /**
         * Constructs a network with the given loss function and optimizer.
         * @param loss Pointer to the loss function
         * @param optim Pointer to the optimizer
         */
        NeuralNetwork(LossFunction* loss, Optimizer* optim);

        /**
         * Constructs a network by selecting loss function and optimizer by name.
         * @param loss_function Name of the loss function ("MeanSquaredError" or "CrossEntropy")
         * @param optimizer Name of the optimizer ("VanillaSGD" or "Adam")
         */
        NeuralNetwork(const string& loss_function, const string& optimizer);

        ~NeuralNetwork();

        /**
         * Adds a new layer to the network and updates the optimizer accordingly.
         * @param layer Pointer to the layer to add
         */
        void add_layer(Layer* layer);

        /**
         * Does a full forward pass through the network.
         * @param input Input data matrix (samples x features)
         * @return MatrixXd Output of the final layer
         */
        MatrixXd forward(const MatrixXd& input);

        /**
         * Does a full backward pass using the given target and prediction.
         * @param y_true Ground truth labels
         * @param y_pred Predicted outputs from the forward pass
         */
        void backward(const MatrixXd& y_true, const MatrixXd& y_pred);

        /**
         * Trains the network using the given training data and optional validation data.
         * @param X_train Input training data
         * @param Y_train Target training labels
         * @param epochs Number of training epochs
         * @param batch_size Size of each training batch
         * @param X_val Validation input data (optional)
         * @param Y_val Validation labels (optional)
         * @param early_stopping Whether to stop early based on validation performance
         * @param verbose Whether to print training progress
         */
        void train(const MatrixXd& X_train, const MatrixXd& Y_train, int epochs, int batch_size,
                   const MatrixXd& X_val = MatrixXd(), const MatrixXd& Y_val = MatrixXd(),
                   bool early_stopping = true, bool verbose = false);

        /**
         * Does inference on new data without modifying internal state.
         * @param input Input data matrix (samples x features)
         * @return MatrixXd Network output
         */
        MatrixXd infer(const MatrixXd& input) const;

        /**
         * Saves the model's architecture and parameters to a .txt file.
         * @param path Filesystem path to save the model
         */
        void save_model(const string& path) const;

        /**
         * Loads a model's architecture and parameters from a file.
         * @param path Filesystem path to the saved model
         */
        void load_model(const string& path);

        // Straightforward getter        
        const vector<Layer*>& get_layers() const;

        // Straightforward setter
        void set_optimizer(Optimizer* optim);
};

#endif