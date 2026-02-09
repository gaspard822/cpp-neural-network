#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Dense>
#include <core/loss_function.hpp>
#include <core/optimizer.hpp>

using namespace std;

enum class NetworkType {
    MULTI_LAYER_PERCEPTRON,
    TRANSFORMER
}; 

class Network {
    protected:
        LossFunction* loss_function;
        Optimizer* optimizer;

    public:
        Network() = default;
        /**
         * Constructs a network with the loss function and optimizer.
         * @param new_network Pointer to the network to be optimized
         */
        Network(LossFunction* loss, Optimizer* optim);

        // TODO: Need to add forward(), backward(), train(), infer(), however, we call it with matrices for the MLP, but with text for the transformers

        /**
         * Saves the model's architecture and parameters to a .txt file.
         * @param path Filesystem path to save the model
         */
        virtual void save_model(const string& path) const = 0;

        /**
         * Loads a model's architecture and parameters from a file.
         * @param path Filesystem path to the saved model
         */
        virtual void load_model(const string& path) = 0;

        // Straightforward getter
        virtual Optimizer* get_optimizer() const = 0;

        /**
         * Returns the type of the network.
         * @return NetworkType Enum value indicating the network type
         */
        virtual NetworkType get_type() const = 0;
};

#endif