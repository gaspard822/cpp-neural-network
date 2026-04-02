#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <mlx/mlx.h>
#include <memory>

using namespace Eigen;
using namespace std;


struct TrainableParameter {
    double* value_data = nullptr;
    double* grad_data  = nullptr;
    Index rows = 0;
    Index cols = 0;

    TrainableParameter() = default;

    // Matrix parameter
    TrainableParameter(MatrixXd& value, MatrixXd& grad) : value_data(value.data()), grad_data(grad.data()), rows(value.rows()), cols(value.cols()) {}

    // Vector parameter (treat as N x 1)
    TrainableParameter(VectorXd& value, VectorXd& grad) : value_data(value.data()), grad_data(grad.data()), rows(value.size()), cols(1) {}

    // RowVector parameter (treat as 1 x N)
    TrainableParameter(RowVectorXd& value, RowVectorXd& grad) : value_data(value.data()), grad_data(grad.data()), rows(1), cols(value.size()) {}

    // Views for optimizers as matrices
    Map<MatrixXd> value() const {
        return Map<MatrixXd>(value_data, rows, cols);
    }
    Map<MatrixXd> grad() const {
        return Map<MatrixXd>(grad_data, rows, cols);
    }
};

enum class LayerType {
    FULLY_CONNECTED_LAYER,
    INPUT_LAYER,
    LAYER_NORM,
    FEED_FORWARD,
    MULTI_HEAD_ATTENTION_LAYER,
    LINEAR_LAYER
};

/**
 * Abstract base class for layers in the neural network.
 * Defines the interface for forward and backward passes as well as parameter access.
 */
class Layer {
    protected:
        mlx::core::array output, d_input;
    
    public:
        Layer() : output(mlx::core::zeros({1,1}, mlx::core::float32)),
                  d_input(mlx::core::zeros({1,1}, mlx::core::float32)) {}
        virtual ~Layer() = default;

        /**
         * Does the forward pass using the given input.
         * @param input Input matrix for the layer
         */
        virtual void forward(const mlx::core::array& input) = 0;

        /**
         * Does the backward pass using the given output gradient and saves the gradient with respect to the layer input in d_input.
         * @param d_output Gradient of the loss passed by the following layer
         */
        virtual void backward(const mlx::core::array& d_output) = 0;

        /**
         * Does inference (forward pass) without modifying the internal state of the layer.
         * @param layer_input Input of the layer
         * @return mlx::core::array Output of the layer
         */
        virtual mlx::core::array infer(const mlx::core::array& layer_input) const = 0;

        virtual const vector<TrainableParameter>& get_parameters() const = 0;

        // Straightforward getters
        virtual const mlx::core::array& get_output() const = 0;
        virtual const mlx::core::array& get_d_input() const = 0;

        /**
         * Returns the name of the layer (used in save_model(const string& path) and load_model(const string& path)).
         * @return string Name of the layer
         */
        virtual string get_layer_name() const = 0;

        /**
         * Returns the name of the activation function used by the layer (used in save_model(const string& path) and
         * load_model(const string& path)).
         * @return string Name of the layer's activation function
         */
        virtual string get_activation_name() const = 0;

        /**
         * Returns the type of the layer.
         * @return LayerType Enum value corresponding to the layer type
         */
        virtual LayerType get_type() const = 0;

        /**
         * Saves the layer's weights and hyperparameters to a stream.
         * @param file Stream to which we save the layer
         */
        virtual void save(ofstream& file) const = 0;

        /**
         * Loads the layer's weights and hyperparameters from a stream.
         * @param file Stream from which to load the layer
         */
        virtual void load(ifstream& file) = 0;        
};

#endif
