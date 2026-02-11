#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
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
        MatrixXd d_input, output;
    
    public:
        Layer() = default;
        virtual ~Layer() = default;

        /**
         * Does the forward pass using the given input.
         * @param input Input matrix for the layer
         */
        virtual void forward(const MatrixXd& input) = 0;

        /**
         * Does the backward pass using the given output gradient and saves the gradient with respect to the layer input in d_input.
         * @param d_output Gradient of the loss passed by the following layer
         */
        virtual void backward(const MatrixXd& d_output) = 0;

        /**
         * Does inference (forward pass) without modifying the internal state of the layer.
         * @param layer_input Input of the layer
         * @return MatrixXd Output of the layer
         */
        virtual MatrixXd infer(const MatrixXd& layer_input) const = 0;

        virtual const vector<TrainableParameter>& get_parameters() const = 0;

        // Straightforward getters
        virtual const MatrixXd& get_output() const = 0;
        virtual const MatrixXd& get_d_input() const = 0;

        /**
         * Returns the name of the activation function used by the layer (used in
         * NeuralNetwork::save_model(const string& path) and NeuralNetwork::load_model(const string& path)).
         * @return string Name of the loss function
         */
        virtual string get_activation_name() const = 0;

        /**
         * Returns the type of the layer.
         * @return LayerType Enum value corresponding to the layer type
         */
        virtual LayerType get_type() const = 0;
};

#endif