#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

enum class LossFunctionType {
    MSE,
    CROSSENTROPY
};

/**
 * This class defines the interface for loss functions used during model training and evaluation..
 */
class LossFunction {
public:
    LossFunction() = default;
    virtual ~LossFunction() = default;

    /**
     * Computes the loss value between predictions and true labels.
     * @param y_true Matrix of true labels (one-hot or continuous) (samples x features)
     * @param y_pred Matrix of predicted outputs (samples x features)
     * @return double Scalar loss value
     */
    virtual double compute(const MatrixXd& y_true, const MatrixXd& y_pred) const = 0;

    /**
     * Computes the derivative of the loss with respect to the predictions.
     * @param y_true Matrix of true labels (samples x features)
     * @param y_pred Matrix of predicted outputs (samples x features)
     * @return MatrixXd Gradient of the loss with respect to y_pred (samples x features)
     */
    virtual MatrixXd derivative(const MatrixXd& y_true, const MatrixXd& y_pred) const = 0;

    /**
     * Returns the name of the loss function (used in NeuralNetwork::save_model(const string& path) and
     * NeuralNetwork::load_model(const string& path)).
     * @return string Name of the loss function
     */
    virtual string get_loss_name() const = 0;

    /**
     * Returns the type of the loss function.
     * @return LossFunctionType Enum value corresponding to the loss function type
     */
    virtual LossFunctionType get_type() const = 0;
};

#endif