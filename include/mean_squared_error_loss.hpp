#ifndef MSE_HPP
#define MSE_HPP

#include "loss_function.hpp"

/**
 * Implementation of the mean squared error loss function.
 */
class MeanSquaredError : public LossFunction {
    public:
        /**
         * Computes the average mean squared error between predictions and targets.
         * @param y_true Matrix of true values
         * @param y_pred Matrix of predicted values
         * @return double Scalar mean squared error
         */
        double compute(const MatrixXd& y_true, const MatrixXd& y_pred) const override;

        /**
         * Computes the gradient of the MSE loss with respect to the predictions.
         * @param y_true Matrix of true values
         * @param y_pred Matrix of predicted values
         * @return MatrixXd Gradient of the loss with respect to y_pred
         */
        MatrixXd derivative(const MatrixXd& y_true, const MatrixXd& y_pred) const override;

        /**
         * Returns the name of the loss function ("mse").
         * @return string Loss function name
         */
        string get_loss_name() const override;

        /**
         * Returns the loss function type (mse).
         * @return LossFunctionType Enum value for MSE
         */
        LossFunctionType get_type() const override;
};

#endif