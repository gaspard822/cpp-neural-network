#ifndef MSE_HPP
#define MSE_HPP

#include "core/loss_function.hpp"

/**
 * Implementation of the mean squared error loss function.
 */
class MeanSquaredError : public LossFunction {
    public:
        /**
         * Computes the average mean squared error between predictions and targets.
         * @param y_true Matrix of true values
         * @param y_pred Matrix of predicted values
         * @return float Scalar mean squared error
         */
        float compute(const mlx::core::array& y_true, const mlx::core::array& y_pred) const override;

        /**
         * Computes the gradient of the MSE loss with respect to the predictions.
         * @param y_true Matrix of true values
         * @param y_pred Matrix of predicted values
         * @return mlx::core::array Gradient of the loss with respect to y_pred
         */
        mlx::core::array derivative(const mlx::core::array& y_true, const mlx::core::array& y_pred) const override;

        /**
         * Returns the name of the loss function ("mse").
         * @return string Loss function name
         */
        std::string get_loss_name() const override;

        /**
         * Returns the loss function type (mse).
         * @return LossFunctionType Enum value for MSE
         */
        LossFunctionType get_type() const override;
};

#endif
