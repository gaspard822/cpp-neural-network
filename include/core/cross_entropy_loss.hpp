#ifndef CROSS_ENTROPY_LOSS_HPP
#define CROSS_ENTROPY_LOSS_HPP

#include "core/loss_function.hpp"

/**
 * Implementation of the cross-entropy loss function.
 */
class CrossEntropy : public LossFunction {
    public:
        /**
         * Computes the average cross-entropy loss over all samples.
         * @param y_true Matrix of true one-hot encoded labels
         * @param y_pred Matrix of predicted probabilities
         * @return double Scalar cross-entropy loss
         */
        double compute(const mlx::core::array& y_true, const mlx::core::array& y_pred) const override;

        /**
         * Computes the average cross-entropy loss over all samples.
         * @param y_true Vector of the true labels
         * @param y_pred Matrix of predicted probabilities
         * @return double Scalar cross-entropy loss
         */
        double compute(const std::vector<int>& y_true, const mlx::core::array& y_pred) const;

        /**
         * Computes the gradient of the cross-entropy loss with respect to the predictions.
         * @param y_true Matrix of one-hot encoded true labels
         * @param y_pred Matrix of predicted probabilities
         * @return mlx::core::array Gradient of the loss with respect to y_pred
         */
        mlx::core::array derivative(const mlx::core::array& y_true, const mlx::core::array& y_pred) const override;

        /**
         * Computes the gradient of the cross-entropy loss with respect to the predictions.
         * @param y_true Vector of the true labels
         * @param y_pred Matrix of predicted probabilities
         * @return mlx::core::array Gradient of the loss with respect to y_pred
         */
        mlx::core::array derivative(const std::vector<int>& y_true, const mlx::core::array& y_pred) const;

        /**
         * Returns the name of the loss function ("cross-entropy").
         * @return string Loss function name
         */
        std::string get_loss_name() const override;

        /**
         * Returns the loss function type (cross-entropy).
         * @return LossFunctionType Enum value for cross-entropy
         */
        LossFunctionType get_type() const override;
};

#endif
