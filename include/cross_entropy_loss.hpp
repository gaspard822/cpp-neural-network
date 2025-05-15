#ifndef CROSS_ENTROPY_LOSS_HPP
#define CROSS_ENTROPY_LOSS_HPP

#include "loss_function.hpp"

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
        double compute(const MatrixXd& y_true, const MatrixXd& y_pred) const override;

        /**
         * Computes the gradient of the cross-entropy loss with respect to the predictions.
         * @param y_true Matrix of true labels
         * @param y_pred Matrix of predicted probabilities
         * @return MatrixXd Gradient of the loss with respect to y_pred
         */
        MatrixXd derivative(const MatrixXd& y_true, const MatrixXd& y_pred) const override;

        /**
         * Returns the name of the loss function ("cross-entropy").
         * @return string Loss function name
         */
        string get_loss_name() const override;

        /**
         * Returns the loss function type (cross-entropy).
         * @return LossFunctionType Enum value for cross-entropy
         */
        LossFunctionType get_type() const override;
};

#endif