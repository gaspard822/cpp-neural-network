#include <iostream>
#include "mean_squared_error_loss.hpp"

double MeanSquaredError::compute(const MatrixXd& y_true, const MatrixXd& y_pred) const {
    return (y_true - y_pred).array().square().sum() / y_true.rows();
}

MatrixXd MeanSquaredError::derivative(const MatrixXd& y_true, const MatrixXd& y_pred) const {
    // the derivative of (y_true - y_pred)^2 w.r.t. y_pred is
    // -1 * 2(y_true - y_pred) = 2(y_pred - y_true)
    return 2 * (y_pred - y_true) / y_true.rows();
}

string MeanSquaredError::get_loss_name() const {
    return "mse";
}

LossFunctionType MeanSquaredError::get_type() const {
    return LossFunctionType::MSE;
}