#include <iostream>
#include "core/mean_squared_error_loss.hpp"

using namespace std;
namespace mx = mlx::core;

double MeanSquaredError::compute(const mx::array& y_true, const mx::array& y_pred) const {
    return mx::sum(mx::square(y_true - y_pred)).item<double>() / y_true.shape(0);
}

mx::array MeanSquaredError::derivative(const mx::array& y_true, const mx::array& y_pred) const {
    return 2 * (y_pred - y_true) / y_true.shape(0);
}

string MeanSquaredError::get_loss_name() const {
    return "mse";
}

LossFunctionType MeanSquaredError::get_type() const {
    return LossFunctionType::MSE;
}