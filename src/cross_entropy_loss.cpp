#include <iostream>
#include "cross_entropy_loss.hpp"

// y_true is the one-hot encoding and y_pred are the logits
double CrossEntropy::compute(const MatrixXd& y_true, const MatrixXd& y_pred) const {
    // Shift logits for numerical stability
    MatrixXd z_max = y_pred.rowwise().maxCoeff();
    MatrixXd shifted_logits = y_pred - z_max.replicate(1, y_pred.cols());

    // Compute log-sum-exp
    MatrixXd exp_shifted = shifted_logits.array().exp();
    VectorXd log_sum_exp = exp_shifted.rowwise().sum().array().log();

    // Compute loss: -z_y + z_max + log(sum(exp))
    // Multiply element-wise: only the correct class contributes (y_true is one-hot)
    VectorXd true_logits = (y_true.array() * y_pred.array()).rowwise().sum();
    VectorXd loss_vector = -true_logits.array() + z_max.array() + log_sum_exp.array();

    // Average over batch
    return loss_vector.mean();
}

// y_true is the one-hot encoding and y_pred are the logits
MatrixXd CrossEntropy::derivative(const MatrixXd& y_true, const MatrixXd& y_pred) const {
    // Shift logits for numerical stability
    MatrixXd z_max = y_pred.rowwise().maxCoeff();
    MatrixXd shifted_logits = y_pred - z_max.replicate(1, y_pred.cols());

    // Compute the exponents of the shifted logits and sum them over each sample
    MatrixXd exp_shifted = shifted_logits.array().exp();
    VectorXd sum_exp = exp_shifted.rowwise().sum();

    // Compute the quotient of the shifted exponents divided by the sum over each sample
    MatrixXd exp_quotient = exp_shifted.array().colwise() / sum_exp.array();

    // Return this quotient minus the true labels divided by the number of samples
    return (exp_quotient - y_true) / y_true.rows();
}

string CrossEntropy::get_loss_name() const {
    return "cross-entropy";
}

LossFunctionType CrossEntropy::get_type() const {
    return LossFunctionType::CROSSENTROPY;
}