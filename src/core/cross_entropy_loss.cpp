#include <iostream>
#include "core/cross_entropy_loss.hpp"

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

// Does the same computation as the other CrossEntropy::compute(), but doesn't need one-hot enoding
double CrossEntropy::compute(const vector<int>& y_true, const MatrixXd& y_pred) const {
    int T = y_true.size();
    double loss = 0.0;

    for (int t = 0; t < T; t++) {
        const RowVectorXd row = y_pred.row(t);

        // log-sum-exp trick
        double max = row.maxCoeff();
        RowVectorXd shifted_logits = row.array() - max;
        double log_sum_exp = log(shifted_logits.array().exp().sum());

        loss += -row(y_true[t]) + max + log_sum_exp;
    }

    return loss / T;
}


MatrixXd CrossEntropy::derivative(const vector<int>& y_true, const MatrixXd& y_pred) const {
    int num_true_tokens = y_true.size();
    int vocab_size = y_pred.cols();

    MatrixXd grad = MatrixXd::Zero(num_true_tokens, vocab_size);

    for (int t = 0; t < num_true_tokens; t++) {
        RowVectorXd row = y_pred.row(t);
        double max = row.maxCoeff();
        RowVectorXd exp_shifted = (row.array() - max).exp();
        double sum_exp = exp_shifted.sum();
        RowVectorXd exp_quotient = exp_shifted / sum_exp;

        grad.row(t) = exp_quotient;
        grad(t, y_true[t]) -= 1.0;
    }

    return grad / num_true_tokens;
}



string CrossEntropy::get_loss_name() const {
    return "cross-entropy";
}

LossFunctionType CrossEntropy::get_type() const {
    return LossFunctionType::CROSSENTROPY;
}