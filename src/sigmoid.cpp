#include <iostream>
#include "sigmoid.hpp"

using namespace std;

MatrixXd Sigmoid::apply(const MatrixXd& z) const {
    return 1.0 / (1.0 + (-z.array()).exp());
}

MatrixXd Sigmoid::derivative(const MatrixXd& z) const {
    MatrixXd sigmoid = apply(z);
    return sigmoid.array() * (1.0 - sigmoid.array());
}

ActivationType Sigmoid::get_type() const {
    return ActivationType::SIGMOID;
}