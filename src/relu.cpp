#include <iostream>
#include "relu.hpp"

using namespace std;

MatrixXd Relu::apply(const MatrixXd& z) const {
    return z.cwiseMax(0.0);
}

MatrixXd Relu::derivative(const MatrixXd& z) const {
    return (z.array() > 0).cast<double>();
}

ActivationType Relu::get_type() const {
    return ActivationType::RELU;
}