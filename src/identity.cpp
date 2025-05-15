#include "identity.hpp"

using namespace std;

MatrixXd Identity::apply(const MatrixXd& z) const {
    return z;
}

MatrixXd Identity::derivative(const MatrixXd& z) const {
    return MatrixXd::Ones(z.rows(), z.cols());
}

ActivationType Identity::get_type() const {
    return ActivationType::IDENTITY;
}