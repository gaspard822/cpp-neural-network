#ifndef IDENTITY_HPP
#define IDENTITY_HPP

#include "activation_function.hpp"

/**
 * Implements the identity activation function.
 * While the identity if not an activation function, it is convenient to represent it as such for implementation.
 */
class Identity : public ActivationFunction {
    public:
        /**
         * Applies the identity activation function to the input matrix.
         * @param z The input matrix
         * @return MatrixXd A copy of the input matrix
         */
        MatrixXd apply(const MatrixXd& z) const override;

        /**
         * Computes the derivative of the identity function.
         * @param z The input matrix
         * @return MatrixXd A matrix of ones
         */
        MatrixXd derivative(const MatrixXd& z) const override;

        /**
         * Returns the activation type (identity).
         * @return ActivationType Enum value for identity
         */
        ActivationType get_type() const override;
};

#endif