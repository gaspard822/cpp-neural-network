#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "activation_function.hpp"

/**
 * Implements the sigmoid activation function.
 */
class Sigmoid : public ActivationFunction {
    public:
        /**
         * Applies the sigmoid activation function to the input matrix.
         * @param z The input matrix
         * @return MatrixXd The result of applying the sigmoid function element-wise
         */
        MatrixXd apply(const MatrixXd& z) const override;

        /**
         * Computes the derivative of the sigmoid function.
         * @param z The input matrix
         * @return MatrixXd The element-wise derivative of the sigmoid function
         */
        MatrixXd derivative(const MatrixXd& z) const override;

        /**
         * Returns the activation type (sigmoid).
         * @return ActivationType Enum value for sigmoid
         */
        ActivationType get_type() const override;
};

#endif