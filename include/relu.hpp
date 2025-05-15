#ifndef RELU_HPP
#define RELU_HPP

#include "activation_function.hpp"

/**
 * Implements the ReLU (Rectified Linear Unit) activation function.
 */
class Relu : public ActivationFunction {
    public:
        /**
         * Applies the ReLU activation function to the input matrix.
         * @param z The input matrix
         * @return MatrixXd The result of applying ReLU element-wise
         */
        MatrixXd apply(const MatrixXd& z) const override;

        /**
         * Computes the derivative of the ReLU function.
         * @param z The input matrix
         * @return MatrixXd The element-wise derivative of ReLU
         */
        MatrixXd derivative(const MatrixXd& z) const override;

        /**
         * Returns the activation type (ReLU).
         * @return ActivationType Enum value for ReLU
         */
        ActivationType get_type() const override;
};

#endif