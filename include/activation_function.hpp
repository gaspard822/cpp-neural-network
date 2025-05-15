#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <Eigen/Dense>

using namespace Eigen;

enum class ActivationType {
    RELU,
    SIGMOID,
    IDENTITY
};

/**
 * Abstract base class defining the interface for all activation functions used in the network.
 */
class ActivationFunction {
    public:
        virtual ~ActivationFunction() = default;

        /**
         * Applies the activation function to the input matrix.
         * @param z The input matrix (samples x features)
         * @return MatrixXd The result after applying the activation function
         */
        virtual MatrixXd apply(const MatrixXd& z) const = 0;

        /**
         * Computes the derivative of the activation function.
         * @param z The input matrix (samples x features)
         * @return MatrixXd The element-wise derivative
         */
        virtual MatrixXd derivative(const MatrixXd& z) const = 0;

        /**
         * Returns the type of the activation function.
         * @return ActivationType Enum value corresponding to the function type
         */
        virtual ActivationType get_type() const = 0;
};

#endif