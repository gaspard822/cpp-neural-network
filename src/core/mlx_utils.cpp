#include "core/mlx_utils.hpp"

namespace mx = mlx::core;

mx::array eigen_to_mlx(const Eigen::MatrixXd& m) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rm = m.cast<float>();
    return mx::array(rm.data(), {(int)m.rows(), (int)m.cols()}, mx::float32);
}

Eigen::MatrixXd mlx_to_eigen(const mx::array& a) {
    mx::eval(a);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        map(a.data<float>(), a.shape(0), a.shape(1));
    return map.cast<double>();
}