#ifndef MLX_UTILS_HPP
#define MLX_UTILS_HPP

#include <Eigen/Dense>
#include <mlx/mlx.h>

mlx::core::array eigen_to_mlx(const Eigen::MatrixXd& m);
Eigen::MatrixXd mlx_to_eigen(const mlx::core::array& a);

#endif