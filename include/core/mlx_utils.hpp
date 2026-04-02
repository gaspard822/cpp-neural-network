#ifndef MLX_UTILS_HPP
#define MLX_UTILS_HPP

#include <fstream>
#include <Eigen/Dense>
#include <mlx/mlx.h>

mlx::core::array eigen_to_mlx(const Eigen::MatrixXd& m);
Eigen::MatrixXd mlx_to_eigen(const mlx::core::array& a);
void save_array(std::ofstream& file, const mlx::core::array& arr);
mlx::core::array load_array(std::ifstream& file);

#endif