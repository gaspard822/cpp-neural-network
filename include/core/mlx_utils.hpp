#ifndef MLX_UTILS_HPP
#define MLX_UTILS_HPP

#include <fstream>
#include <mlx/mlx.h>

void save_array(std::ofstream& file, const mlx::core::array& arr);
mlx::core::array load_array(std::ifstream& file);

#endif