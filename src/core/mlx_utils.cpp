#include "core/mlx_utils.hpp"

using namespace std;
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

void save_array(ofstream& file, const mx::array& arr) {
    file << arr.ndim();
    for (int dim : arr.shape()) file << " " << dim;
    file << "\n";
    const float* data = arr.data<float>();
    int size = 1;
    for (int dim : arr.shape()) size *= dim;
    for (int i = 0; i < size; i++) file << data[i] << " ";
    file << "\n";
}

mx::array load_array(ifstream& file) {
    int ndim;
    file >> ndim;
    mx::Shape shape(ndim);
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        file >> shape[i];
        size *= shape[i];
    }
    vector<float> data(size);
    for (int i = 0; i < size; i++) file >> data[i];
    return mx::array(data.begin(), shape, mx::float32);
}
