#include <iostream>
#include "core/mlx_utils.hpp"

using namespace std;
namespace mx = mlx::core;

void save_array(ofstream& file, const mx::array& arr) {
    mx::eval(arr);
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
