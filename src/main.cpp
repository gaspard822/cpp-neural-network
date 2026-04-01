#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <mlx/mlx.h>
#include <Eigen/Dense>
#include "mlp/mnist.cpp"
#include "transformer/translation.cpp"

using namespace std;
namespace mx = mlx::core;


int test_mlx() {
    const int N = 8192;

    // Eigen benchmark
    Eigen::MatrixXf eA = Eigen::MatrixXf::Random(N, N);
    Eigen::MatrixXf eB = Eigen::MatrixXf::Random(N, N);

    auto t0 = chrono::high_resolution_clock::now();
    Eigen::MatrixXf eC = eA * eB;
    auto t1 = chrono::high_resolution_clock::now();
    double eigen_ms = chrono::duration<double, milli>(t1 - t0).count();

    // MLX benchmark
    mx::array mA = mx::random::uniform({N, N});
    mx::array mB = mx::random::uniform({N, N});
    mx::eval(mA, mB);

    auto t2 = chrono::high_resolution_clock::now();
    mx::array mC = mx::matmul(mA, mB);
    mx::eval(mC);
    auto t3 = chrono::high_resolution_clock::now();
    double mlx_ms = chrono::duration<double, milli>(t3 - t2).count();

    cout << "Device: " << mx::default_device() << "\n";
    cout << "Matrix size: " << N << "x" << N << "\n";
    cout << "Eigen time: " << eigen_ms << " ms\n";
    cout << "MLX   time: " << mlx_ms   << " ms\n";
    cout << "Speedup (Eigen/MLX): " << eigen_ms / mlx_ms << "x\n";
    return 0;
}

int main() {

    // train_test_mnist();
    // infer_mnist();

    TrainingConfig cfg;
    // init_transformer_model(cfg);
    train_transformer_model(cfg, false, false);

    // test_mlx();

    return 0;
}
