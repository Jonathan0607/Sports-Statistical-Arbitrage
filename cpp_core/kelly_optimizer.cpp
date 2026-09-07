#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

/**
 * Multivariate Kelly Portfolio Optimizer using Eigen.
 *
 * Computes optimal fractional Kelly allocations across simultaneous correlated bets.
 * Formula: f = Σ⁻¹ * G
 *
 * Uses Eigen's ColPivHouseholderQR decomposition for robust handling of
 * near-singular or ill-conditioned covariance matrices.
 *
 * @param edge_vector     Vector of expected value edges (G)
 * @param covariance_matrix  Flattened row-major covariance matrix (Σ), dimension n×n
 * @param max_limit       Maximum total portfolio allocation (e.g. 0.05 for 5%)
 * @param kelly_fraction  Fractional Kelly scaling (e.g. 0.25 for quarter-Kelly)
 * @return                Optimal allocation vector as std::vector<double>
 */
std::vector<double> optimize_multivariate_kelly(
    const std::vector<double>& edge_vector,
    const std::vector<std::vector<double>>& covariance_matrix,
    double max_limit,
    double kelly_fraction
) {
    size_t n = edge_vector.size();

    if (n == 0) {
        return {};
    }

    // Validate covariance matrix dimensions
    if (covariance_matrix.size() != n) {
        throw std::invalid_argument("Covariance matrix row count must match edge vector length.");
    }
    for (size_t i = 0; i < n; ++i) {
        if (covariance_matrix[i].size() != n) {
            throw std::invalid_argument("Covariance matrix must be square (n x n).");
        }
    }

    // Map inputs to Eigen types
    Eigen::VectorXd G(n);
    Eigen::MatrixXd Sigma(n, n);

    for (size_t i = 0; i < n; ++i) {
        G(i) = edge_vector[i];
        for (size_t j = 0; j < n; ++j) {
            Sigma(i, j) = covariance_matrix[i][j];
        }
    }

    // Solve Σ * f = G using ColPivHouseholderQR for robust near-singular handling
    Eigen::VectorXd f = Sigma.colPivHouseholderQr().solve(G);

    // Long-only clipping: negative allocations → 0
    for (Eigen::Index i = 0; i < f.size(); ++i) {
        if (f(i) < 0.0) {
            f(i) = 0.0;
        }
    }

    // Scale allocations so their sum does not exceed max_limit
    double total = f.sum();
    if (total > max_limit && total > 1e-12) {
        f *= (max_limit / total);
    }

    // Apply fractional Kelly scaling
    f *= kelly_fraction;

    // Convert back to std::vector
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = f(i);
    }

    return result;
}
