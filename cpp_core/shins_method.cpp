#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

std::vector<double> calculate_shins_probs(const std::vector<double> &inputs) {
  if (inputs.empty()) {
    throw std::invalid_argument("Input list cannot be empty.");
  }

  size_t n = inputs.size();
  std::vector<double> pi(n);

  // Heuristic: Check if inputs are decimal odds or implied probabilities
  bool is_decimal_odds = false;
  for (double val : inputs) {
    if (val <= 0.0) {
      throw std::invalid_argument(
          "Odds or probabilities must be strictly positive.");
    }
    if (val > 1.0) {
      is_decimal_odds = true;
    }
  }

  if (is_decimal_odds) {
    for (size_t i = 0; i < n; ++i) {
      pi[i] = 1.0 / inputs[i];
    }
  } else {
    pi = inputs; // copy
  }

  double s = std::accumulate(pi.begin(), pi.end(), 0.0);

  // Safeguard if sum is less than or equal to 1.0 (no vig / negative margin)
  if (s <= 1.0) {
    std::vector<double> fallback(n);
    if (s > 0.0) {
      for (size_t i = 0; i < n; ++i) {
        fallback[i] = pi[i] / s;
      }
    } else {
      // uniform fallback
      double val = 1.0 / static_cast<double>(n);
      std::fill(fallback.begin(), fallback.end(), val);
    }
    return fallback;
  }

  // Bisection method solver for z in [0, 1)
  double low = 0.0;
  double high = 1.0 - 1e-9;
  double z = 0.0;

  const int max_iterations = 100;

  for (int iter = 0; iter < max_iterations; ++iter) {
    double mid = (low + high) / 2.0;

    double p_sum = 0.0;
    double one_minus_mid = 1.0 - mid;
    double double_one_minus_mid = 2.0 * one_minus_mid;
    double mid_sq = mid * mid;

    for (size_t i = 0; i < n; ++i) {
      double val = mid_sq + 4.0 * one_minus_mid * (pi[i] * pi[i]) / s;
      double p_i = (std::sqrt(std::max(0.0, val)) - mid) / double_one_minus_mid;
      p_sum += p_i;
    }

    if (std::abs(p_sum - 1.0) < 1e-12) {
      z = mid;
      break;
    } else if (p_sum > 1.0) {
      low = mid;
    } else {
      // p_sum < 1.0
      high = mid;
    }
  }
  z = (low + high) / 2.0;

  // Compute final true probabilities with converged z
  std::vector<double> true_probs(n);
  double one_minus_z = 1.0 - z;
  double double_one_minus_z = 2.0 * one_minus_z;
  double z_sq = z * z;
  double total = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double val = z_sq + 4.0 * one_minus_z * (pi[i] * pi[i]) / s;
    true_probs[i] = (std::sqrt(std::max(0.0, val)) - z) / double_one_minus_z;
    total += true_probs[i];
  }

  // Normalize strictly to sum to 1.0
  if (total > 0.0) {
    for (size_t i = 0; i < n; ++i) {
      true_probs[i] /= total;
    }
  } else {
    double val = 1.0 / static_cast<double>(n);
    std::fill(true_probs.begin(), true_probs.end(), val);
  }

  return true_probs;
}
