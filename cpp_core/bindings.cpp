#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declaration of the C++ solver functions
std::vector<double> calculate_shins_probs(const std::vector<double>& inputs);

std::vector<double> optimize_multivariate_kelly(
    const std::vector<double>& edge_vector,
    const std::vector<std::vector<double>>& covariance_matrix,
    double max_limit,
    double kelly_fraction
);

std::pair<py::array_t<double>, py::array_t<double>> calculate_nb_gradient_hessian(
    py::array_t<double> preds,
    py::array_t<double> labels,
    double alpha
);

PYBIND11_MODULE(quant_core, m) {
    m.doc() = "High-performance C++ quantitative core library for Sports Statistical Arbitrage";
    
    m.def("calculate_shins_probs", &calculate_shins_probs, 
          py::arg("odds_list"),
          "Calculates true de-vigged probabilities using Shin's method (bisection solver)");

    m.def("optimize_multivariate_kelly", &optimize_multivariate_kelly,
          py::arg("edge_vector"),
          py::arg("covariance_matrix"),
          py::arg("max_limit"),
          py::arg("kelly_fraction"),
          "Computes optimal Multivariate Kelly portfolio allocations using Eigen (Sigma^-1 * G)");

    m.def("calculate_nb_gradient_hessian", &calculate_nb_gradient_hessian,
          py::arg("preds"),
          py::arg("labels"),
          py::arg("alpha"),
          "Calculates Negative Binomial gradient and hessian for XGBoost custom objective");
}
