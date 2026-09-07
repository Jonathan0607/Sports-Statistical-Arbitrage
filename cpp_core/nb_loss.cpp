#include <vector>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Computes the Negative Binomial Gradient and Hessian for XGBoost/LightGBM
// Uses the log-link function (predictions are log(mu)).
// Returns a tuple of (gradients, hessians)
std::pair<py::array_t<double>, py::array_t<double>> calculate_nb_gradient_hessian(
    py::array_t<double> preds,
    py::array_t<double> labels,
    double alpha
) {
    py::buffer_info preds_info = preds.request();
    py::buffer_info labels_info = labels.request();

    if (preds_info.ndim != 1 || labels_info.ndim != 1) {
        throw std::runtime_error("Predictions and labels must be 1-dimensional arrays");
    }
    if (preds_info.size != labels_info.size) {
        throw std::runtime_error("Predictions and labels must be the same size");
    }
    if (alpha <= 0) {
        throw std::invalid_argument("Dispersion parameter alpha must be strictly positive");
    }

    auto result_grad = py::array_t<double>(preds_info.size);
    auto result_hess = py::array_t<double>(preds_info.size);

    py::buffer_info grad_info = result_grad.request();
    py::buffer_info hess_info = result_hess.request();

    double* ptr_preds = static_cast<double*>(preds_info.ptr);
    double* ptr_labels = static_cast<double*>(labels_info.ptr);
    double* ptr_grad = static_cast<double*>(grad_info.ptr);
    double* ptr_hess = static_cast<double*>(hess_info.ptr);

    for (size_t i = 0; i < preds_info.size; ++i) {
        double pred = ptr_preds[i];
        double y = ptr_labels[i];
        
        // Predictions from trees are log(mu) (log-link function)
        // Bound predictions to prevent overflow
        if (pred > 20.0) pred = 20.0;
        if (pred < -20.0) pred = -20.0;
        
        double mu = std::exp(pred);

        // g_i = (mu_i - y_i) / (1 + alpha * mu_i)
        double denom = 1.0 + alpha * mu;
        double g = (mu - y) / denom;

        // h_i = (mu_i * (1 + alpha * y_i)) / ((1 + alpha * mu_i)^2)
        double h = (mu * (1.0 + alpha * y)) / (denom * denom);

        // Ensure hessian is strictly positive to prevent XGBoost failure
        if (h < 1e-16) {
            h = 1e-16;
        }

        ptr_grad[i] = g;
        ptr_hess[i] = h;
    }

    return std::make_pair(result_grad, result_hess);
}
