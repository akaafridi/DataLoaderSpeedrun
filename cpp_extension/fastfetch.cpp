#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

/*
 * Zero-copy tensor fetching function.
 * This function fetches a tensor from a batch without making a copy.
 * It uses PyTorch's advanced indexing and avoids unnecessary copies.
 */
torch::Tensor fetch_tensor_zero_copy(torch::Tensor tensor_batch, int64_t index) {
    // Simple error check
    if (index < 0 || index >= tensor_batch.size(0)) {
        throw std::runtime_error("Index out of bounds");
    }
    
    // Return a view using select operation (no copy)
    return tensor_batch.select(0, index);
}

PYBIND11_MODULE(fastfetch, m) {
    m.doc() = "pybind11 extension for zero-copy tensor fetching";
    
    m.def("fetch_tensor_zero_copy", &fetch_tensor_zero_copy, 
          "Fetch a tensor from a batch without making a copy",
          py::arg("tensor_batch"), py::arg("index"));
}
