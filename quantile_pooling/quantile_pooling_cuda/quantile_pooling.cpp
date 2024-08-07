#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void pooling_cuda(const float* input, float* output, int* indices, int total_elements, int last_dim, int D, float quant_low, float quant_high);
void pooling_backward_cuda(const float* grad_output, const int* indices, float* grad_input, int total_elements, int last_dim, int D, float quant_low, float quant_high);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> pooling(torch::Tensor input, float quant_low, float quant_high) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2-dimensional");
    TORCH_CHECK(input.size(-2) >= 2, "Input must have at least 2 elements in the second last dimension");
    TORCH_CHECK(input.size(-1) >= 2, "Input must have at least 2 elements in the last dimension");

    auto output_shape = input.sizes().vec();
    output_shape.back() = 1;
    auto output = torch::empty(output_shape, input.options());
    auto indices_shape = input.sizes().vec();
    indices_shape.back() = 2;
    auto indices = torch::empty(indices_shape, input.options().dtype(torch::kInt32));

    int total_elements = input.numel() / input.size(-1);
    int last_dim = input.size(-1);
    int D = input.size(-2);

    pooling_cuda(input.data_ptr<float>(), output.data_ptr<float>(), indices.data_ptr<int>(), \
        total_elements, last_dim, D, quant_low, quant_high);

    return {output, indices};
}

torch::Tensor pooling_backward(torch::Tensor grad_output, torch::Tensor indices, torch::Tensor input,
    float quant_low, float quant_high) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(indices);
    CHECK_INPUT(input);

    auto grad_input = torch::zeros_like(input);

    int total_elements = input.numel() / input.size(-1);
    int last_dim = input.size(-1);
    int D = input.size(-2);

    pooling_backward_cuda(grad_output.data_ptr<float>(), indices.data_ptr<int>(), grad_input.data_ptr<float>(), \
        total_elements, last_dim, D, quant_low, quant_high);

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pooling", &pooling, "Pooling (CUDA)");
    m.def("pooling_backward", &pooling_backward, "Pooling Backward (CUDA)");
}