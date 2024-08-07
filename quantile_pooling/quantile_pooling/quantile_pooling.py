import torch
import quantile_pooling_cuda
from torch.autograd import Function

class QPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, quant_low=0.9, quant_high=1.0):
        assert 0.0 <= quant_low <= quant_high <= 1.0, "quant_low and quant_high should be between 0 and 1, and quant_low <= quant_high"
        output, indices = quantile_pooling_cuda.pooling(input.contiguous(), quant_low, quant_high)
        quant_param = torch.tensor([quant_low, quant_high], device=input.device)
        ctx.save_for_backward(indices, input, quant_param)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, input, quant_param = ctx.saved_tensors
        quant_low = quant_param[0].item()
        quant_high = quant_param[1].item()
        grad_output = grad_output.contiguous()
        quant_low = max(0.0, min(1.0, quant_low))
        quant_high = max(0.0, min(1.0, quant_high))
        grad_input = quantile_pooling_cuda.pooling_backward(grad_output, indices, input, quant_low, quant_high)
        return grad_input, None, None

def quant_pooling(input, keepdim=False, quant_low=0.9, quant_high=1.0):
    """
    input: (B, C, N)
    output: (B, C) if keepdim is False else (B, C, 1)
    PARAMS:
        quant_low: float, between 0 and 1, higher value speeds up the computation
        quant_high: float, between 0 and 1
    """
    o = QPoolingFunction.apply(input, quant_low, quant_high)
    if keepdim:
        return o
    return o.squeeze(-1)
