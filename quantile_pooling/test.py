import torch
from quantile_pooling.quantile_pooling import quant_pooling
from torch.autograd import Function
import time
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(0)

def test_q_pooling():
    test_cases = [
        torch.randn((3, 6), device='cuda', dtype=torch.float32, requires_grad=True),       # 2D tensor
        torch.randn((512, 1024), device='cuda', dtype=torch.float32, requires_grad=True),    # 3D tensor
        torch.randn((128, 512, 128), device='cuda', dtype=torch.float32, requires_grad=True), # 3D tensor
        torch.randn((1, 128, 512, 1024), device='cuda', dtype=torch.float32, requires_grad=True), # 4D tensor
        torch.randn((1, 2, 3, 4, 6), device='cuda', dtype=torch.float32, requires_grad=True), # 5D tensor
        torch.randn((8, 512, 10000), device='cuda', dtype=torch.float32, requires_grad=True), 
        torch.randn((8, 64, 99999), device='cuda', dtype=torch.float32, requires_grad=True), 
    ]
    
    num_runs = 10  # Number of runs to average the time
    
    for i, input_tensor in enumerate(test_cases):
        print(f"Test Case {i+1}: Input Tensor Shape: {input_tensor.shape}")

        total_time = 0.0
        for _ in range(num_runs):
            # zero the gradients
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()

            start_time = time.time()
            output_tensor = quant_pooling(input_tensor, quant_low=0.95, quant_high=1)
            loss = output_tensor.sum()
            # loss.backward()
            end_time = time.time()
            total_time += (end_time - start_time)

            print('>>> test iteration:', _)
        
        avg_time = total_time / num_runs
        
        # Print the input, output tensors, and gradient of the input tensor
        print("Input Tensor:")
        print(input_tensor.shape)
        
        print("Output Tensor:")
        print(output_tensor, output_tensor.shape)
        
        # print("Gradient of Input Tensor:")
        # print(input_tensor.grad)
        
        print(f"Average Time for {num_runs} runs: {avg_time:.6f} seconds")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    test_q_pooling()