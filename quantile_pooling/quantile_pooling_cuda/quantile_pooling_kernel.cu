#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>
#include <cfloat>

#define BRUTE_FORCE_THRESHOLD 5
#define STACK_HEAP_THRESHOLD 512

// Min-heapify function
__device__ void min_heapify(float* heap, int* idx, int i, int heap_size) {
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < heap_size && heap[left] < heap[smallest]) {
        smallest = left;
    }
    if (right < heap_size && heap[right] < heap[smallest]) {
        smallest = right;
    }
    if (smallest != i) {
        float temp_val = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = temp_val;

        int temp_idx = idx[i];
        idx[i] = idx[smallest];
        idx[smallest] = temp_idx;

        min_heapify(heap, idx, smallest, heap_size);
    }
}

// Build min-heap function
__device__ void build_min_heap(float* heap, int* idx, int heap_size) {
    for (int i = heap_size / 2 - 1; i >= 0; --i) {
        min_heapify(heap, idx, i, heap_size);
    }
}

__device__ void brute_force_k_largest(float* values, int* idx, int k, int last_dim) {
    // the k-th largest element is at the k-th position in the sorted array
    for (int i = 0; i < k; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < last_dim; ++j) {
            if (values[j] > values[max_idx]) {
                max_idx = j;
            }
        }
        float temp_val = values[i];
        values[i] = values[max_idx];
        values[max_idx] = temp_val;

        int temp_idx = idx[i];
        idx[i] = idx[max_idx];
        idx[max_idx] = temp_idx;
    }
}

__device__ void selection_sort(float* values, int* idx, int start, int size) {
    // sort in ascending order
    for (int i = start; i < size - 1; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < size; ++j) {
            if (values[j] < values[min_idx]) {
                min_idx = j;
            }
        }
        // Swap the found minimum element with the first element
        if (min_idx != i) {
            float temp_val = values[i];
            values[i] = values[min_idx];
            values[min_idx] = temp_val;

            int temp_idx = idx[i];
            idx[i] = idx[min_idx];
            idx[min_idx] = temp_idx;
        }
    }
}

__device__ void brute_force_k_largest(float* values, int* idx, const float* input, int k, int last_dim){
    // put the first k elements into the heap
    for (int i = 0; i < k; ++i) {
        values[i] = input[i];
        idx[i] = i;
    }
    // sort the first k elements
    selection_sort(values, idx, 0, k);

    // iterate through the rest of the elements
    for (int i = k; i < last_dim; ++i) {
        float tmp_val = input[i];
        if (tmp_val > values[0]) {
            // the elements are already sorted except the first element
            // 1. find the suitable position for the new element, 2. insert the new element, 3. shift the elements, 4. insert the new element
            int j = 1;
            while (j < k && values[j] < tmp_val) {
                values[j - 1] = values[j];
                idx[j - 1] = idx[j];
                ++j;
            }
            values[j - 1] = tmp_val;
            idx[j - 1] = i;
        }
    }
}

__global__ void pooling_kernel(const float* input, float* output, int* indices, \
    float* shared_values, int* shared_idx, int total_elements, int last_dim, int D, \
    float quant_low, float quant_high
    ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_elements) {
        int base_index = index * last_dim;

        // Determine the rank of the D dimension
        int rank = index % D;
        float quantile = quant_low + (quant_high - quant_low) * (float)(rank) / (float)(D - 1);

        // Determine left_neighbor and k, and then build min-heap of size k using the first k elements
        float q_index_f = quantile * (last_dim - 1);
        int left_neighbor = (int)q_index_f; // index of the left neighbor
        // int right_neighbor = left_neighbor == last_dim - 1 ? left_neighbor : left_neighbor + 1;
        // Interpolation
        float dist_to_left = q_index_f - left_neighbor;
        float dist_to_right = 1.0f - dist_to_left;

        // k is the number of elements greater than the left neighbor plus 1 for the left neighbor
        // This is also the size of the min-heap
        int k = last_dim - left_neighbor;

        float left_value, right_value;
        int ori_index_left, ori_index_right;

        float* values;
        int* idx;
        if (k <= STACK_HEAP_THRESHOLD) {
            // Use stack memory
            float local_values[STACK_HEAP_THRESHOLD];
            int local_idx[STACK_HEAP_THRESHOLD];
            values = local_values;
            idx = local_idx;
        } else {
            // Use shared memory
            values = &shared_values[index * k];
            idx = &shared_idx[index * k];
        }

        if (k < BRUTE_FORCE_THRESHOLD ) {
            // the pointer to the corresponding input
            const float* input_ptr = (float*)input + base_index;
            brute_force_k_largest(values, idx, input_ptr, k, last_dim);
            left_value = values[0];
            ori_index_left = idx[0];
            if (k == 1) {
                right_value = left_value;
                ori_index_right = ori_index_left;
            } else {
                right_value = values[1];
                ori_index_right = idx[1];
            }
        } else {
            // load the first k elements into the heap
            for (int j = 0; j < k; ++j) {
                values[j] = input[base_index + j];
                idx[j] = j;
            }
            build_min_heap(values, idx, k);

            // Compare the rest of the elements with the heap root
            float heap_root = values[0];
            for (int j = k; j < last_dim; ++j) {
                if (input[base_index + j] > heap_root) {
                    values[0] = input[base_index + j];
                    idx[0] = j;
                    min_heapify(values, idx, 0, k);
                    heap_root = values[0];
                }
            }
            // Calculate the quantile value
            left_value = values[0]; // the smallest value in the heap
            ori_index_left = idx[0];
            // replace the smallest value with a random number in the heap
            values[0] = values[1];
            idx[0] = idx[1];
            min_heapify(values, idx, 0, k); // re-heapify the heap
            right_value = values[0]; // the second smallest value in the heap
            ori_index_right = idx[0];
        }

        float quantile_value = left_value * dist_to_right + right_value * dist_to_left;

        output[index] = quantile_value;

        indices[index * 2] = ori_index_left;
        indices[index * 2 + 1] = ori_index_right;
    }
}

__global__ void pooling_backward_kernel(const float* grad_output, const int* indices, float* grad_input, \
int total_elements, int last_dim, int D, float quant_low, float quant_high) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_elements) {
        int base_index = index * last_dim;
        float grad = grad_output[index];

        // Initialize gradient input to zero
        for (int j = 0; j < last_dim; ++j) {
            grad_input[base_index + j] = 0.0f;
        }

        // Set the gradient to the quantile value
        int idx_left = indices[index * 2];
        int idx_right = indices[index * 2 + 1];

        int rank = index % D;
        // float quantile = (float)(rank) / (float)(D - 1);
        float quantile = quant_low + (quant_high - quant_low) * (float)(rank) / (float)(D - 1);

        // Determine left_neighbor and k, and then build min-heap of size k using the first k elements
        float q_index_f = quantile * (last_dim - 1);
        int left_neighbor = (int)q_index_f; // index of the left neighbor
        // Interpolation
        float dist_to_left = q_index_f - left_neighbor;
        float dist_to_right = 1.0f - dist_to_left;

        float grad_right = grad * dist_to_left; // The closer to the left, the less the gradient is distributed to the right
        float grad_left = grad * dist_to_right;

        grad_input[base_index + idx_left] = grad_left;
        if (idx_left != idx_right)
            grad_input[base_index + idx_right] = grad_right;
    }
}

void pooling_cuda(const float* input, float* output, int* indices, int total_elements, int last_dim, int D,
    float quant_low, float quant_high) {
    int threads_per_block = 1024;
    int number_of_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Pre-allocate memory for shared values and indices
    float* d_shared_values;
    int* d_shared_idx;

    // Determine the max heap size, allocate memory only up to necessary size for the heap
    float q_index_f = quant_low * (last_dim - 1);
    int left_neighbor = (int)q_index_f; // index of the left neighbor
    int max_heap_size = last_dim - left_neighbor;

    if (max_heap_size <= STACK_HEAP_THRESHOLD) {
        d_shared_values = NULL;
        d_shared_idx = NULL;
        pooling_kernel<<<number_of_blocks, threads_per_block>>>(input, output, indices, \
            d_shared_values, d_shared_idx, total_elements, last_dim, D,
        quant_low, quant_high);
    }
    else {

        size_t shared_size = total_elements * max_heap_size * sizeof(float);
        size_t idx_size = total_elements * max_heap_size * sizeof(int);
        cudaMalloc(&d_shared_values, shared_size);
        cudaMalloc(&d_shared_idx, idx_size);

        pooling_kernel<<<number_of_blocks, threads_per_block>>>(input, output, indices, \
            d_shared_values, d_shared_idx, total_elements, last_dim, D,
        quant_low, quant_high);

        // Free allocated memory
        cudaFree(d_shared_values);
        cudaFree(d_shared_idx);
    }
}

void pooling_backward_cuda(const float* grad_output, const int* indices, float* grad_input, int total_elements, \
int last_dim, int D, float quant_low, float quant_high) {
    int threads_per_block = 1024;
    int number_of_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    pooling_backward_kernel<<<number_of_blocks, threads_per_block>>>(grad_output, indices, grad_input, \
        total_elements, last_dim, D, quant_low, quant_high);
}