#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.
__device__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 2;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src){
    int i = 0;
    while ((dest[i] = src[i]) != '\0') {
        i++;
    }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src){
    while (*dest != '\0') {
        dest++;
    }
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
    size_t input_length = d_strlen((const char *)input);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}

// Function for computing the SHA256 hash for a transation
__global__ void compute_transaction(int transaction_size, BYTE *transactions, BYTE (*hashes)[SHA256_HASH_SIZE], int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        apply_sha256(transactions + i * transaction_size, hashes[i]);
    }
}

__global__ void build_merkle_tree(BYTE (*hashes)[SHA256_HASH_SIZE], int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (2 * i < n) {
        BYTE combined[SHA256_HASH_SIZE * 2];
        if (2 * i + 1 < n) {
            d_strcpy((char *)combined, (const char *)hashes[2 * i]);
            d_strcat((char *)combined, (const char *)hashes[2 * i + 1]);
        } else {
            d_strcpy((char *)combined, (const char *)hashes[2 * i]);
            d_strcat((char *)combined, (const char *)hashes[2 * i]);
        }
        apply_sha256(combined, hashes[i]);
    }
}

// Construction for Merkle tree in CUDA
void construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]){
    BYTE (*hashes)[SHA256_HASH_SIZE] = (BYTE (*)[SHA256_HASH_SIZE])malloc(max_transactions_in_a_block * SHA256_HASH_SIZE);

    if (!hashes) {
        fprintf(stderr, "Error: Unable to allocate memory for hashes\n");
        exit(EXIT_FAILURE);
    }

    // Compute the SHA256 hash for each transaction
    // parallel implementation of compute transaction
    BYTE *device_transactions = 0;
    BYTE (*device_hashes)[SHA256_HASH_SIZE] = 0;

    cudaMalloc((void **)&device_transactions, transaction_size * n);
    cudaMalloc((void **)&device_hashes, max_transactions_in_a_block * SHA256_HASH_SIZE);

    const size_t block_size = 126;
    size_t blocks_no = n / block_size;

    if (n % block_size) 
        ++blocks_no;

    cudaMemcpy(device_transactions, transactions, transaction_size * n, cudaMemcpyHostToDevice);

    compute_transaction<<<blocks_no, block_size>>>(transaction_size, device_transactions, device_hashes, n);
    cudaDeviceSynchronize();

    while (n > 1) {
        blocks_no = n / block_size;
        if (n % block_size) 
            ++blocks_no;
        
        build_merkle_tree<<<blocks_no, block_size>>>(device_hashes, n);
        cudaDeviceSynchronize();
        n = (n + 1) / 2;
    }

    cudaMemcpy(hashes, device_hashes, n * SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);

    memcpy(merkle_root, hashes[0], SHA256_HASH_SIZE);

    free(hashes);

    cudaFree(device_transactions);
    cudaFree(device_hashes);
}


__global__ void calculate_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce, int* found){
    uint32_t idx = threadIdx.x +  blockIdx.x * blockDim.x;
    uint32_t stride = blockDim.x * gridDim.x;

    char local_block_content[BLOCK_SIZE];
    char local_block_hash[SHA256_BLOCK_SIZE];
    char nonce_string[NONCE_SIZE];

    d_strcpy(local_block_content, (char *)block_content);

    for (uint32_t i = idx; i <= max_nonce && *found == 0; i += stride) {
        intToString(i, nonce_string);
        d_strcpy((char *)local_block_content + current_length, nonce_string);
        apply_sha256((BYTE *)local_block_content, (BYTE *)local_block_hash);

        if (compare_hashes((BYTE *)local_block_hash, difficulty) <= 0) {
            if (atomicExch(found, 1) == 0) {
                *valid_nonce = i;
                d_strcpy((char *) block_hash, local_block_hash);
                return;
            }
        }
    }
}

// TODO 2: Implement this function in CUDA
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    int *found = (int *) malloc(sizeof(int));
    *found = 0;

    BYTE *device_difficulty = 0;
    BYTE *device_block_content = 0;
    BYTE *device_block_hash = 0;
    uint32_t *device_valid_nonce = 0;
    int *device_found;

    cudaMalloc((void **)&device_difficulty, SHA256_HASH_SIZE);
    cudaMalloc((void **)&device_block_content, BLOCK_SIZE);
    cudaMalloc((void **)&device_block_hash, SHA256_HASH_SIZE);
    cudaMalloc((void **)&device_valid_nonce, sizeof(uint32_t));
    cudaMalloc((void **)&device_found, sizeof(int));
    cudaMemset(device_found, 0, sizeof(int));

    cudaMemcpy(device_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_block_content, block_content, BLOCK_SIZE, cudaMemcpyHostToDevice);

    const size_t block_size = 64;
    // size_t blocks_no = max_nonce / block_size;

    // if (max_nonce % block_size) 
    //     ++blocks_no;

    calculate_nonce<<<2048, block_size>>>(device_difficulty, max_nonce, device_block_content, current_length, device_block_hash, device_valid_nonce, device_found);
    cudaDeviceSynchronize();

    cudaMemcpy(block_hash, device_block_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(valid_nonce, device_valid_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(found, device_found, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_difficulty);
    cudaFree(device_block_content);
    cudaFree(device_block_hash);
    cudaFree(device_valid_nonce);
    cudaFree(device_found);

    return !(*found);
}

__global__ void dummy_kernel() {}

// Warm-up function
void warm_up_gpu() {
    BYTE *dummy_data;
    cudaMalloc((void **)&dummy_data, 256);
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(dummy_data);
}