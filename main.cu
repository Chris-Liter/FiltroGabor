#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <book.h>

using namespace std;
using namespace cv;

vector<float> generateGaborKernel(int ksize, float sigma, float theta, 
    float lambda, float psi, float gamma) {

    vector<float> kernel(ksize * ksize);
    int half = ksize / 2;

    float cosTheta = cos(theta);
    float sinTheta = sin(theta);

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float xTheta = x * cosTheta + y * sinTheta;
            float yTheta = -x * sinTheta + y * cosTheta;
            float gauss = exp(-(xTheta * xTheta + gamma * gamma * yTheta * yTheta) / (2 * sigma * sigma));
            float wave = cos(2 * CV_PI * xTheta / lambda + psi);
            kernel[(y + half) * ksize + (x + half)] = gauss * wave;
        }
    }

    return kernel;
}

Mat applyGaborCPU(const Mat& img, const vector<float>& kernel, int ksize) {
    int half = ksize / 2;
    Mat output = Mat::zeros(img.size(), img.type());

    for (int y = half; y < img.rows - half; y++) {
        for (int x = half; x < img.cols - half; x++) {
            Vec3f sum = {0, 0, 0};
            for (int i = -half; i <= half; i++) {
                for (int j = -half; j <= half; j++) {
                    Vec3b pixel = img.at<Vec3b>(y + i, x + j);
                    float weight = kernel[(i + half) * ksize + (j + half)];
                    sum[0] += weight * pixel[0];
                    sum[1] += weight * pixel[1];
                    sum[2] += weight * pixel[2];
                }
            }
            Vec3b& dstPixel = output.at<Vec3b>(y, x);
            for (int c = 0; c < 3; c++)
                dstPixel[c] = static_cast<uchar>(min(max(int(sum[c]), 0), 255));
        }
    }

    return output;
}

__global__ void applyGaborCUDA(const uchar3* input, uchar3* output, 
    float* kernel, int ksize, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = ksize / 2;

    if (x >= half && y >= half && x < width - half && y < height - half) {
        float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;

        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int imgX = x + kx;
                int imgY = y + ky;
                int idx = imgY * width + imgX;
                int kidx = (ky + half) * ksize + (kx + half);

                uchar3 pixel = input[idx];
                float weight = kernel[kidx];

                sumB += weight * pixel.x;
                sumG += weight * pixel.y;
                sumR += weight * pixel.z;
            }
        }

        int outIdx = y * width + x;
        output[outIdx].x = min(max(int(sumB), 0), 255);
        output[outIdx].y = min(max(int(sumG), 0), 255);
        output[outIdx].z = min(max(int(sumR), 0), 255);
    }
}

int main() {
    Mat img = imread("imagenArbol.jpg", IMREAD_COLOR);
    if (img.empty()) {
        cerr << "No se pudo cargar la imagen" << endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int imgSize = width * height;
    int ksize = 9;

    // Crear kernel
    auto kernel = generateGaborKernel(ksize, 5.0, CV_PI / 4.0, 10.0, 2.0, 0.5);

    //-------------------- CPU --------------------//
    auto startCPU = chrono::high_resolution_clock::now();
    Mat cpuResult = applyGaborCPU(img, kernel, ksize);
    auto endCPU = chrono::high_resolution_clock::now();
    auto durationCPU = chrono::duration_cast<chrono::milliseconds>(endCPU - startCPU);

    cout << "Tiempo CPU: " << durationCPU.count() << "ms" << endl;
    imwrite("gabor_cpu.jpg", cpuResult);

    //-------------------- CUDA --------------------//
    uchar3 *d_input, *d_output;
    float* d_kernel;

    cudaMalloc(&d_input, sizeof(uchar3) * imgSize);
    cudaMalloc(&d_output, sizeof(uchar3) * imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * ksize * ksize);

    cudaMemcpy(d_input, img.ptr<uchar3>(), sizeof(uchar3) * imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), sizeof(float) * ksize * ksize, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    applyGaborCUDA<<<grid, block>>>(d_input, d_output, d_kernel, ksize, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    cout << "Tiempo GPU: " << gpuTime << "ms" << endl;

    Mat gpuResult(height, width, CV_8UC3);
    cudaMemcpy(gpuResult.ptr<uchar3>(), d_output, sizeof(uchar3) * imgSize, cudaMemcpyDeviceToHost);
    imwrite("gabor_cuda.jpg", gpuResult);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}