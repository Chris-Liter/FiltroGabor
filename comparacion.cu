#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define WIDTH 6000
#define HEIGHT 6000
#define CHANNELS 3

__global__ void compararImagenesRGB(const unsigned char* img1, const unsigned char* img2, int* contador, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalPixels) {
        int baseIdx = idx * CHANNELS;

        // Comparamos canal R, G y B
        bool diferente = false;
        for (int c = 0; c < CHANNELS; ++c) {
            if (img1[baseIdx + c] != img2[baseIdx + c]) {
                diferente = true;
                break;
            }
        }

        if (diferente) {
            atomicAdd(contador, 1);
        }
    }
}

int main() {
    // Cargar imágenes en color
    cv::Mat imagen1 = cv::imread("gabor_cpu_mask_21.jpg", cv::IMREAD_COLOR);
    cv::Mat imagen2 = cv::imread("gabor_cuda_mask_21.jpg", cv::IMREAD_COLOR);

    if (imagen1.empty() || imagen2.empty()) {
        std::cerr << "No se pudieron cargar las imágenes.\n";
        return -1;
    }

    // Redimensionar a 6000x6000 si es necesario
    //cv::resize(imagen1, imagen1, cv::Size(WIDTH, HEIGHT));
    //cv::resize(imagen2, imagen2, cv::Size(WIDTH, HEIGHT));

    int totalPixels = WIDTH * HEIGHT;
    int totalBytes = totalPixels * CHANNELS;

    // Reservar memoria en GPU
    unsigned char* d_img1, * d_img2;
    int* d_contador;

    cudaMalloc(&d_img1, totalBytes);
    cudaMalloc(&d_img2, totalBytes);
    cudaMalloc(&d_contador, sizeof(int));

    // Inicializar contador en 0
    int contador = 0;
    cudaMemcpy(d_contador, &contador, sizeof(int), cudaMemcpyHostToDevice);

    // Copiar imágenes a la GPU
    cudaMemcpy(d_img1, imagen1.data, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, imagen2.data, totalBytes, cudaMemcpyHostToDevice);

    // Configurar kernel
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;

    // Ejecutar kernel
    compararImagenesRGB << <numBlocks, blockSize >> > (d_img1, d_img2, d_contador, totalPixels);
    cudaDeviceSynchronize();

    // Recuperar resultado
    cudaMemcpy(&contador, d_contador, sizeof(int), cudaMemcpyDeviceToHost);

    // Calcular porcentaje de diferencia
    double porcentajeError = (double)contador / totalPixels * 100.0;
    std::cout << "Diferencia de píxeles RGB: " << contador << " (" << porcentajeError << "%)\n";

    // Liberar memoria
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_contador);

    return 0;
}