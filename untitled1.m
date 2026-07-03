%% Image Compression and Denoising via Singular Value Decomposition (SVD)
% Author: Goldfish Prodigy
% Description: This script compresses an image using low-rank matrix 
%              approximations and applies SVD thresholding to eliminate 
%              Gaussian noise.

clear; clc; close all;

%% 1. Load and Preprocess Image
% Load a built-in MATLAB image (or replace with your own 'image.jpg')
imageRGB = imread('peppers.png'); 
imageGray = double(rgb2gray(imageRGB)); % Convert to grayscale and double precision

% Normalize image matrix to range [0, 1]
A = imageGray / 255;
[rows, cols] = size(A);

%% 2. Compute Singular Value Decomposition
[U, S, V] = svd(A);
singularValues = diag(S);

%% 3. Image Compression (Low-Rank Approximation)
% Define the target rank 'k' (number of singular values to retain)
% Feel free to experiment with this value (e.g., 10, 30, 80)
k = 30; 

U_k = U(:, 1:k);
S_k = S(1:k, 1:k);
V_k = V(:, 1:k);

A_compressed = U_k * S_k * V_k';

% Calculate compression ratio
original_elements = rows * cols;
compressed_elements = k * (rows + cols + 1);
compression_percentage = (1 - (compressed_elements / original_elements)) * 100;

%% 4. Image Denoising via SVD Thresholding
% Introduce artificial Gaussian white noise to the original image
noise_intensity = 0.05;
A_noisy = A + noise_intensity * randn(rows, cols);

% Compute SVD of the noisy image
[U_n, S_n, V_n] = svd(A_noisy);

% Apply a hard threshold to eliminate smaller singular values representing noise
% Energy-based thresholding: keep components capturing the core structure
threshold_rank = 45; 
A_denoised = U_n(:, 1:threshold_rank) * S_n(1:threshold_rank, 1:threshold_rank) * V_n(:, 1:threshold_rank)';

%% 5. Performance Evaluation (PSNR Metrics)
% Peak Signal-to-Noise Ratio measures image reconstruction quality
psnr_compressed = psnr(A_compressed, A);
psnr_denoised = psnr(A_denoised, A);
psnr_noisy = psnr(A_noisy, A);

%% 6. Visualization Matrix
figure('Name', 'SVD Image Processing Pipeline', 'Position', [100, 100, 1000, 700]);

subplot(2, 2, 1);
imshow(A);
title('Original Matrix (Full Rank)');

subplot(2, 2, 2);
imshow(A_compressed);
title(sprintf('Compressed (Rank %d)\nStorage Saved: %.1f%% \nPSNR: %.2f dB', k, compression_percentage, psnr_compressed));

subplot(2, 2, 3);
imshow(A_noisy);
title(sprintf('Noisy Image\nPSNR: %.2f dB', psnr_noisy));

subplot(2, 2, 4);
imshow(A_denoised);
title(sprintf('Denoised (Rank %d Subspace)\nPSNR: %.2f dB', threshold_rank, psnr_denoised));

% Print summary metrics to console
fprintf('=== SVD Pipeline Metrics ===\n');
fprintf('Original Image Dimensions: %d x %d\n', rows, cols);
fprintf('Compressed Storage Reduction: %.2f%%\n', compression_percentage);
fprintf('Compressed Image Quality (PSNR): %.2f dB\n', psnr_compressed);
fprintf('Denoised Image Quality (PSNR): %.2f dB\n', psnr_denoised);