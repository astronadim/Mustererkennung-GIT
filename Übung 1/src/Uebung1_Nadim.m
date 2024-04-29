%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Musterkennung Übung 1
% Gruppe 1
% Optimized Version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc; %#ok<CLALL>
format longG;
run init;
warning('off', 'Images:initSize:adjustingMag');
rng(1);

%% Parameters:
scale = 0.5;  % Reduction in image size for faster processing
r = 2;        % Tile row
c = 13;       % Tile column
k_means_max_iter = 20;
k = 40;
disp('----------------------------');

%% Load Data and Process:
RGBIR = single(d_RGBIR.loadData(r, c)) / 255;
RGB = RGBIR(:,:,1:3);
IR = RGBIR(:,:,4);

nDSM = single(d_nDSM.loadData(r, c));
nDSM = (nDSM - min(nDSM(:))) / (max(nDSM(:)) - min(nDSM(:)));

gt = d_GT.loadData(r, c);
gt = uint8(data.potsdam.rgbLabel2classLabel(gt));

% Image resizing
RGB = imresize(RGB, scale, 'method', 'nearest');
nDSM = imresize(nDSM, scale, 'method', 'nearest');
IR = imresize(IR, scale, 'method', 'nearest');
gt = imresize(gt, scale, 'method', 'nearest');

% Display RGB image
figure;
imshow(RGB);
title(sprintf('Input image RGB, scale = %f', scale));

%% kmeans Clustering
% Reshape image matrices to vectors and prepare for kmeans
RGB_R = reshape(RGB(:,:,1), [], 1);
RGB_G = reshape(RGB(:,:,2), [], 1);
RGB_B = reshape(RGB(:,:,3), [], 1);
IR_ = reshape(IR, [], 1);
nDSM_ = reshape(nDSM, [], 1);

input_image = [RGB_R, RGB_G, RGB_B, IR_, nDSM_];

% kmeans computation
[idx, C] = kmeans(input_image, k, 'MaxIter', k_means_max_iter, 'Start', 'plus');

% Reshape index matrix to the original image size
mask = reshape(idx, size(RGB, 1), size(RGB, 2));

% Display the kmeans segmentation mask
figure;
imshow(label2rgb(mask), []);
title(['kmeans Segmentation Mask with k = ', num2str(k), ' (No Boundary)']);

% Calculate and display boundary mask
bmask = boundarymask(mask);

% Display RGB image with boundary mask
figure;
imshow(imoverlay(RGB, bmask, 'cyan'));
title(['kmeans Boundary Mask with k = ', num2str(k)]);



%% Feature Extraction and Labeling
% Preallocate feature arrays
feature_R = zeros(numel(RGB_R), 1);
feature_G = zeros(numel(RGB_G), 1);
feature_B = zeros(numel(RGB_B), 1);

% Efficient computation of features using vectorization
for i = 1:k
    idx_current = (idx == i);
    feature_R(idx_current) = mean(RGB_R(idx_current));
    feature_G(idx_current) = mean(RGB_G(idx_current));
    feature_B(idx_current) = mean(RGB_B(idx_current));
end

% Construct feature image
feature_RGB = zeros(size(RGB));
feature_RGB(:,:,1) = reshape(feature_R, size(RGB, 1), size(RGB, 2));
feature_RGB(:,:,2) = reshape(feature_G, size(RGB, 1), size(RGB, 2));
feature_RGB(:,:,3) = reshape(feature_B, size(RGB, 1), size(RGB, 2));

% Display the feature image
figure;
imshow(uint8(feature_RGB));
title('kmeans segmented image with RGB features');
