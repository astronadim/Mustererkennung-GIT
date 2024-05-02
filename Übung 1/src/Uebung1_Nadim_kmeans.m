%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Musterkennung Übung 1
% Gruppe 1
% Optimized Version with Custom Color Codes for Segmentation Mask
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
k_means_max_iter = 40;
k = 6;
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
RGBIR = imresize(RGBIR, scale, 'method', 'nearest');
nDSM = imresize(nDSM, scale, 'method', 'nearest');
IR = imresize(IR, scale, 'method', 'nearest');
gt = imresize(gt, scale, 'method', 'nearest');

% Display RGB image
figure;
imshow(RGB);
title(sprintf('Input image RGB, scale = %f', scale));

%% kmeans Clustering
% Reshape image matrices to vectors and prepare for kmeans
RGB_vec = reshape(RGB, [], 3);  % Reshape RGB into an N by 3 matrix
IR_vec = reshape(IR, [], 1);    % Reshape IR into an N by 1 vector
nDSM_vec = reshape(nDSM, [], 1); % Reshape nDSM into an N by 1 vector

input_image = [RGB_vec, IR_vec, nDSM_vec];  % Concatenate horizontally

% kmeans computation
[idx, C] = kmeans(input_image, k, 'MaxIter', k_means_max_iter);

% Reshape index matrix to the original image size
mask=reshape(idx,3000,3000);

% Custom color coding for visualization
imp_surf = mask == 1; % Adjust the label number as per actual clustering results
building = mask == 2;
low_veg = mask == 3;
tree = mask == 4;
car = mask == 5;
clutter = mask == 6;

r = imp_surf | car | clutter;
g = imp_surf | low_veg | tree | car;
b = imp_surf | building | low_veg;

RGB_label_image = cat(3, r, g, b) * 255;

% Display the custom kmeans segmentation mask
figure;
imshow(RGB_label_image, []);
title('Custom kmeans Segmentation Mask with Color Coding');

% Calculate and display boundary mask
bmask = boundarymask(mask);

% Display RGB image with boundary mask
figure;
imshow(imoverlay(RGB, bmask, 'cyan'));
title('kmeans Boundary Mask with Color Coding');
