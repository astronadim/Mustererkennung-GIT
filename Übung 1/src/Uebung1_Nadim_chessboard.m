clear; close all; clc;
format longG;
run init; % Assuming 'init' script setups the environment correctly
warning('off', 'Images:initSize:adjustingMag');
rng(1);

%% Parameters
scale = 0.5;  % Reduction in image size
segment_size = [10, 10]; % Segment size for chessboard segmentation, adjustable
r = 2;        % Tile row
c = 13;       % Tile column

disp('----------------------------');

%% Load and Scale Data
RGBIR = single(d_RGBIR.loadData(r, c)) / 255;
RGB = imresize(RGBIR(:,:,1:3), scale, 'nearest');
IR = imresize(RGBIR(:,:,4), scale, 'nearest');
nDSM = imresize(single(d_nDSM.loadData(r, c)), scale, 'nearest');
nDSM = (nDSM - min(nDSM(:))) / (max(nDSM(:)) - min(nDSM(:)));
gt = imresize(uint8(data.potsdam.rgbLabel2classLabel(d_GT.loadData(r, c))), scale, 'nearest');

% Display initial RGB image
figure;
imshow(RGB);
title(sprintf('Scaled Input RGB image, scale = %f', scale));

%% Chessboard Segmentation
num_segments_x = ceil(size(RGB,1) / segment_size(1));
num_segments_y = ceil(size(RGB,2) / segment_size(2));
segmentation_mask = zeros(size(RGB,1), size(RGB,2));

% Prepare lists for feature vectors and labels
feature_vectors = [];
labels = [];

for ix = 1:num_segments_x
    for iy = 1:num_segments_y
        x_start = (ix-1)*segment_size(1) + 1;
        y_start = (iy-1)*segment_size(2) + 1;
        x_end = min(ix*segment_size(1), size(RGB,1));
        y_end = min(iy*segment_size(2), size(RGB,2));

        segment_RGB = RGB(x_start:x_end, y_start:y_end, :);
        segment_gt = gt(x_start:x_end, y_start:y_end);

        avg_R = mean(segment_RGB(:,:,1), 'all');
        avg_G = mean(segment_RGB(:,:,2), 'all');
        avg_B = mean(segment_RGB(:,:,3), 'all');
        feature_vector = [avg_R, avg_G, avg_B];
        feature_vectors = [feature_vectors; feature_vector];
        
        segment_label = mode(segment_gt(:));
        labels = [labels; segment_label];
        
        segmentation_mask(x_start:x_end, y_start:y_end) = segment_label;
    end
end

% Display the segmentation mask
figure;
imshow(label2rgb(segmentation_mask));
title('Chessboard Segmentation Mask');

% Optionally, overlay the segment borders on the RGB image
edges = edge(segmentation_mask, 'Sobel'); % Detect edges
se = strel('disk', 1);  % Create a structuring element for dilation
edges_dilated = imdilate(edges, se);  % Dilate edges

figure;
imshow(imoverlay(RGB, edges_dilated, 'w'));
title('Enhanced Segment Borders on RGB Image');

%% Output results
disp('Feature vectors per segment:');
disp(feature_vectors);
disp('Labels per segment:');
disp(labels);
