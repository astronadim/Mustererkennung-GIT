%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Musterkennung Übung 1
% Gruppe 1
% Christian Edelmann 3560916
% Lars Pfeiffer 
% Nadim Maraqten
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; format longG; close all; clc;
run init;
warning('off', 'Images:initSize:adjustingMag');
rng(1);

%% Parameters:
scale = 1/2;
r = 2;   % tile row
c = 13;  % tile column
numSuperpixels = 30;

disp('----------------------------');

%% Load Data:
RGBIR = single(d_RGBIR.loadData(r, c))/255;
RGB = RGBIR(:,:,1:3);
IR = RGBIR(:,:,4);
nDSM = single(d_nDSM.loadData(r, c));
nDSM = (nDSM - min(nDSM(:))) / (max(nDSM(:)) - min(nDSM(:)));
gt = d_GT.loadData(r, c);
gt = uint8(data.potsdam.rgbLabel2classLabel(gt));

% Rescale all input channels
RGB = imresize(RGB, scale, 'method', 'nearest');
nDSM = imresize(nDSM, scale, 'method', 'nearest');
IR = imresize(IR, scale, 'method', 'nearest');
gt = imresize(gt, scale, 'method', 'nearest');

%% SLIC Segmentation
[L, N] = superpixels(RGB, numSuperpixels);

%% Feature Calculation and Label Assignment
feature_vector = zeros(N, 5); % [Mean R, Mean G, Mean B, Mean nDSM, Mean IR]
labels = zeros(N, 1);
for labelVal = 1:N
    mask = L == labelVal;
    segmentPixelsRGB = RGB(repmat(mask, [1 1 3])); % Mask replicated for 3 channels
    numPixels = sum(mask(:)); % Total number of pixels in the segment
    if numPixels > 0
        segmentPixelsRGB = reshape(segmentPixelsRGB, [], 3);
        feature_vector(labelVal, 1:3) = mean(segmentPixelsRGB, 1);
    else
        feature_vector(labelVal, 1:3) = 0; % Default to 0 if no pixels in mask
    end
    feature_vector(labelVal, 4) = mean(nDSM(mask), 'all');
    feature_vector(labelVal, 5) = mean(IR(mask), 'all');
    labels(labelVal) = mode(gt(mask));
end

%% Visualize the results
feature_vector_normalized = feature_vector(:, 1:3) / 255;
feature_vector_normalized = max(0, min(1, feature_vector_normalized));

% Visualize segmentation boundaries
figure;
BW = boundarymask(L);
imshow(imoverlay(RGB, BW, 'cyan'));
title('SLIC Segmentation Boundaries');

%% SLIC Segmentation
[L, N] = superpixels(RGB, numSuperpixels);

% Ensure L is properly initialized and contains superpixel labels
if isempty(L) || any(size(L) == 0)
    error('Segmentation resulted in an empty label matrix. Please check input image and parameters.');
end

% Correctly initialize label_to_index with scalar input for zeros()
maxLabel = max(L(:));  % Ensure scalar maximum label
label_to_index = zeros(maxLabel, 1); % Initialize mapping for labels to indices

%% Map labels to unique index values
unique_labels = unique(L);
label_to_index(unique_labels) = 1:numel(unique_labels);  % Map each unique label to an index


% Display features using the average color per segment
unique_labels = unique(L);
label_to_index = zeros(max(L), 1); % Create mapping for used labels
label_to_index(unique_labels) = 1:numel(unique_labels);

colormap_features = zeros(numel(unique_labels), 3); % Allocate only for used labels
for label = unique_labels' % Use transpose to ensure column vector
    mask = (L == label); % Logical index for current label
    if any(mask(:))
        index = label_to_index(label);
        colormap_features(index, :) = mean(feature_vector_normalized(mask, :), 1);
    end
end

% Display feature image
feature_image = label2rgb(L, colormap_features(label_to_index(L), :), [1 1 1]);
figure;
imshow(feature_image);
title('SLIC Segment Feature Colors');

% Display labeled segments using a predefined colormap
labeled_image = label2rgb(L, jet(numel(unique_labels)), [1 1 1]);
figure;
imshow(labeled_image);
title('SLIC Segment Labels');
