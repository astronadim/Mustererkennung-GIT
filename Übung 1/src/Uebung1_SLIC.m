%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Musterkennung Übung 1
% Gruppe 1
% Christian Edelmann 3560916
% Lars Pfeiffer      3514519
% Nadim Maraqten     3384833
% Johannes Bladt     3541171
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; format longG; close all; clc;
run init;
%#ok<*NOPTS>
warning('off', 'Images:initSize:adjustingMag');
rng(1);

%% Parameters:
% Input:
scale  = 1/2;
r      = 3;   % tile row
c      = 14;  % tile column
k_vector=[250, 500, 1000, 2000, 4000];

compactness=1;
method='slic0';
disp('----------------------------');

% Preallocation for performance
accuracies = zeros(length(k_vector), 1);
times = zeros(length(k_vector), 1);

%% Load Data:
% Load data by row / column number:

RGBIR = single(d_RGBIR.loadData(r, c))/255;
RGB   = RGBIR(:,:,1:3);
IR    = RGBIR(:,:,4);

nDSM  = single(d_nDSM.loadData(r, c));
nDSM  = (nDSM - min(nDSM(:))) / (max(nDSM(:)) - min(nDSM(:)));

gt    = d_GT.loadData(r, c);
gt    = uint8(data.potsdam.rgbLabel2classLabel(gt));

RGB   = imresize(RGB,  scale,  'method', 'nearest');
nDSM  = imresize(nDSM, scale,  'method', 'nearest');
IR    = imresize(IR,   scale,  'method', 'nearest');
gt    = imresize(gt,   scale,  'method', 'nearest');

figure;
imshow(RGB);
title(sprintf('Input image RGB, scale = %f', scale));

figure;
imshow(IR);
title(sprintf('Input image IR, scale = %f', scale));

figure;
imshow(nDSM, []);
title(sprintf('Input image nDSM, scale = %f', scale));

figure;
imshow(gt, getColorMap('V2DLabels'));
title(sprintf('Input labels, scale = %f', scale));

RGB_R=reshape(RGB(:,:,1),[],1);
RGB_G=reshape(RGB(:,:,2),[],1);
RGB_B=reshape(RGB(:,:,3),[],1);
IR_=reshape(IR,[],1);
nDSM_=reshape(nDSM,[],1);

input_image=RGB;  % SLIC uses the RGB image

%% SLIC
for idx_current = 1:length(k_vector)
    k = k_vector(idx_current)
    tic;  % Start timing
    
    [idx, ~] = superpixels(input_image, k, 'Compactness', compactness, 'Method', method);
    
    figure;
    imshow(idx,[])
    title(['SLIC segmentation mask with k=' ,num2str(k), ', compactness= ',num2str(compactness), ' and method =', method]);
   
    figure;
    bmask = boundarymask(idx);
    imshow(imoverlay(RGB, bmask, 'cyan'));
    title(['SLIC segments on RGB image with k=' ,num2str(k), ', compactness= ',num2str(compactness), ' and method =', method]);
    
    idx = reshape(idx, [], 1);
    gt_ = reshape(gt, [], 1);
    
    feature_R = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_G = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_B = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_nDSM = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_IR = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    
    label_image = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);

    list_labels=zeros(max(idx),1);
    list_features=zeros(max(idx),5);

    for i = 1:max(idx)
        segment_indices = find(idx == i);
        
        segment_gt_label = mode(gt_(segment_indices));
        label_image(segment_indices) = segment_gt_label;
        list_labels(i)=segment_gt_label;
        
        % Compute mean values for each feature using the segment indices
        feature_R(segment_indices) = mean(RGB_R(segment_indices));
        feature_G(segment_indices) = mean(RGB_G(segment_indices));
        feature_B(segment_indices) = mean(RGB_B(segment_indices));
        feature_nDSM(segment_indices) = mean(nDSM_(segment_indices));
        feature_IR(segment_indices) = mean(IR_(segment_indices));

        list_features(i,:)=[mean(RGB_R(segment_indices));
                          mean(RGB_G(segment_indices));
                          mean(RGB_B(segment_indices));
                          mean(nDSM_(segment_indices));
                          mean(IR_(segment_indices));];
    end
    
    label_image = reshape(label_image, [length(RGB(:,:,1)), length(RGB(:,:,1))]);
    feature_R = reshape(feature_R, [length(RGB(:,:,1)), length(RGB(:,:,1))]);
    feature_G = reshape(feature_G, [length(RGB(:,:,1)), length(RGB(:,:,1))]);
    feature_B = reshape(feature_B, [length(RGB(:,:,1)), length(RGB(:,:,1))]);
    feature_RGB = cat(3, feature_R, feature_G, feature_B);
    
    times(idx_current) = toc  % End timing
    
    figure;
    imshow(feature_RGB, []);
    title(['SLIC RGB features with k=' ,num2str(k), ', compactness= ', num2str(compactness), ' and method =', method]);
    
    figure;
    imshow(label_image, getColorMap('V2DLabels'));
    title(['SLIC labeled image with k=' ,num2str(k), ', compactness= ', num2str(compactness), ' and method =', method]);
    
    % Calculate accuracy
    correctly_identified_labels = label_image == reshape(gt, [length(RGB(:,:,1)), length(RGB(:,:,1))]);
    accuracies(idx_current) = sum(correctly_identified_labels, 'all') / (length(RGB(:,:,1)) * length(RGB(:,:,1)))
end

%% Plot accuracies and times
f = figure;
yyaxis left;
plot(k_vector, accuracies * 100, '-o');
ylabel('Accuracy [%]');
ylim([0 100]);

yyaxis right;
plot(k_vector, times, '-o');
ylabel('Time (s)');
ylim([0 max(times)]);

xlabel('Cluster numbers k');
title(['Accuracy and Runtime for Various Cluster Numbers k with ', num2str(compactness), ' compactness and method', method]);
legend('Accuracy', 'Runtime', 'Location', 'best');
% saveas(f, ['results_slic/accuracy_runtime_' datestr(now, 'yyyy-mm-dd_HHMMSS') '.png']);
