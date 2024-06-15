%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Musterkennung Übung 1 - mit saving routine
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
tic;  % Start timing




%% Parameters:
% Input:
scale  = 1/2;
row = [2,3];   % tile row for training
column = [13,14];  % tile column for training

segments_sizes_vector = 15;
list_labels_tmp=zeros(40000,2);
list_features_tmp=zeros(40000,5,2);

%% Load Training Data:
% Load data by row / column number:

for k=1:length(row)

RGBIR = single(d_RGBIR.loadData(row(k), column(k)))/255;
RGB   = RGBIR(:,:,1:3);
IR    = RGBIR(:,:,4);

nDSM  = single(d_nDSM.loadData(row(k), column(k)));
nDSM  = (nDSM - min(nDSM(:))) / (max(nDSM(:)) - min(nDSM(:)));

gt    = d_GT.loadData(row(k), column(k));
gt    = uint8(data.potsdam.rgbLabel2classLabel(gt));

RGB   = imresize(RGB,  scale,  'method', 'nearest');
nDSM  = imresize(nDSM, scale,  'method', 'nearest');
IR    = imresize(IR,   scale,  'method', 'nearest');
gt    = imresize(gt,   scale,  'method', 'nearest');

% figure;
% imshow(RGB);
% title(sprintf('Input image RGB, scale = %f',scale));
% 
% 
% figure;
% imshow(IR);
% title(sprintf('Input image IR, scale = %f',scale));
% 
% figure;
% imshow(nDSM, []);
% title(sprintf('Input image nDSM, scale = %f',scale));
% 
figure;
imshow(gt, getColorMap('V2DLabels'));
title(sprintf('Ground truth image, scale = %f',scale));

RGB_R=RGB(:,:,1);
RGB_G=RGB(:,:,2);
RGB_B=RGB(:,:,3);

label_image=zeros(length(RGB(:,:,1)));

feature_R=ones(length(RGB(:,:,1)));
feature_G=ones(length(RGB(:,:,1)));
feature_B=ones(length(RGB(:,:,1)));
feature_nDSM=ones(length(RGB(:,:,1)));
feature_IR=ones(length(RGB(:,:,1)));
   
    
    segment_size = segments_sizes_vector;
    segments = length(RGB(:,:,1)) / segment_size;
    idx = zeros(length(RGB(:,:,1)));  % Pre-allocate idx for the maximum required size
    list_labels=zeros(segments);
    list_features=zeros(segments,segments,5);

    for i=1:segments
        for j=1:segments

            x_start = (i-1)*segment_size + 1;
            x_end = i * segment_size;
            y_start = (j-1)*segment_size + 1;
            y_end = j * segment_size;
            idx(x_start:x_end, y_start:y_end) = i*j;  % Assign segment ID


            % Directly use ground truth data without storing indices
            segment_gt_label = mode(gt(x_start:x_end, y_start:y_end), 'all');
            label_image(x_start:x_end, y_start:y_end) = segment_gt_label;

            list_labels(i,j)=segment_gt_label;

            % Calculate and store features
            feature_R(x_start:x_end, y_start:y_end) = mean(RGB_R(x_start:x_end, y_start:y_end), 'all');
            feature_G(x_start:x_end, y_start:y_end) = mean(RGB_G(x_start:x_end, y_start:y_end), 'all');
            feature_B(x_start:x_end, y_start:y_end) = mean(RGB_B(x_start:x_end, y_start:y_end), 'all');
            feature_nDSM(x_start:x_end, y_start:y_end) = mean(nDSM(x_start:x_end, y_start:y_end), 'all');
            feature_IR(x_start:x_end, y_start:y_end) = mean(IR(x_start:x_end, y_start:y_end), 'all');
            
            list_features(i,j,:)=[mean(RGB_R(x_start:x_end, y_start:y_end), 'all');
                                  mean(RGB_G(x_start:x_end, y_start:y_end), 'all');
                                  mean(RGB_B(x_start:x_end, y_start:y_end), 'all');  
                                  mean(nDSM(x_start:x_end, y_start:y_end), 'all');
                                  mean(IR(x_start:x_end, y_start:y_end), 'all');];
            
        end
    end
    
    list_labels_tmp(:,k)=reshape(list_labels,[i*j,1]);
    list_features_tmp(:,:,k)=reshape(list_features,[i*j,5]);
    

    idx=reshape(idx,[],1);

    mask=reshape(idx,length(RGB(:,:,1)),length(RGB(:,:,1)));

    % figure
    % imshow(mask,[])
    % title(['chessboard segmentation mask with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

    % figure
    % bmask = boundarymask(mask);
    % imshow(imoverlay(RGB,bmask,'cyan'))
    % title(['chessboard segments on RGB image with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

    imp_surf = label_image == 1;
    building = label_image == 2;
    low_veg  = label_image == 3;
    tree     = label_image == 4;
    car      = label_image == 5;
    clutter  = label_image == 6;

    r = imp_surf | car | clutter;
    g = imp_surf | low_veg | tree | car;
    b = imp_surf | building | low_veg;

    RGB_label_image(:,:,:,k) = cat(3,r,g,b) * 255;

    feature_RGB=ones(length(RGB(:,:,1)),length(RGB(:,:,1)),3);
    feature_RGB(:,:,1)=feature_R;
    feature_RGB(:,:,2)=feature_G;
    feature_RGB(:,:,3)=feature_B;
    
end 
    
    
    %% plot images
    % figure
    % imshow(feature_RGB)
    % title(['chessboard RGB features with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

    RGB_label_image_train=RGB_label_image(:,:,:,1);
    RGB_label_image_test=RGB_label_image(:,:,:,2);

    list_labels_train=list_labels_tmp(:,1);
    list_labels_test=list_labels_tmp(:,2);

    list_features_train=list_features_tmp(:,:,1);
    list_features_test=list_features_tmp(:,:,2);

    figure
    imshow(RGB_label_image_train)
    title(['chessboard labeled training image with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

    figure
    imshow(RGB_label_image_test)
    title(['chessboard labeled test image with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

% Save intermediate results to a .mat file
save('intermediate_results_ue1.mat', 'gt', 'segment_size', 'segments', 'RGB', 'scale', 'row', 'column', 'segments_sizes_vector', 'list_labels_tmp', 'list_features_tmp', 'RGB_label_image_train', 'RGB_label_image_test', 'list_labels_train', 'list_labels_test', 'list_features_train', 'list_features_test');

