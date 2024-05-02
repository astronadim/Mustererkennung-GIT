%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Musterkennung Übung 1
% Gruppe 1
% Christian Edelmann 3560916
% Lars Pfeiffer 
% Nadim Maraqten
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; format longG; close all; clc;
run init;
%#ok<*NOPTS>
warning('off', 'Images:initSize:adjustingMag');
rng(1);



%% Parameters:
% Input:
scale  = 1/2
r      = 2   % tile row
c      = 13  % tile column
k=5000
compactness=1
method='slic0'

disp('----------------------------')



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
title(sprintf('Input image RGB, scale = %f',scale));
% 
% figure;
% imshow(IR);
% title(sprintf('Input image IR, scale = %f',scale));
% 
% figure;
% imshow(nDSM, []);
% title(sprintf('Input image nDSM, scale = %f',scale));
% 
% figure;
% imshow(gt, getColorMap('V2DLabels'));
% title(sprintf('Input labels, scale = %f',scale));
% 
RGB_R=reshape(RGB(:,:,1),[],1);
RGB_G=reshape(RGB(:,:,2),[],1);
RGB_B=reshape(RGB(:,:,3),[],1);
IR_=reshape(IR,[],1);
nDSM_=reshape(nDSM,[],1);


input_image=RGB;



%% SLIC

[idx,~]=superpixels(input_image,k,'Compactness',compactness,'Method',method);

figure
imshow(idx,[])
title(['SLIC segmentation mask with number of clusters k=' ,num2str(k), ' , compactness= ',num2str(compactness), ' and method =',method])

figure
bmask = boundarymask(idx);
imshow(imoverlay(RGB,bmask,'cyan'))
title(['SLIC segments on RGB image with number of clusters k=' ,num2str(k), ' , compactness= ',num2str(compactness), ' and method =',method])

idx=reshape(idx,[],1);

feature_R=ones(3000*3000,1);
feature_G=ones(3000*3000,1);
feature_B=ones(3000*3000,1);
feature_nDSM=ones(3000*3000,1);
feature_IR=ones(3000*3000,1);

gt_=reshape(gt,[],1);
label_image=idx;

% for i=1:k
% length_index=length(find(idx==i));
% index(1:length_index)=find(idx==i);
% 
% segment_gt_label=mode(gt_(index));
% label_image(label_image==i)=segment_gt_label;
% 
% feature_R(index)=mean(RGB_R(index));
% feature_G(index)=mean(RGB_G(index));
% feature_B(index)=mean(RGB_B(index));
% feature_nDSM(index)=mean(nDSM(index));
% feature_IR(index)=mean(IR(index));
% end

for i = 1:max(idx)
    fprintf('%d\n', i)
    segment_indices = find(idx == i);  % Only find indices once to save computation

    segment_gt_label = mode(gt_(segment_indices));
    label_image(idx == i) = segment_gt_label;  % Assign labels based on segment index

    % Compute mean values for each feature using the segment indices
    feature_R(segment_indices) = mean(RGB_R(segment_indices));
    feature_G(segment_indices) = mean(RGB_G(segment_indices));
    feature_B(segment_indices) = mean(RGB_B(segment_indices));
    feature_nDSM(segment_indices) = mean(nDSM_(segment_indices));
    feature_IR(segment_indices) = mean(IR_(segment_indices));
end

label_image=reshape(label_image,[3000,3000]);

imp_surf = label_image == 1;
building = label_image == 2;
low_veg  = label_image == 3;
tree     = label_image == 4;
car      = label_image == 5;
clutter  = label_image == 6;
    
r = imp_surf | car | clutter;
g = imp_surf | low_veg | tree | car;
b = imp_surf | building | low_veg;
    
RGB_label_image = cat(3,r,g,b) * 255;

feature_R=reshape(feature_R,[3000,3000]);
feature_G=reshape(feature_G,[3000,3000]);
feature_B=reshape(feature_B,[3000,3000]);

feature_RGB=ones(3000,3000,3);
feature_RGB(:,:,1)=feature_R;
feature_RGB(:,:,2)=feature_G;
feature_RGB(:,:,3)=feature_B;

figure
imshow(feature_RGB,[])
title(['SLIC RGB features with number of clusters k=' ,num2str(k), ' , compactness= ',num2str(compactness), ' and method =',method])

figure
imshow(RGB_label_image,[])
title(['SLIC labeled image with number of clusters k=' ,num2str(k), ' , compactness= ',num2str(compactness), ' and method =',method])



