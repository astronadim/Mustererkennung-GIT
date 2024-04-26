%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Musterkennung Übung 1
% Gruppe 1
% Christian Edelmann 3560916
% Lars Pfeiffer 
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
k_means_max_iter=20;
k=40;
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

%% Chessboard





%% kmeans

RGB_R=reshape(RGB(:,:,1),[],1);
RGB_G=reshape(RGB(:,:,2),[],1);
RGB_B=reshape(RGB(:,:,3),[],1);
IR_=reshape(IR,[],1);
nDSM_=reshape(nDSM,[],1);
input_image=[RGB_R,RGB_G,RGB_B,IR_,nDSM_];

idx=kmeans(input_image,k,'MaxIter',k_means_max_iter);
mask=reshape(idx,3000,3000);

figure
imshow(mask,[])
title(['kmeans segmentation mask with k=' ,num2str(k), ' and max number of iterations= ',num2str(k_means_max_iter)])

figure
bmask = boundarymask(mask);
imshow(imoverlay(RGB,bmask,'cyan'))
title(['kmeans boundary mask with k=' ,num2str(k), ' and max number of iterations= ',num2str(k_means_max_iter)])

feature_R=ones(3000*3000,1);
feature_G=ones(3000*3000,1);
feature_B=ones(3000*3000,1);
feature_nDSM=ones(3000*3000,1);
feature_IR=ones(3000*3000,1);

% gt_=reshape(gt,[],1);
% 
% index_gt1=find(gt_==1);
% index_gt2=find(gt_==2);
% index_gt3=find(gt_==3);
% index_gt4=find(gt_==4);
% index_gt5=find(gt_==5);
% index_gt6=find(gt_==6);




for i=1:k
length_index=length(find(idx==i));
index(1:length_index)=find(idx==i);

feature_R(index)=mean(RGB_R(index));
feature_G(index)=mean(RGB_G(index));
feature_B(index)=mean(RGB_B(index));
feature_nDSM(index)=mean(nDSM(index));
feature_IR(index)=mean(IR(index));
end

feature_R=reshape(feature_R,[3000,3000]);
feature_G=reshape(feature_G,[3000,3000]);
feature_B=reshape(feature_B,[3000,3000]);

feature_RGB=ones(3000,3000,3);
feature_RGB(:,:,1)=feature_R;
feature_RGB(:,:,2)=feature_G;
feature_RGB(:,:,3)=feature_B;

figure
imshow(feature_RGB)
title(['kmeans segmented image with RGB features with k=' ,num2str(k), ' and max number of iterations= ',num2str(k_means_max_iter)])

%% SLIC




