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
tic;  % Start timing




%% Parameters:
% Input:
scale  = 1/2
r      = 2   % tile row
c      = 13  % tile column
segment_size=10
segments=3000/10
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
figure;
imshow(gt, getColorMap('V2DLabels'));
title(sprintf('Input labels, scale = %f',scale));


%% Chessboard

RGB_R=RGB(:,:,1);
RGB_G=RGB(:,:,2);
RGB_B=RGB(:,:,3);



label_image=zeros(3000);

feature_R=ones(3000);
feature_G=ones(3000);
feature_B=ones(3000);
feature_nDSM=ones(3000);
feature_IR=ones(3000);

for i=1:segments
    for j=1:segments
        idx(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size)=i*j*ones(segment_size);
        segment_gt_label=mode(gt(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size),'all');
        label_image(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size)=segment_gt_label;

        feature_R(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size)=mean(RGB_R(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size),'all');
        feature_G(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size)=mean(RGB_G(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size),'all');
        feature_B(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size)=mean(RGB_B(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size),'all');
        feature_nDSM(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size)=mean(nDSM(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size),'all');
        feature_IR(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size)=mean(IR(((i-1)*segment_size)+1:i*segment_size,((j-1)*segment_size)+1:j*segment_size),'all');

    end
end
idx=reshape(idx,[],1);

mask=reshape(idx,3000,3000);

figure
imshow(mask,[])
title(['chessboard segmentation mask with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

figure
bmask = boundarymask(mask);
imshow(imoverlay(RGB,bmask,'cyan'))
title(['chessboard segments on RGB image with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

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

feature_RGB=ones(3000,3000,3);
feature_RGB(:,:,1)=feature_R;
feature_RGB(:,:,2)=feature_G;
feature_RGB(:,:,3)=feature_B;

figure
imshow(feature_RGB)
title(['chessboard RGB features with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

figure
imshow(RGB_label_image)
title(['chessboard labeled image with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

%% determine accuracy
result_label_assignments = 1*imp_surf + 2*building + 3*low_veg + 4*tree + 5*car + 6*clutter;
correctly_identified_labels = result_label_assignments == gt;
accuracy = sum(correctly_identified_labels, "all") / (3000*3000)

%% determine runtime
elapsedTime = toc;  % End timing and capture the elapsed time
disp(['Elapsed Time: ', num2str(elapsedTime), ' seconds']);

