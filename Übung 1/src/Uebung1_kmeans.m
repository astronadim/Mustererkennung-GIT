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
scale  = 1/2
r      = 3   % tile row
c      = 14  % tile column
k_means_max_iter = 20
k_vector=[10,20,30,40,50];

disp('----------------------------')

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
title(sprintf('Input image RGB, scale = %f',scale));

figure;
imshow(IR);
title(sprintf('Input image IR, scale = %f',scale));

figure;
imshow(nDSM, []);
title(sprintf('Input image nDSM, scale = %f',scale));

figure;
imshow(gt, getColorMap('V2DLabels'));
title(sprintf('Input labels, scale = %f',scale));

RGB_R=reshape(RGB(:,:,1),[],1);
RGB_G=reshape(RGB(:,:,2),[],1);
RGB_B=reshape(RGB(:,:,3),[],1);
IR_=reshape(IR,[],1);
nDSM_=reshape(nDSM,[],1);
input_image=[RGB_R,RGB_G,RGB_B,IR_,nDSM_];


%% kmeans
for idx_current = 1:length(k_vector)
    close all;
    k = k_vector(idx_current)
    idx = zeros(length(RGB(:,:,1)));  % Pre-allocate idx for the maximum required size
    
    idx=kmeans(input_image,k,'MaxIter',k_means_max_iter);

    mask=reshape(idx,length(RGB(:,:,1)),length(RGB(:,:,1)));

    figure
    imshow(mask,[])
    title(['kmeans segmentation mask with k=' ,num2str(k), ' and max number of iterations= ',num2str(k_means_max_iter)])

    figure
    bmask = boundarymask(mask);
    imshow(imoverlay(RGB,bmask,'cyan'))
    title(['kmeans boundary mask with k=' ,num2str(k), ' and max number of iterations= ',num2str(k_means_max_iter)])

    feature_R=ones(length(RGB(:,:,1))*length(RGB(:,:,1)),1);
    feature_G=ones(length(RGB(:,:,1))*length(RGB(:,:,1)),1);
    feature_B=ones(length(RGB(:,:,1))*length(RGB(:,:,1)),1);
    feature_nDSM=ones(length(RGB(:,:,1))*length(RGB(:,:,1)),1);
    feature_IR=ones(length(RGB(:,:,1))*length(RGB(:,:,1)),1);

    gt_=reshape(gt,[],1);
    label_image=idx;
    list_labels=zeros(max(idx),1);
    list_features=zeros(max(idx),5);

    for i = 1:max(idx)
        segment_indices = find(idx == i);  % Only find indices once to save computation

        segment_gt_label = mode(gt_(segment_indices));
        label_image(idx == i) = segment_gt_label;  % Assign labels based on segment index
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

    label_image=reshape(label_image,[length(RGB(:,:,1)),length(RGB(:,:,1))]);

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

    feature_R=reshape(feature_R,[length(RGB(:,:,1)),length(RGB(:,:,1))]);
    feature_G=reshape(feature_G,[length(RGB(:,:,1)),length(RGB(:,:,1))]);
    feature_B=reshape(feature_B,[length(RGB(:,:,1)),length(RGB(:,:,1))]);

    feature_RGB=ones(length(RGB(:,:,1)),length(RGB(:,:,1)),3);
    feature_RGB(:,:,1)=feature_R;
    feature_RGB(:,:,2)=feature_G;
    feature_RGB(:,:,3)=feature_B;

    %% determine runtime
    times(idx_current) = toc;  % End timing and capture the elapsed time
    disp(['Elapsed Time: ', num2str(times(idx_current)), ' seconds'])

    figure
    imshow(feature_RGB,[])
    title(['kmeans segmented image with RGB features with k=' ,num2str(k), ' and max number of iterations= ',num2str(k_means_max_iter)])

    figure
    imshow(RGB_label_image,[])
    title(['kmeans segmented image with labels with k=' ,num2str(k), ' and max number of iterations= ',num2str(k_means_max_iter)])
    

    %% determine accuracy
    result_label_assignments = 1*imp_surf + 2*building + 3*low_veg + 4*tree + 5*car + 6*clutter;
    correctly_identified_labels = result_label_assignments == gt;
    accuracies(idx_current) = sum(correctly_identified_labels, "all") / (length(RGB(:,:,1))*length(RGB(:,:,1)))

end


%% Plot accuracies and times
f = figure;
yyaxis left;
plot(k_vector, accuracies*100, '-o');
ylabel('Accuracy [%]');
ylim([0 100]);  % Set y-axis limits from 0 to 100


yyaxis right;
plot(k_vector, times, '-o');
ylabel('Time (s)');
ylim([0 max(times)]);  % Set y-axis limits from 0 to the maximum time


xlabel('Cluster numbers k');
title(['Accuracy and Runtime for Various Cluster Numbers k and ', num2str(k_means_max_iter), ' max. iterations']);
legend('Accuracy', 'Runtime', 'Location', 'best');
% saveas(f, ['results_kmeans/accuracy_runtime_' datestr(now, 'yyyy-mm-dd_HHMMSS') '.png']);

