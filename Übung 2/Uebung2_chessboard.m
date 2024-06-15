%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Musterkennung Übung 2
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
scale  = 1/2
r      = 3   % tile row
c      = 14  % tile column
segments_sizes_vector = [15];

disp('----------------------------')

% Preallocation for performance
accuracies = zeros(length(segments_sizes_vector), 1);
times = zeros(length(segments_sizes_vector), 1);

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
% figure;
% imshow(gt, getColorMap('V2DLabels'));
% title(sprintf('Input labels, scale = %f',scale));


%% Chessboard

RGB_R=RGB(:,:,1);
RGB_G=RGB(:,:,2);
RGB_B=RGB(:,:,3);



label_image=zeros(length(RGB(:,:,1)));

feature_R=ones(length(RGB(:,:,1)));
feature_G=ones(length(RGB(:,:,1)));
feature_B=ones(length(RGB(:,:,1)));
feature_nDSM=ones(length(RGB(:,:,1)));
feature_IR=ones(length(RGB(:,:,1)));

% Loop through each segment size
for idx_current = 1:length(segments_sizes_vector)
   
    close all;
    
    tic;  % Start timing for each segment size
    segment_size = segments_sizes_vector(idx_current);
    segments = length(RGB(:,:,1)) / segment_size
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
    
    list_labels=reshape(list_labels,[i*j,1]);
    list_features=reshape(list_features,[i*j,5]);

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

    RGB_label_image = cat(3,r,g,b) * 255;

    feature_RGB=ones(length(RGB(:,:,1)),length(RGB(:,:,1)),3);
    feature_RGB(:,:,1)=feature_R;
    feature_RGB(:,:,2)=feature_G;
    feature_RGB(:,:,3)=feature_B;
    
    %% determine runtime
    times(idx_current) = toc;  % End timing and capture the elapsed time
    disp(['Elapsed Time: ', num2str(times(idx_current)), ' seconds']);
    
    
    %% plot images
    figure
    imshow(feature_RGB)
    title(['chessboard RGB features with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

    figure
    imshow(RGB_label_image)
    title(['chessboard labeled image with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments)])

    %% determine accuracy
    result_label_assignments = 1*imp_surf + 2*building + 3*low_veg + 4*tree + 5*car + 6*clutter;
    correctly_identified_labels = result_label_assignments == gt;
    accuracies(idx_current) = sum(correctly_identified_labels, "all") / (length(RGB(:,:,1))*length(RGB(:,:,1)))
end

%% Plot accuracies and times
% f = figure;
% yyaxis left;
% plot(segments_sizes_vector, accuracies*100, '-o');
% ylabel('Accuracy [%]');
% ylim([0 100]);  % Set y-axis limits from 0 to 100
% 
% 
% yyaxis right;
% plot(segments_sizes_vector, times, '-o');
% ylabel('Time (s)');
% ylim([0 max(times)]);  % Set y-axis limits from 0 to the maximum time
% 
% 
% xlabel('Segment Size [pixels]');
% title('Accuracy and Runtime for Various Segment Sizes');
% legend('Accuracy', 'Runtime', 'Location', 'best');
%grid on;  % This turns the grid lines on
% saveas(f, ['results_chessboard/accuracy_runtime_' datestr(now, 'yyyy-mm-dd_HHMMSS') '.png']);


%% Übung 2 - Random Forest Classification

trees=100;
Mdl=TreeBagger(trees,list_features,list_labels,Method="classification",OOBPrediction="on");

predicted_labels = oobPredict(Mdl);
predicted_labels=str2double(predicted_labels);

idx = zeros(length(RGB(:,:,1)));  % Pre-allocate idx for the maximum required size
predicted_label_image=zeros(length(RGB(:,:,1)));

for i=1:segments
        for j=1:segments

            x_start = (i-1)*segment_size + 1;
            x_end = i * segment_size;
            y_start = (j-1)*segment_size + 1;
            y_end = j * segment_size;
            idx(x_start:x_end, y_start:y_end) = i*j;  % Assign segment ID

            %
            predicted_label_image(x_start:x_end, y_start:y_end)=predicted_labels(i*j);
        end
end

imp_surf = predicted_label_image == 1;
building = predicted_label_image == 2;
low_veg  = predicted_label_image == 3;
tree     = predicted_label_image == 4;
car      = predicted_label_image == 5;
clutter  = predicted_label_image == 6;
    
r = imp_surf | car | clutter;
g = imp_surf | low_veg | tree | car;
b = imp_surf | building | low_veg;

predicted_RGB_label_image = cat(3,r,g,b) * 255;

figure
imshow(predicted_RGB_label_image)
title(['chessboard segmented image with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments),' and labeled by Random Forest with ',num2str(trees),' trees'])
