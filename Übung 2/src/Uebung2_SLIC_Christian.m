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

% Create folder for results if it does not exist
output_folder = 'Results Lab 2 slic';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end


%% Parameters:
% Input:
scale  = 1/2;
row = [2,3];   % tile row for training
column = [13,14];  % tile column for training
tree_sizes=[1,5,10,20,50];
k=4000;
compactness=1;
method='slic';




%% Load Training Data:
% Load data by row / column number:


RGBIR = single(d_RGBIR.loadData(row(1), column(1)))/255;
RGB   = RGBIR(:,:,1:3);
IR    = RGBIR(:,:,4);

nDSM  = single(d_nDSM.loadData(row(1), column(1)));
nDSM  = (nDSM - min(nDSM(:))) / (max(nDSM(:)) - min(nDSM(:)));

gt_train    = d_GT.loadData(row(1), column(1));
gt_train    = uint8(data.potsdam.rgbLabel2classLabel(gt_train));

RGB   = imresize(RGB,  scale,  'method', 'nearest');
nDSM  = imresize(nDSM, scale,  'method', 'nearest');
IR    = imresize(IR,   scale,  'method', 'nearest');
gt_train    = imresize(gt_train,   scale,  'method', 'nearest');

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
imshow(gt_train, getColorMap('V2DLabels'));
title(sprintf('Ground truth image, scale = %f',scale));

    RGB_R=reshape(RGB(:,:,1),[],1);
RGB_G=reshape(RGB(:,:,2),[],1);
RGB_B=reshape(RGB(:,:,3),[],1);
IR_=reshape(IR,[],1);
nDSM_=reshape(nDSM,[],1);

input_image=RGB;  % SLIC uses the RGB image

%% SLIC

   
    [idx_, ~] = superpixels(input_image, k, 'Compactness', compactness, 'Method', method);
    
    

    
    idx_train = reshape(idx_, [], 1);
    gt_train = reshape(gt_train, [], 1);
    
    feature_R = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_G = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_B = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_nDSM = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_IR = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    
    label_image = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);

  

    for i = 1:max(idx_train)
        segment_indices = find(idx_train == i);
        
        segment_gt_label = mode(gt_train(segment_indices));
        label_image(segment_indices) = segment_gt_label;
        list_labels_train(i)=segment_gt_label;
        
        % Compute mean values for each feature using the segment indices
        feature_R(segment_indices) = mean(RGB_R(segment_indices));
        feature_G(segment_indices) = mean(RGB_G(segment_indices));
        feature_B(segment_indices) = mean(RGB_B(segment_indices));
        feature_nDSM(segment_indices) = mean(nDSM_(segment_indices));
        feature_IR(segment_indices) = mean(IR_(segment_indices));

        list_features_train(i,:)=[mean(RGB_R(segment_indices));
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
    
   imp_surf = label_image == 1;
    building = label_image == 2;
    low_veg  = label_image == 3;
    tree     = label_image == 4;
    car      = label_image == 5;
    clutter  = label_image == 6;

    r = imp_surf | car | clutter;
    g = imp_surf | low_veg | tree | car;
    b = imp_surf | building | low_veg;

    RGB_label_image_train = cat(3,r,g,b) * 255;
    
   
    
   

   
    
%% Load Test Data:
% Load data by row / column number:


RGBIR = single(d_RGBIR.loadData(row(2), column(2)))/255;
RGB   = RGBIR(:,:,1:3);
IR    = RGBIR(:,:,4);

nDSM  = single(d_nDSM.loadData(row(2), column(2)));
nDSM  = (nDSM - min(nDSM(:))) / (max(nDSM(:)) - min(nDSM(:)));

gt_test    = d_GT.loadData(row(2), column(2));
gt_test    = uint8(data.potsdam.rgbLabel2classLabel(gt_test));

RGB   = imresize(RGB,  scale,  'method', 'nearest');
nDSM  = imresize(nDSM, scale,  'method', 'nearest');
IR    = imresize(IR,   scale,  'method', 'nearest');
gt_test    = imresize(gt_test,   scale,  'method', 'nearest');

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
imshow(gt_test, getColorMap('V2DLabels'));
title(sprintf('Ground truth image, scale = %f',scale));

    RGB_R=reshape(RGB(:,:,1),[],1);
RGB_G=reshape(RGB(:,:,2),[],1);
RGB_B=reshape(RGB(:,:,3),[],1);
IR_=reshape(IR,[],1);
nDSM_=reshape(nDSM,[],1);

input_image=RGB;  % SLIC uses the RGB image

%% SLIC

   
    [idx_, ~] = superpixels(input_image, k, 'Compactness', compactness, 'Method', method);
    
    
   
    
    idx_test = reshape(idx_, [], 1);
    gt_test = reshape(gt_test, [], 1);
    
    feature_R = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_G = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_B = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_nDSM = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    feature_IR = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);
    
    label_image = zeros(length(RGB(:,:,1))*length(RGB(:,:,1)), 1);

    

    for i = 1:max(idx_test)
        segment_indices = find(idx_test == i);
        
        segment_gt_label = mode(gt_test(segment_indices));
        label_image(segment_indices) = segment_gt_label;
        list_labels_test(i)=segment_gt_label;
        
        % Compute mean values for each feature using the segment indices
        feature_R(segment_indices) = mean(RGB_R(segment_indices));
        feature_G(segment_indices) = mean(RGB_G(segment_indices));
        feature_B(segment_indices) = mean(RGB_B(segment_indices));
        feature_nDSM(segment_indices) = mean(nDSM_(segment_indices));
        feature_IR(segment_indices) = mean(IR_(segment_indices));

        list_features_test(i,:)=[mean(RGB_R(segment_indices));
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
    
   imp_surf = label_image == 1;
    building = label_image == 2;
    low_veg  = label_image == 3;
    tree     = label_image == 4;
    car      = label_image == 5;
    clutter  = label_image == 6;

    r = imp_surf | car | clutter;
    g = imp_surf | low_veg | tree | car;
    b = imp_surf | building | low_veg;

    RGB_label_image_test = cat(3,r,g,b) * 255;
    
    
 
    gt_train=reshape(gt_train,length(RGB),length(RGB));
    gt_test=reshape(gt_test,length(RGB),length(RGB));


    figure
    imshow(RGB_label_image_train)
    title(['SLIC labeled training image'])
    saveas(gcf, fullfile(output_folder, ['SLIC_training_image_segmentation.png']));

    figure
    imshow(RGB_label_image_test)
    title(['SLIC labeled test image '])
    saveas(gcf, fullfile(output_folder, ['SLIC_test_image_segmentation.png']));

  %% Random Forest mit Trainingsbild


runtime=zeros(length(tree_sizes));
main_diagonal=zeros(6,length(tree_sizes));
overall_accuracy=zeros(1,length(tree_sizes));



for p=1:length(tree_sizes)


trees=tree_sizes(p);

tic;
Mdl=TreeBagger(trees,list_features_train,list_labels_train,Method="classification",OOBPrediction="on");
runtime(p)=toc;

predicted_labels_train = oobPredict(Mdl);
predicted_labels_train=str2double(predicted_labels_train);

predicted_label_image_train=zeros(length(RGB(:,:,1)));

for i = 1:max(idx_train)
        segment_indices = find(idx_train == i);
        predicted_label_image_train(segment_indices)=predicted_labels_train(i);
end

imp_surf_train = predicted_label_image_train == 1;
building_train = predicted_label_image_train == 2;
low_veg_train  = predicted_label_image_train == 3;
tree_train     = predicted_label_image_train == 4;
car_train      = predicted_label_image_train == 5;
clutter_train  = predicted_label_image_train == 6;
    
r_train = imp_surf_train | car_train | clutter_train;
g_train = imp_surf_train | low_veg_train | tree_train | car_train;
b_train = imp_surf_train | building_train | low_veg_train;

predicted_RGB_label_image_train = cat(3,r_train,g_train,b_train) * 255;

figure
imshow(predicted_RGB_label_image_train)
title(['SLIC segmented training image labeled by Random Forest with ',num2str(trees),' trees'])
saveas(gcf, fullfile(output_folder, ['SLIC_training_image_', num2str(trees),'_trees.png']));

%% Confusionsmatrix 

gt_vector_train=double(reshape(gt_train,[],1));
predicted_label_image_train_vector=reshape(predicted_label_image_train,[],1);
C_train=confusionmat(gt_vector_train,predicted_label_image_train_vector);


%% Accuracy and runtime
difference=double(gt_train)-predicted_label_image_train;
number_of_zeros=length(find(difference==0));
overall_accuracy_train(1,p)=number_of_zeros/numel(gt_train)*100;


%% Confusionchart 

figure
C_train_chart=confusionchart(gt_vector_train,predicted_label_image_train_vector,'RowSummary','row-normalized','ColumnSummary','column-normalized');
C_train_chart.Normalization = 'row-normalized'; 
title(['Confusion matrix training image, number of trees = ',num2str(trees),', overall accuracy = ',num2str(overall_accuracy_train(p)),'%'])
saveas(gcf, fullfile(output_folder, ['SLIC_training_image_', num2str(trees),'_trees_confusion.png']));

%% Random Forest auf Testbild anwenden

predicted_labels_test=predict(Mdl,list_features_test);
predicted_labels_test=str2double(predicted_labels_test);

predicted_label_image_test=zeros(length(RGB(:,:,1)));

for i = 1:max(idx_test)
        segment_indices = find(idx_test == i);
        predicted_label_image_test(segment_indices)=predicted_labels_test(i);
end

imp_surf = predicted_label_image_test == 1;
building = predicted_label_image_test == 2;
low_veg  = predicted_label_image_test == 3;
tree     = predicted_label_image_test == 4;
car      = predicted_label_image_test == 5;
clutter  = predicted_label_image_test == 6;
    
r = imp_surf | car | clutter;
g = imp_surf | low_veg | tree | car;
b = imp_surf | building | low_veg;

predicted_RGB_label_image_test = cat(3,r,g,b) * 255;

figure
imshow(predicted_RGB_label_image_test)
title(['SLIC segmented test image labeled by Random Forest with ',num2str(trees),' trees'])
saveas(gcf, fullfile(output_folder, ['SLIC_test_image_', num2str(trees),'_trees.png']));

%% Confusionsmatrix 

gt_vector_test=double(reshape(gt_test,[],1));
predicted_label_image_test_vector=reshape(predicted_label_image_test,[],1);
C_test=confusionmat(gt_vector_test,predicted_label_image_test_vector);


%% Accuracy and runtime
difference=double(gt_test)-predicted_label_image_test;
number_of_zeros=length(find(difference==0));
overall_accuracy_test(1,p)=number_of_zeros/numel(gt_test)*100;

for i=1:6
main_diagonal(i,p)=C_test(i,i)/sum(C_test(i,:))*100;
end

%% Confusionchart 

figure
C_test_chart=confusionchart(gt_vector_test,predicted_label_image_test_vector,'RowSummary','row-normalized','ColumnSummary','column-normalized');
C_test_chart.Normalization = 'row-normalized'; 
title(['Confusion matrix test image, number of trees = ',num2str(trees),', overall accuracy = ',num2str(overall_accuracy_test(p)),'%'])
saveas(gcf, fullfile(output_folder, ['SLIC_test_image_', num2str(trees),'_trees_confusion.png']));

end

%% Plotparameter
percentages = zeros(8, length(tree_sizes));
percentages(1:6, 1:end) = main_diagonal;
percentages(7, 1:end) = overall_accuracy_test;
percentages(8, 1:end) = overall_accuracy_train;

figure
ax = gca;
yyaxis left
plot(tree_sizes, percentages(1, :), '-o', 'Color', 'k')
hold on
plot(tree_sizes, percentages(2, :), '-o', 'Color', 'b')
hold on
plot(tree_sizes, percentages(3, :), '-o', 'Color', 'c')
hold on
plot(tree_sizes, percentages(4, :), '-o', 'Color', 'g')
hold on
plot(tree_sizes, percentages(5, :), '-o', 'Color', 'y')
hold on
plot(tree_sizes, percentages(6, :), '-o', 'Color', 'r')
hold on
plot(tree_sizes, percentages(7, :), '--o', 'Color', 'k')
hold on
plot(tree_sizes, percentages(8, :), '--o', 'Color', 'b')
hold on
ylabel('precision/accuracy [%]')
ax.YColor = 'k';

hold on
yyaxis right
plot(tree_sizes, runtime(:, 1), '--o', 'Color', 'm')
ylabel('runtime [s]')
ax.YColor = 'm';

xticks('auto');
xticklabels('auto');
grid on
grid minor

title('Development of precision/accuracy and runtime for different tree sizes')
xlabel('number of trees')
legend({'Class 1: Impervious surfaces [Test]', 'Class 2: Building [Test]', 'Class 3: Low Vegetation [Test]', ...
    'Class 4: Tree [Test]', 'Class 5: Car [Test]', 'Class 6: Clutter/Background [Test]', ...
    'Test Overall accuracy', 'Train Overall accuracy', 'runtime'},'Position',[0.75 0.15 0.14 0.12]);
saveas(gcf, fullfile(output_folder, ['SLIC_results.png']));