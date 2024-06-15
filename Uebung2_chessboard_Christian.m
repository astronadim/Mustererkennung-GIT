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

  %% Random Forest mit Trainingsbild

tree_sizes=[10,25,50,100,250,500,1000];
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

for i=1:segments
        for j=1:segments

            x_start = (i-1)*segment_size + 1;
            x_end = i * segment_size;
            y_start = (j-1)*segment_size + 1;
            y_end = j * segment_size;
            

            segment_index=(j-1)*segments+i;
            predicted_label_image_train(x_start:x_end, y_start:y_end)=predicted_labels_train(segment_index);
        end
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
title(['chessboard segmented training image with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments),' and labeled by Random Forest with ',num2str(trees),' trees'])


%% Random Forest auf Testbild anwenden

predicted_labels_test=predict(Mdl,list_features_test);
predicted_labels_test=str2double(predicted_labels_test);

predicted_label_test_image=zeros(length(RGB(:,:,1)));

for i=1:segments
    for j=1:segments

        x_start = (i-1)*segment_size + 1;
        x_end = i * segment_size;
        y_start = (j-1)*segment_size + 1;
        y_end = j * segment_size;
        
        segment_index=(j-1)*segments+i;
        predicted_label_test_image(x_start:x_end, y_start:y_end)=predicted_labels_test(segment_index);
    end
end

imp_surf = predicted_label_test_image == 1;
building = predicted_label_test_image == 2;
low_veg  = predicted_label_test_image == 3;
tree     = predicted_label_test_image == 4;
car      = predicted_label_test_image == 5;
clutter  = predicted_label_test_image == 6;
    
r = imp_surf | car | clutter;
g = imp_surf | low_veg | tree | car;
b = imp_surf | building | low_veg;

predicted_RGB_label_test_image = cat(3,r,g,b) * 255;

figure
imshow(predicted_RGB_label_test_image)
title(['chessboard segmented test image with segment width=' ,num2str(segment_size), ' pix total segments= ',num2str(segments*segments),' and labeled by Random Forest with ',num2str(trees),' trees'])

%% Confusionsmatrix 

gt_vector=double(reshape(gt,[],1));
predicted_label_test_image_vector=reshape(predicted_label_test_image,[],1);
C_test=confusionmat(gt_vector,predicted_label_test_image_vector);


%% Accuracy and runtime
difference=double(gt)-predicted_label_test_image;
number_of_zeros=length(find(difference==0));
overall_accuracy(1,p)=number_of_zeros/numel(gt)*100;

for i=1:6
main_diagonal(i,p)=C_test(i,i)/sum(C_test(i,:))*100;
end

%% Confusionchart 

figure
C_test_chart=confusionchart(gt_vector,predicted_label_test_image_vector,'RowSummary','row-normalized','ColumnSummary','column-normalized');
C_test_chart.Normalization = 'row-normalized'; 
title(['Confusion matrix, number of trees = ',num2str(trees),', overall accuracy = ',num2str(overall_accuracy(p)),'%'])


end

%% Plotparameter
percentages=zeros(7,length(tree_sizes));
percentages(1:6,1:end)=main_diagonal;
percentages(7,1:end)=overall_accuracy;

figure
ax = gca;
yyaxis left
plot(tree_sizes,percentages(1,:),'-o','Color','k')
hold on
plot(tree_sizes,percentages(2,:),'-o','Color','b')
hold on
plot(tree_sizes,percentages(3,:),'-o','Color','c')
hold on
plot(tree_sizes,percentages(4,:),'-o','Color','g')
hold on
plot(tree_sizes,percentages(5,:),'-o','Color','y')
hold on
plot(tree_sizes,percentages(6,:),'-o','Color','r')
hold on
plot(tree_sizes,percentages(7,:),'--o','Color','k')
hold on
ylabel('precision/accuracy [%]')
ax.YColor = 'k';

hold on 
yyaxis right
plot(tree_sizes,runtime(:,1),'--o','Color','m')
ylabel('runtime [s]')
ax.YColor = 'm';


xticks('auto');
xticklabels('auto');

title('Development of precision/accuracy and runtime for different tree sizes')
xlabel('number of trees')
legend('Class 1: Impervious surfaces','Class 2: Building','Class 3: Low Vegetation', ...
       'Class 4: Tree','Class 5: Car','Class 6: Clutter/Background','Overall accuracy','runtime')