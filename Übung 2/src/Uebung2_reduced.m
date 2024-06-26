%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Musterkennung Übung 2
% Gruppe 1
% Christian Edelmann 3560916
% Lars Pfeiffer      3514519
% Nadim Maraqten     3384833
% Johannes Bladt     3541171
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; format longG; close all; clc;
warning('off', 'Images:initSize:adjustingMag');
rng(1);

% Load intermediate results from a .mat file
load('intermediate_results_ue1.mat');

% Create folder for results if it does not exist
output_folder = 'Results Lab 2';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% Random Forest mit Trainingsbild

tree_sizes = [1, 5, 10, 20 , 50];
runtime = zeros(length(tree_sizes));
main_diagonal = zeros(6, length(tree_sizes));
overall_accuracy = zeros(1, length(tree_sizes));
train_accuracy = zeros(1, length(tree_sizes));

for p = 1:length(tree_sizes)
    trees = tree_sizes(p);
    tic;
    Mdl = TreeBagger(trees, list_features_train, list_labels_train, 'Method', 'classification', 'OOBPrediction', 'on');
    runtime(p) = toc;

    predicted_labels_train = oobPredict(Mdl);
    predicted_labels_train = str2double(predicted_labels_train);

    predicted_label_image_train = zeros(length(RGB(:,:,1)));

    for i = 1:segments
        for j = 1:segments
            x_start = (i-1)*segment_size + 1;
            x_end = i * segment_size;
            y_start = (j-1)*segment_size + 1;
            y_end = j * segment_size;

            segment_index = (j-1)*segments + i;
            predicted_label_image_train(x_start:x_end, y_start:y_end) = predicted_labels_train(segment_index);
        end
    end

    imp_surf_train = predicted_label_image_train == 1;
    building_train = predicted_label_image_train == 2;
    low_veg_train = predicted_label_image_train == 3;
    tree_train = predicted_label_image_train == 4;
    car_train = predicted_label_image_train == 5;
    clutter_train = predicted_label_image_train == 6;

    r_train = imp_surf_train | car_train | clutter_train;
    g_train = imp_surf_train | low_veg_train | tree_train | car_train;
    b_train = imp_surf_train | building_train | low_veg_train;

    predicted_RGB_label_image_train = cat(3, r_train, g_train, b_train) * 255;

    figure
    imshow(predicted_RGB_label_image_train)
    title(['chessboard segmented training image with segment width=', num2str(segment_size), ...
        ' pix total segments= ', num2str(segments*segments), '\n and labeled by Random Forest with ', num2str(trees), ' trees'])

    saveas(gcf, fullfile(output_folder, ['training_image_trees_', num2str(trees), '.png']));

    %% Confusion Matrix for Training Data
    % Reshape the labels to vectors for comparison
    gt_vector_train = list_labels_train; % Already a vector
    predicted_label_train_vector = predicted_labels_train; % Already a vector
    C_train = confusionmat(gt_vector_train, predicted_label_train_vector);

    % Calculate accuracy for training data
    train_accuracy(p) = sum(gt_vector_train == predicted_label_train_vector) / length(gt_vector_train) * 100;

    figure
    C_train_chart = confusionchart(gt_vector_train, predicted_label_train_vector, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
    C_train_chart.Normalization = 'row-normalized';
    title(['Training Confusion Matrix, number of trees = ', num2str(trees), ', accuracy = ', num2str(train_accuracy(p), '%.2f'), '%'])

    saveas(gcf, fullfile(output_folder, ['confusion_matrix_train_trees_', num2str(trees), '.png']));

    %% Random Forest auf Testbild anwenden

    predicted_labels_test = predict(Mdl, list_features_test);
    predicted_labels_test = str2double(predicted_labels_test);

    predicted_label_test_image = zeros(length(RGB(:,:,1)));

    for i = 1:segments
        for j = 1:segments
            x_start = (i-1)*segment_size + 1;
            x_end = i * segment_size;
            y_start = (j-1)*segment_size + 1;
            y_end = j * segment_size;

            segment_index = (j-1)*segments + i;
            predicted_label_test_image(x_start:x_end, y_start:y_end) = predicted_labels_test(segment_index);
        end
    end

    imp_surf = predicted_label_test_image == 1;
    building = predicted_label_test_image == 2;
    low_veg = predicted_label_test_image == 3;
    tree = predicted_label_test_image == 4;
    car = predicted_label_test_image == 5;
    clutter = predicted_label_test_image == 6;

    r = imp_surf | car | clutter;
    g = imp_surf | low_veg | tree | car;
    b = imp_surf | building | low_veg;

    predicted_RGB_label_test_image = cat(3, r, g, b) * 255;

    figure
    imshow(predicted_RGB_label_test_image)
    title(['chessboard segmented test image with segment width=', num2str(segment_size), ...
        ' pix total segments= ', num2str(segments*segments), ' and labeled by Random Forest with ', num2str(trees), ' trees'])

    saveas(gcf, fullfile(output_folder, ['test_image_trees_', num2str(trees), '.png']));

    %% Confusionsmatrix

    gt_vector = double(reshape(gt, [], 1));
    predicted_label_test_image_vector = reshape(predicted_label_test_image, [], 1);
    C_test = confusionmat(gt_vector, predicted_label_test_image_vector);

    %% Accuracy and runtime
    difference = double(gt) - predicted_label_test_image;
    number_of_zeros = length(find(difference == 0));
    overall_accuracy(1, p) = number_of_zeros / numel(gt) * 100;

    for i = 1:6
        main_diagonal(i, p) = C_test(i, i) / sum(C_test(i, :)) * 100;
    end

    %% Confusionchart

    figure
    C_test_chart = confusionchart(gt_vector, predicted_label_test_image_vector, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
    C_test_chart.Normalization = 'row-normalized';
    title(['Confusion matrix (test), number of trees = ', num2str(trees), ', overall accuracy = ', num2str(overall_accuracy(p)), '%'])

    saveas(gcf, fullfile(output_folder, ['confusion_matrix_test_trees_', num2str(trees), '.png']));
end

%% Plotparameter
percentages = zeros(8, length(tree_sizes));
percentages(1:6, 1:end) = main_diagonal;
percentages(7, 1:end) = overall_accuracy;
percentages(8, 1:end) = train_accuracy;

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

title('Development of precision/accuracy and runtime for different tree sizes')
xlabel('number of trees')
legend({'Class 1: Impervious surfaces [Test]', 'Class 2: Building [Test]', 'Class 3: Low Vegetation [Test]', ...
    'Class 4: Tree [Test]', 'Class 5: Car [Test]', 'Class 6: Clutter/Background [Test]', ...
    'Test Overall accuracy', 'Train Overall accuracy', 'runtime'}, 'Location', 'eastoutside');
saveas(gcf, fullfile(output_folder, 'performance_plot.png'));
