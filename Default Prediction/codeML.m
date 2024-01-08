%% SECTION TITLE
% *LOAN DEFAULT PREDICTION* 



%% 1.Read data and summary
credit_data= readtable('UCI_Credit_Card.csv');
head(credit_data, 100);

credit_data = credit_data(: , 2:end);
colunm_units = credit_data.Properties.VariableUnits;
summary(credit_data);

missingvalues= sum(ismissing(credit_data), 'all');

VariableNames = credit_data.Properties.VariableNames;
%% 2.Variable statistics

ColsIdx = cellfun(@(x) isnumeric(credit_data.(x)), credit_data.Properties.VariableNames); %cellfunc apply isnumeric function in each Variable of credit data.
numericColumns = credit_data{:, ColsIdx};

summaryStats = array2table([mean(numericColumns); ...
    std(numericColumns); ...
    min(numericColumns); ...
    prctile(numericColumns, 25); ...
    median(numericColumns); ...
    prctile(numericColumns, 75); ...
    max(numericColumns)], ...
    'VariableNames', credit_data.Properties.VariableNames, ...
    'RowNames', {'mean', 'std', 'min', '25%', 'median', '75%', 'max'});

% Rounding values to 2 decimals.
summaryStatsRounded = array2table(round(table2array(summaryStats), 2), ...
    'VariableNames', summaryStats.Properties.VariableNames, ...
    'RowNames', summaryStats.Properties.RowNames);

% No scientific notation
format bank; 
disp(summaryStatsRounded);

formattedRoundedStats = table2cell(summaryStatsRounded);

writetable(summaryStatsRounded, 'summary_statistics.xlsx', 'WriteRowNames', true);

%% 2.1.Target Class imbalance:Pie Chart
target= credit_data.default_payment_next_month;


% Filter the data when target = 0 (non-default) and target = 1 (default)
nonDefaultData = credit_data(target == 0, :);
defaultData = credit_data(target == 1, :);

% Calculate percentages

nondefaultpercentage = height(nonDefaultData) / height(credit_data) * 100;
defaultpercentage = height(defaultData) / height(credit_data) * 100;

labels = {'Non-default', 'Default'};
figure;
colors = [0.2 0.7 0.5;  %Soft green
          0.9 0.5 0.3];
explode = [0 1]; %Visualization effect of space between two classes
pie([nondefaultpercentage defaultpercentage], explode, labels );
colormap(colors);


%% 2.2.Heatmap 

data_array = table2array(credit_data);


% Calculate correlation matrix
corr_matrix = corr(data_array, 'Type', 'Pearson');

HeatMap
% Create heatmap
figure;
%labels
HeatMap
hm = HeatMap(corr_matrix,'Colormap',redgreencmap, ...
    'ColumnLabels',VariableNames,'RowLabels', VariableNames, ...
    'ColumnLabelsRotate', 45 ,...
    'Annotate',true);

title = addTitle(hm,'Correlation matrix','Color','black');
title.FontSize = 12;




%% 2.3 AGE boxplots with target classes.

% Extracting target classes from AGE variable.
default = credit_data.AGE(credit_data.default_payment_next_month == 1);
nondefault = credit_data.AGE(credit_data.default_payment_next_month == 0);

% Generating indices.
group = [zeros(size(nondefault)); ones(size(default))];

% Boxplot data
boxplotData = [nondefault; default];

% Creating boxplot for the 'AGE' variable using target variable.
figure;
boxplot(boxplotData, group, 'Labels', {'Non-default credit ', 'Default credit'});
xlabel('Payment Type');
ylabel('Age');
title('Boxplot of Age for Default and Non default Payments');



%% 2.3.1 Automatazing variables box plots

column_names = credit_data.Properties.VariableNames;
excluded_columns = {'ID', 'SEX', 'default_payment_next_month'}; %Columns not needed.

column_names = column_names(~ismember(column_names, excluded_columns)); %excluding columns not needed.
% Define subplots
num_vars = numel(column_names);
cols = 3; % Number of columns in the subplot
rows = ceil(num_vars / cols); % Rows needed

figure('Position', [100, 100, 1200, 800]);

for i = 1:num_vars
    subplot(rows, cols, i); 
     % Create a subplot for each variable
    defaultvar = credit_data.(column_names{i})(credit_data.default_payment_next_month == 1);
    nondefaultvar = credit_data.(column_names{i})(credit_data.default_payment_next_month == 0);
    
    % Generating group indices for boxplots
    group = [zeros(size(nondefaultvar)); ones(size(defaultvar))];
    
    % Data for boxplots
    boxplotData = [nondefaultvar; defaultvar];
    
    % Creating boxplots for each variable separated by default and non default credit client data
    boxplot(boxplotData, group, 'Labels', {'Non-default credit', 'Default credit'});
    ylabel(column_names{i}); % Variable name
    
end

%% 2.4 Distribution Visualization
figure;
age_data= credit_data.AGE;
hold on
histfit(age_data);
hold off
%% 2.4.1 Automatizing Distribution Visualization

%Selecting continuos variables.
columns = {'AGE', 'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',...
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'}; 

figure('Position', [100, 100, 3500, 3500]);

for i = 1:length(columns)
    subplot(5, 5, i); % Creating a subplots
    
    variable_data = credit_data.(columns{i}); % data for each selected column
    
    histfit(variable_data); % Plotting histogram with a fitted curve
    
    ylabel(columns{i}); % Setting a title for each subplot
end

%% 3.Data partion and DT base model


%First data split
rng(13);
cv = cvpartition(size(credit_data, 1), "HoldOut",0.2);
dataTrain = credit_data(cv.training,:);
dataTest = credit_data(cv.test, :);

%Second data split
Xtrain = dataTrain{:, 1:end-1};
ytrain = dataTrain{: , end};
Xtest = dataTest{:, 1:end-1};
ytest = dataTest{:, end};
FirstDT = fitctree(dataTrain,"default_payment_next_month");
trainLoss = resubLoss(FirstDT);

%Evaluating missclasification error in Train and Test set.
FirstDTLoss = loss(FirstDT,dataTest, 'default_payment_next_month'); % how well tree classifies the data 
predictionsDT1 = predict(FirstDT, Xtest);
actualLabelsDT1 = dataTest.default_payment_next_month;



% Acc in test set.
accuracyDT = sum(predictionsDT1 == actualLabelsDT1) / numel(actualLabelsDT1);
%% 3.1 DT Optimization 

%Setting options and K-Fold to 10 splits.
cvpt = cvpartition(dataTrain.default_payment_next_month,"KFold",10); 
opt = struct('Optimizer','bayesopt','AcquisitionFunctionName',...
    'expected-improvement-plus', "CVPartition",cvpt,"MaxObjectiveEvaluations",25);
rng(13);
%Hyperparameter Optimization
mdl = fitctree(dataTrain, 'default_payment_next_month',...
    'OptimizeHyperparameters','all', ...
    'HyperparameterOptimizationOptions', opt);


%Evaluating missclasification error in Train and Test set.
trainLossOptim = resubLoss(mdl);
testLossOptim = loss(mdl,dataTest, 'default_payment_next_month');
parameters = mdl.ModelParameters;
%% 3.1.1 Model retraining

bestModel = fitctree(dataTrain, 'default_payment_next_month', ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', parameters.MaxSplits, ...
    'MinParent', parameters.MinParent, ... 
    'MinLeafSize', parameters.MinLeaf, ...
    'NumVariablesToSample', parameters.NVarToSample, ...
    'MergeLeaves', parameters.MergeLeaves, ...
    'Prune', parameters.Prune, ...
    'PruneCriterion', parameters.PruneCriterion, ...
    'Surrogate', 'off');  
 
trainLossbestModel = resubLoss(bestModel);
testLossbestModel = loss(bestModel, dataTest, 'default_payment_next_month');
%% 3.1.2 Feature importance
%View DT model.
view(mdl, 'Mode', 'graph');
importance= predictorImportance(mdl);

%Plotting feature importance.
figure;
hold on
yticks(1:24); 
yticklabels(VariableNames)
barh(importance);
xlabel('Features');
ylabel('Importance');
title('Feature Importance Scores');
grid on;



%%  3.1.3 DT Metrics 
tic;
% Makeing predictions.
predictions = predict(bestModel, Xtest);
truevalues = dataTest.default_payment_next_month;
DTtesttime=toc; % Measuring model time in test set.

% Create confusion matrix manually using confusionmat
ConfusionM = confusionmat(truevalues, predictions);
figure;
cmatrix= confusionchart(truevalues,predictions);
cmatrix.ColumnSummary = 'column-normalized';
cmatrix.RowSummary = 'row-normalized';
cmatrix.Title = 'Decision Tree Confusion Matrix';

% Accuracy, precission, recall, F1-Score
accuracy = sum(predictions == truevalues) / numel(truevalues);

confmatrix = confusionmat(ytest, predictions);
cmtx = confmatrix';
diagonal = diag(cmtx);
sum_rows = sum(cmtx, 2);
precision = diagonal./ sum_rows;
overall_precision = mean(precision);
sum_columns = sum(cmtx, 1);
recall = diagonal ./sum_columns';
overall_recall = mean(recall);
f1_score= 2*((overall_precision * overall_recall)/ (overall_precision + overall_recall));


%% 3.1.4 Roc curve


[~, scores] = predict(mdl, Xtest); %Predictions scores(probabilities)

diffscore= scores(: ,2); %scores for class 1.

[X, Y, ~, AUC] = perfcurve(ytest, diffscore, 1); %Calculate X, Y and AUC values


%Plotting AUC ROC Curve
figure;
hold on;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve in DT');

plot([0,1], [0,1], '--');
hold off;


%% 4.RF base model
rng(13);
% Fittin RF base model.
mdlensemble2 = fitcensemble(dataTrain,"default_payment_next_month","Method","Bag");
lossEns2 = resubLoss(mdlensemble2);

parametersRF1 = mdlensemble2.ModelParameters;
predictionsRF1 = predict(mdlensemble2, Xtest);
actualLabelsRF1 = dataTest.default_payment_next_month;


% Calculate performance  acc. metric
accuracyRF1 = sum(predictionsRF1 == actualLabelsRF1) / numel(actualLabelsRF1); % Accuracy


%% 4.1 RF Optimization
rng(13);


% Setting up the cross-validation partitioning
cvpt = cvpartition(dataTrain.default_payment_next_month, 'KFold', 10);
opt = struct('Optimizer','bayesopt','AcquisitionFunctionName','expected-improvement-plus' ...
    ,'CVPartition', cvpt, 'MaxObjectiveEvaluations', 25);

% Hyperparameter Optimization setting search hyperparameters to a smaller space.
mdlRfTree = fitcensemble(dataTrain, 'default_payment_next_month', 'Method', 'Bag', ...
    'OptimizeHyperparameters', {'NumLearningCycles','SplitCriterion','MinLeafSize'}, ...  %'NumLearningCycles'
    'HyperparameterOptimizationOptions', opt);

%Loss in train and test set.
trainLossOptimRF = resubLoss(mdlRfTree);
testLossOptimRF = loss(mdlRfTree, dataTest, 'default_payment_next_month');
parametersRF = mdlRfTree.ModelParameters;


%% 4.2 RF metrics
tic;
predictionsRF = predict(mdlRfTree, Xtest); 
actualLabelsRF = dataTest.default_payment_next_month;
Rftimetest= toc; %RF model time in test set.
% Calculate performance metrics
accuracyRF = sum(predictionsRF == actualLabelsRF) / numel(actualLabelsRF); % Accuracy


% Plotting confusion matrix manually using confusionmat
C = confusionmat(actualLabelsRF, predictionsRF);
figure;
cmatrix= confusionchart(actualLabelsRF,predictionsRF);
cmatrix.ColumnSummary = 'column-normalized';
cmatrix.RowSummary = 'row-normalized';
cmatrix.Title = 'RF Confusion Matrix';


%calculating precision, recall, F1-score
cmt = C';
diagonal = diag(cmt);
sum_rows = sum(cmt, 2);
precisionRF = diagonal./ sum_rows;
overallprecisionRF = mean(precision);
sum_columns = sum(cmt, 1);
recall = diagonal ./sum_columns';
overallrecallRF = mean(recall);
f1scoreRf= 2*((overallprecisionRF * overallrecallRF)/ (overallprecisionRF + overallrecallRF));

%% 4.3 ROC curve
[~, scores] = predict(mdlRfTree, Xtest);

diffscore= scores(: ,2);

[X, Y, T, AUCRF] = perfcurve(ytest, diffscore, 1);

figure;
plot([0,1], [0,1], '--');
hold on;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve in RF');

hold off;
%% 4.5 Variable importance
importance= predictorImportance(mdlRfTree);

figure;
hold on
yticks(1:24); 
yticklabels(VariableNames); 
barh(importance);
xlabel('Features');
ylabel('Importance');
title('Feature Importance Scores RF');
 
grid on;

%% Download dataTest

writetable(dataTest, 'dataTest.csv');

%%

save('finalRFmodel.mat', 'mdlRfTree');
save('finalDTmodel.mat', 'mdl');


%% RUN in Test Set.
dataTest = readtable('dataTest.csv');
load('finalDTmodel.mat');
load('finalRFmodel.mat');

%Data partition
Xtest = dataTest{:, 1:end-1};
ytest = dataTest{:, end};

%DT predictions Accuracy and CM
rng(13);
predictionsDT = predict(mdl, Xtest);
actualLabelsDT = dataTest.default_payment_next_month;
% Acc in test set.
accuracyDT = sum(predictionsDT == actualLabelsDT) / numel(actualLabelsDT);
% Confusion matrix DT
ConfusionM = confusionmat(actualLabelsDT, predictionsDT);
figure;
cmatrix= confusionchart(actualLabelsDT,predictionsDT);
cmatrix.ColumnSummary = 'column-normalized';
cmatrix.RowSummary = 'row-normalized';
cmatrix.Title = 'Decision Tree Confusion Matrix';

%RF model
rng(13);
predictionsRF = predict(mdlRfTree, Xtest); 
actualLabelsRF = dataTest.default_payment_next_month;
% Calculate performance metrics
accuracyRF = sum(predictionsRF == actualLabelsRF) / numel(actualLabelsRF); % Accuracy

% Plotting confusion matrix 
C = confusionmat(actualLabelsRF, predictionsRF);
figure;
cmatrix= confusionchart(actualLabelsRF,predictionsRF);
cmatrix.ColumnSummary = 'column-normalized';
cmatrix.RowSummary = 'row-normalized';
cmatrix.Title = 'RF Confusion Matrix';

%% REFERENCES

% 2.Pie Chart: code adapted from https://uk.mathworks.com/help/matlab/ref/pie.html?searchHighlight=pie%20arguments&s_tid=srchtitle_support_results_1_pie%20arguments
% 2.1 % Heatmap:code adapted from https://uk.mathworks.com/help/matlab/ref/heatmap.html?s_tid=doc_ta
%3.Data partion and DT base model code adapted from:
%https://uk.mathworks.com/matlabcentral/answers/513374-print-parameters-of-classification-model?s_tid=srchtitle_site_search_1_split%20criterion%20%2527gdi%2527
%https://matlabacademy.mathworks.com/R2023b/portal.html?course=mlml#chapter=3&lesson=3&section=4
%3.1.3 DT Metrics: code adapted from https://uk.mathworks.com/help/stats/confusionchart.html?searchHighlight=confusionchart&s_tid=srchtitle_support_results_1_confusionchart
%3.1.4 Roc curve: code adapted from https://uk.mathworks.com/help/stats/classificationtree.predict.

