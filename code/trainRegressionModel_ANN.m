function [trainedModel, validationRMSE] = trainRegressionModel_ANN(trainingData, responseData)
%
% convert the input to a table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12'};
predictors = inputTable(:, predictorNames);
response = responseData;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a regression model
% The following code specifies all model options and trains the model.
regressionNeuralNetwork = fitrnet(...
    predictors, ...
    response, ...
    'LayerSizes', 100, ...
    'Activations', 'relu', ...
    'Lambda', 0, ...
    'IterationLimit', 1000, ...
    'Standardize', true);

% Create result structure using predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
neuralNetworkPredictFcn = @(x) predict(regressionNeuralNetwork, x);
trainedModel.predictFcn = @(x) neuralNetworkPredictFcn(predictorExtractionFcn(x));

% Add fields to result structure
trainedModel.RegressionNeuralNetwork = regressionNeuralNetwork;
trainedModel.About = 'This structure is a trained model exported from Regression Learner R2021a.';
trainedModel.HowToPredict = sprintf('To make predictions on a new predictor column matrix X, use: \n yfit = c.predictFcn(X) \nReplace ''c'' with the name of the variable that is this structure, such as ''trainedModel''. \n \nX must contain exactly 24 columns because the model was trained with 24 predictors. \nX must contain only\npredictor columns in the exact same order and format as the training data. Do not include the response column or any columns that are not imported into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and responses
% The following code processes the data into a suitable shape to train the model.
%
% convert the input to a table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12'});
predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12'};
predictors = inputTable(:, predictorNames);
response = responseData;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];


% Computing Resubstituted Forecasts
validationPredictions = predict(trainedModel.RegressionNeuralNetwork, predictors);

% Calculation Verification RMSE
validationRMSE = sqrt(resubLoss(trainedModel.RegressionNeuralNetwork, 'LossFun', 'mse'));
