clc;clear;format compact;close all;tic
rng('default')
%% data preprocessing
load eemd_data
imf=u;
c=size(imf,1);
pre_result=[];
true_result=[];
data=xlsread('data.xlsx','E2:E721');
[x,y]=data_process(data,12);%The previous 12 moments predict the next moment
n=size(x,1);
m=round(n*0.7);
YTest_final=y(m+1:end,:)';

%% Model each component
for i=1:c
disp(['Model the',num2str(i),'component'])
[x,y]=data_process(imf(i,:),12);
%Normalization
[xs,mappingx]=mapminmax(x',0,1);x=xs';
[ys,mappingy]=mapminmax(y',0,1);y=ys';
%partition data
n=size(x,1);
m=round(n*0.7);% Train on the first 70% and make predictions on the last 30%     
XTrain=x(1:m,:)';
XTest=x(m+1:end,:)';
YTrain=y(1:m,:)';
YTest=y(m+1:end,:)';

numFeatures = size(XTrain,1);
numResponses = 1;
numHiddenUnits1 = 200;    
layers = [ ...
    sequenceInputLayer(numFeatures)
    
    lstmLayer(numHiddenUnits1)
 %   lstmLayer(numHiddenUnits2)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0);
net = trainNetwork(XTrain,YTrain,layers,options);
numTimeStepsTest = size(XTest,2);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

% denormalization
pre_value=mapminmax('reverse',YPred,mappingy);
true_value=mapminmax('reverse',YTest,mappingy);
pre_result=[pre_result;pre_value];
true_result=[true_result;true_value];
end
%% Add the predicted results of each component
true_value=sum(true_result);
predict_value=sum(pre_result);
save predicted_result/eemd_lstm predict_value true_value

%%
load predicted_result/eemd_lstm
disp('Result analysis')
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['root mean square error(RMSE)£º',num2str(rmse)])

mae=mean(abs(true_value-predict_value));
disp(['mean absolute error£¨MAE£©£º',num2str(mae)])

mape=mean(abs(true_value-predict_value)./true_value);
disp(['mean relative percent error£¨MAPE£©£º',num2str(mape*100),'%'])

fprintf('\n')
figure
plot(true_value,'-*','linewidth',3)
hold on
plot(predict_value,'-s','linewidth',3)
legend('actual value','predicted value')
grid on
title('EEMD-LSTM')