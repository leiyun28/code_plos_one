%% PSO-optimized LSTM time series forecasting
clc;clear;close all;format compact
%%
data=xlsread('data.xlsx','E2:E721');
[x,y]=data_process(data,12);
%Normalization
[xs,mappingx]=mapminmax(x',0,1);x=xs';
[ys,mappingy]=mapminmax(y',0,1);y=ys';
%partition data
n=size(x,1);
m=round(n*0.7);%Train on the first 70% and make predictions on the last 30%
XTrain=x(1:m,:)';
XTest=x(m+1:end,:)';
YTrain=y(1:m,:)';
YTest=y(m+1:end,:)';

%% Using PSO optimization
optimization=1;%whether to re-optimize
if optimization==1
    [x ,fit_gen,process]=psoforlstm(XTrain,YTrain,XTest,YTest);%ptimize the training times and learning rate of hidden layer nodes respectively
    save result/pso_para_result x fit_gen process
else
    load result/pso_para_result
end
%% Draw the fitness curve and the change curve of the 4 parameters
huatu(fit_gen,process,'PSO')
disp('The optimized hyperparameters are��')
disp('L1:'),x(1)
disp('L2:'),x(2)
disp('K:'),x(3)
disp('lr:'),x(4)

%% Retrain with optimized parameters
train=1;%whether to retrain
if train==1
    rng(0)
    numFeatures = size(XTrain,1);
    numResponses = size(YTrain,1);
    miniBatchSize = 16; %batchsize is consistent with fitness.m
    numHiddenUnits1 = x(1);
    numHiddenUnits2 = x(2);
    maxEpochs=x(3);
    learning_rate=x(4);
    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits1)
        lstmLayer(numHiddenUnits2)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    options = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'InitialLearnRate',learning_rate, ...
        'GradientThreshold',1, ...
        'Shuffle','every-epoch', ...
        'Verbose',true,...
        'Plots','training-progress');

    net = trainNetwork(XTrain,YTrain,layers,options);
    save model/psolstm net
else
    load model/psolstm
end
% Ԥ��
YPred = predict(net,XTest,'MiniBatchSize',1);YPred=double(YPred);
% ����һ��
predict_value=mapminmax('reverse',YPred,mappingy);
true_value=mapminmax('reverse',YTest,mappingy);
save predicted_result/PSOlstm predict_value true_value
%%
load predicted_result/PSOlstm
disp('Result analysis')
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['root mean square error(RMSE)��',num2str(rmse)])

mae=mean(abs(true_value-predict_value));
disp(['mean absolute error��MAE����',num2str(mae)])

mape=mean(abs(true_value-predict_value)/true_value);
disp(['mean relative percent error��MAPE����',num2str(mape*100),'%'])

fprintf('\n')


figure
plot(true_value,'-*','linewidth',3)
hold on
plot(predict_value,'-s','linewidth',3)
legend('actual value','predicted value')
grid on
title('PSO-LSTM')
