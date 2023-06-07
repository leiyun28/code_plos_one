tic;
clc;clear;format compact;close all;tic
rng('default')
%% data preprocessing
load ceemd_data
imf=u;
c=size(imf,1);
pre_result=[];
true_result=[];
%% Model each component
for i=2:c
disp(['Model the',num2str(i),'component'])
[x,y]=data_process(imf(i,:),12);
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
    [x ,fit_gen,process]=psoforlstm(XTrain,YTrain,XTest,YTest);%Optimize the training times and learning rate of hidden layer nodes respectively
    save result/pso_para_result x fit_gen process
else
    load result/pso_para_result
end
%% Draw the fitness curve and the change curve of the 4 parameters
huatu(fit_gen,process,'PSO')
disp('优化的超参数为：')
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



numTimeStepsTest = size(XTest,2);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

%denormalization
pre_value=mapminmax('reverse',YPred,mappingy);
true_value=mapminmax('reverse',YTest,mappingy);
pre_result=[pre_result;pre_value];
true_result=[true_result;true_value];
end
%% Add the predicted results of each component
true_value=sum(true_result);
predict_value=sum(pre_result);
save predicted_result/ceemd_pso_lstm predict_value true_value

%%
load predicted_result/ceemd_pso_lstm
disp('Result analysis')
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['root mean square error(RMSE)：',num2str(rmse)])

mae=mean(abs(true_value-predict_value));
disp(['mean absolute error（MAE）：',num2str(mae)])

mape=mean(abs(true_value-predict_value)./true_value);
disp(['mean relative percent error（MAPE）：',num2str(mape*100),'%'])
r2=1-(sum((true_value-predict_value).^2)/sum((true_value-mean(true_value)).^2));
fprintf('\n')
figure
plot(true_value,'-*','linewidth',3)
hold on
plot(predict_value,'-s','linewidth',3)
legend('actual value','predicted value')
grid on
title('CEEMD-PSO-LSTM')
toc;