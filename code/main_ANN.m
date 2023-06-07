clear
close  all
%% raw data preprocessing
load emd_data
imf=u;
c=size(imf,1);
%% svm regression prediction - data preprocessing
data=sum(u);%All sequences add up
[x,y]=data_process(data,12);%The previous 12 moments predict the next moment
%Normalization
[xs,mappingx]=mapminmax(x',0,1);x=xs';
[ys,mappingy]=mapminmax(y',0,1);y=ys';
%partition data
n=size(x,1);
m=round(n*0.7);%Train on the first 70% and make predictions on the last 30%
XTrain=x(1:m,:);
XTest=x(m+1:end,:);
YTrain=y(1:m,:);
YTest=y(m+1:end,:);
%% training
%Obtain a model using matlab regression learner
%Export the function trainRegressionModel_ANN.m for later model prediction
[trainedModel, validationRMSE] = trainRegressionModel_ANN(XTrain, YTrain);
%% test set prediction
numTimeStepsTest = size(XTest,1);
yfit1=[];
for i = 1:numTimeStepsTest
    yfit = trainedModel.predictFcn(XTest(i,:));
    yfit1=[yfit1;yfit];
end
%% test
% denormalization
predict_value=mapminmax('reverse',yfit1',mappingy);
true_value=mapminmax('reverse',YTest',mappingy);
save predicted_result/ann predict_value true_value
%%
load predicted_result/ann
disp('Result analysis')
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['root mean square error(RMSE)：',num2str(rmse)])

mae=mean(abs(true_value-predict_value));
disp(['mean absolute error（MAE）：',num2str(mae)])

mape=mean(abs(true_value-predict_value)/true_value);
disp(['mean relative percent error（MAPE）：',num2str(mape*100),'%'])

fprintf('\n')

figure
plot(true_value,'-*','linewidth',1)
hold on
plot(predict_value,'-s','linewidth',1)
ylabel('Wind power output(MW)','fontname','times new roman','fontsize',12)
legend('Actual value','Predictive value','fontname','times new roman','fontsize',12)