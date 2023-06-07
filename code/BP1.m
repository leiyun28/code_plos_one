clc
clear
close all
%%
%read data
data=xlsread('data.xlsx','E2:E721');
[shuru,shuchu]=data_process(data,12);
m=size(shuru,1);
geshu=round(m*0.7);%The number of samples in the training set
nn=1:size(shuru,1);%normal sort
input_train =shuru(nn(1:geshu),:);input_train=input_train';
output_train=shuchu(nn(1:geshu),:);output_train=output_train';
input_test =shuru(nn((geshu+1):end),:);input_test=input_test';
output_test=shuchu(nn((geshu+1):end),:);output_test=output_test';
%%
inputnum=size(input_train,1);%the number of nodes entered
hiddennum=2*inputnum+1;%the number of nodes in the middle
outputnum=1;%output node count
%Sample input and output data normalization
[aa,bb]=mapminmax([input_train input_test]);
[cc,dd]=mapminmax([output_train output_test]);
[inputn,inputps]=mapminmax('apply',input_train,bb);
[outputn,outputps]=mapminmax('apply',output_train,dd);
%%
%build network
% net=newff(inputn,outputn,hiddennum,{'tansig','purelin'},'traingdm');%1
net=newff(inputn,outputn,hiddennum);%2
%Set training parameters
net.trainParam.lr=0.1; %Learning rate 0.05% adaptive LR variable step size
net.trainParam.show = 50;%Display the training results every 50 steps
net.trainParam.mc = 0.9;%add momentum
net.trainParam.goal = 1e-5;      %The mse root mean square error is less than this value and the training ends
net.trainParam.min_grad = 1e-5;
net.trainParam.epochs = 100;   %iterations
net.trainParam.max_fail=10;   

%%
[net,~]=train(net,inputn,outputn);
%Test set sample input and output data normalization
inputn_test=mapminmax('apply',input_test,bb);
%%
an=sim(net,inputn_test);
test_simu1=mapminmax('reverse',an,dd);%denormalization
error1=test_simu1-output_test;
save predicted_result/bp test_simu1 output_test error1
%%
load predicted_result/bp
figure
plot(output_test','r:*');hold on;
plot(test_simu1','b-o');hold on;
legend('original data','predictive data')
disp('-------predictive data-------')
disp([output_test' test_simu1'])
disp('Relative error')
disp([abs(error1./output_test)]')
figure
plot(error1);hold on;
legend('error')
disp('Evaluation index based on prediction error')

%% 
T_sim_optimized = test_simu1;  %simulation data

num=size(output_test,2);%Total number of statistical samples
error=T_sim_optimized-output_test;  %Calculation error
errorPercent=abs(error./output_test); %Compute the absolute percent error for each sample

sse=sum(error.*error);   %Calculate the sum of squared errors
mae=sum(abs(error))/num; %Calculate mean absolute error
me=sum((error))/num; %Calculate mean absolute error
mse=sum(error.*error)/num;  %Calculate mean squared error
rmse=sqrt(mse);     %Calculate root mean squared error
mape=mean(errorPercent);  %Calculate the mean absolute percentage error
fangcha=var(error,1);
r=min(min(corrcoef(T_sim_optimized,output_test)));
disp(r)
% R2=r*r;
tn_sim = T_sim_optimized';
tn_test =output_test';
N = size(tn_test,1);
R2=(N*sum(tn_sim.*tn_test)-sum(tn_sim)*sum(tn_test))^2/((N*sum((tn_sim).^2)-(sum(tn_sim))^2)*(N*sum((tn_test).^2)-(sum(tn_test))^2)); 

disp(' ')
disp('----------------------------------------------------------')
%disp(['The sum of squared errors sse£º              ',num2str(sse)])
disp(['mae£º            ',num2str(mae)])
%disp(['me£º            ',num2str(me)])
%disp(['mse£º                ',num2str(mse)])
disp(['rmse£º             ',num2str(rmse)])
disp(['mape£º     ',num2str(mape*100),' %'])
disp(['R2£º                ' ,num2str(R2)])
%disp(['r£º                 ',num2str(r)])
%disp(['error variance£º                 ',num2str(fangcha)])
%bp2=test_simu1;
%save('bp2','bp2')

%yuanshi=output_test;
%save('yuanshi','yuanshi')



