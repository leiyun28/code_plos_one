clc;
clear 
close all;
%% II. Import Data 
data=xlsread('data.xlsx','C1:C2074');
[attributes,strength]=data_process(data,12);
attributes=attributes';strength=strength';

%%
% n = randperm(size(attributes,2));%Randomly selected
n=1:size(attributes,2);%sequential selection
m=size(attributes,2);
%%
% 2. Training set
xungeshu=round(m*0.7);%The number of samples in the training set
p_train = attributes(:,n(1:xungeshu))';
t_train = strength(:,n(1:xungeshu))';                
%%
% 3. test set
p_test = attributes(:,n((xungeshu+1):end))';
t_test = strength(:,n((xungeshu+1):end))';
%% III. data normalization
%%
% 1. Training set
[pn_train,inputps] = mapminmax(p_train');
pn_train = pn_train';
pn_test = mapminmax('apply',p_test',inputps);
pn_test = pn_test';
%%
% 2.test set
[tn_train,outputps] = mapminmax(t_train');
tn_train = tn_train';
tn_test = mapminmax('apply',t_test',outputps);
tn_test = tn_test';
%% IV. SVM model creation/training
%%
% 1. Finding the best c-parameters/g-parameters
[c,g] = meshgrid(-8:1:8,-8:1:8);%step size is 1
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 0;
bestg = 0;
error = Inf;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];
        cg(i,j) = svmtrain(tn_train,pn_train,cmd);
        if cg(i,j) < error
            error = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)
            error = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    end
end
%%
% 2. Create/train SVM
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.1'];
model = svmtrain(tn_train,pn_train,cmd);
%% V. SVM simulation prediction
Predict_2 = svmpredict(tn_test,pn_test,model);
%%
% 1. denormalization
predict_2 = mapminmax('reverse',Predict_2,outputps);
save predicted_result/svm predict_2 t_test
%% VI. plot
load predicted_result/svm
figure
plot(1:length(t_test),t_test,'r-*',1:length(t_test),predict_2,'b-o')
grid on
legend('actual value','predicted value')
disp([t_test predict_2])
disp('optimal parameters£¬c£¬g')
disp([bestc bestg])
%% Error Analysis
output_test = t_test'; % Real data, the number of rows represents the number of features, and the number of columns represents the number of samples
T_sim_optimized = predict_2';  % simulation data

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
%disp(['sse£º              ',num2str(sse)])
disp(['mae£º            ',num2str(mae)])
%disp(['me£º            ',num2str(me)])
%disp(['mse£º                ',num2str(mse)])
disp(['rmse£º             ',num2str(rmse)])
disp(['mape£º     ',num2str(mape*100),' %'])
disp(['R2£º                ' ,num2str(R2)])
%disp(['r£º                 ',num2str(r)])
%disp(['error variance£º                 ',num2str(fangcha)])

%svm2 = predict_2';
%save('svm2','svm2')
