clear;
close all;
clc;
%% parameter settings

%% data loading
f=xlsread('data.xlsx','E2:E721');
%u=emd(f);%emd
% % If eemd ceemd is used, there are still parameters to be set. The most
% influential factors on the final decomposition result are the standard deviation Nstd of white noise and the total mean NR.
Nstd = 0.02;%It is used to set the standard deviation of the added Gaussian white noise to eliminate the noise in the original signal, and Nstd is the standard deviation of the added random white noise
NR = 1;
MaxIter =200;%Maximum number of iterations during EMD decomposition
[u ,~]=eemd(f,Nstd,NR,MaxIter);%eemd
% [u ,~]=ceemd(f,Nstd,NR,MaxIter);%ceemd
%u=emd(f);%emd
% % If eemd ceemd is used, there are still parameters to be set,eemd(Y,Nstd,NE)
K=size(u,1);
figure
subplot(K+1,1,1)
plot(f)
ylabel('original')

for i=2:K+1
    subplot(K+1,1,i)
    plot(u(i-1,:))
    ylabel(['IMF',num2str(i-1)])
end
ylabel('res')
save eemd_data u


