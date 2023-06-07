clear;
close all;
clc;
%% parameter settings

%% data loading
f=xlsread('data.xlsx','E2:E721');

% u=emd(f);%emd
% % If eemd ceemd is used, there are still parameters to be set
Nstd = 0.02;
NE=50;
TNM =8;
% [u ,~]=eemd(f,Nstd,NR,MaxIter);%eemd
u=ceemd(f,Nstd,NE,TNM);%ceemd
u=u';
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
save ceemd_data u