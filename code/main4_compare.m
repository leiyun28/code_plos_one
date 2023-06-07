clc;clear;close all
%% 
lstm=load('predicted_result/lstm.mat');%1
ann=load('predicted_result/ann');%2
svm=load('predicted_result/svm');
bp=load('predicted_result/bp');
emd_lstm=load('predicted_result/emd_lstm');
eemd_lstm=load('predicted_result/eemd_lstm');
ceemd_lstm=load('predicted_result/ceemd_lstm');
pso_lstm=load('predicted_result/PSOlstm');
ceemd_pso_lstm=load('predicted_result/ceemd_pso_lstm');
%% 1
disp('1--lstm')
result(lstm.true_value,lstm.predict_value)
fprintf('\n')
%% 2
disp('2--ann')
result(ann.true_value,ann.predict_value)
fprintf('\n')

%% 3
disp('3--svm')
result(svm.t_test,svm.predict_2)
fprintf('\n')

%% 4
disp('4--bp')
result(bp.output_test,bp.test_simu1)
fprintf('\n')

%% 5
disp('5--emd_lstm')
result(emd_lstm.true_value,emd_lstm.predict_value)
fprintf('\n')

%% 6
disp('6--eemd_lstm')
result(eemd_lstm.true_value,eemd_lstm.predict_value)
fprintf('\n')

%% 7
disp('7--ceemd_lstm')
result(ceemd_lstm.true_value,ceemd_lstm.predict_value)
fprintf('\n')

%% 8
disp('8--pso_lstm')
result(pso_lstm.true_value,pso_lstm.predict_value)
fprintf('\n')

%% 9
disp('9--ceemd_pso_lstm')
result(ceemd_pso_lstm.true_value,ceemd_pso_lstm.predict_value)
fprintf('\n')
%% plot
figure
plot(lstm.true_value,'linewidth',2)
hold on;grid on
%plot(lstm.predict_value,'linewidth',2)
%plot(ann.predict_value,'linewidth',2)
%plot(svm.predict_2,'linewidth',2)
%plot(bp.test_simu1,'linewidth',2)
%plot(emd_lstm.predict_value,'linewidth',2)
%plot(eemd_lstm.predict_value,'linewidth',2)
plot(ceemd_lstm.predict_value,'linewidth',2)
plot(pso_lstm.predict_value,'linewidth',2)
plot(ceemd_pso_lstm.predict_value,'linewidth',2)
%legend('true value','lstm','ann','svm','bp','Interpreter','latex','FontSize',10)
%legend('true value','emd lstm','eemd lstm','ceemd lstm','Interpreter','latex','FontSize',10)
legend('true value','ceemd lstm','pso lstm','ceemd pso lstm','Interpreter','latex','FontSize',10)
%legend('true value','lstm','ann','svm','bp','emd lstm','eemd lstm','ceemd lstm','pso lstm','ceemd pso lstm','Interpreter','latex','FontSize',10)%,'pso lstm','ceemd pso lstm','ceemd lstm','Interpreter','latex','FontSize',10)
ylabel('value','Interpreter','latex','FontSize',15)
xlabel('Test set samples','Interpreter','latex','FontSize',15)