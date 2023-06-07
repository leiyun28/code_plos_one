function huatu(fitness,process,type)
figure
plot(fitness)
grid on
title([type,'fitness curve'])
xlabel('iterations/time')
ylabel('Fitness value/MSE')

figure
subplot(2,2,1)
plot(process(:,1))
grid on
xlabel('iterations/time')
ylabel('L1/unit')

subplot(2,2,2)
plot(process(:,2))
grid on
xlabel('iterations/time')
ylabel('L1/unit')

subplot(2,2,3)
plot(process(:,3))
grid on
xlabel('iterations/time')
ylabel('K/time')

subplot(2,2,4)
plot(process(:,4))
grid on
xlabel('iterations/time')
ylabel('lr')
% suptitle([type,'Variation of hyperparameters with the number of iterations'])