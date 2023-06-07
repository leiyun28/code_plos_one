function y=fitness(x,p,t,pt,tt)
rng(0)
numFeatures = size(p,1);%Enter the number of nodes
numResponses = size(t,1);%Number of output nodes
miniBatchSize = 16; %Batchsize is consistent with main....m
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
    'Verbose',false);


net = trainNetwork(p,t,layers,options);

YPred = predict(net,pt,'MiniBatchSize',1);YPred=double(YPred);
[m,n]=size(YPred);
YPred=reshape(YPred,[1,n*m]);
tt=reshape(tt,[1,n*m]);

y =mse(YPred-tt);
% Taking mse as the fitness function, the purpose of the optimization algorithm is to find a set of hyperparameters to make the mse of the network the lowest
rng((100*sum(clock)))

