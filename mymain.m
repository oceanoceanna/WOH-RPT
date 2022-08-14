nbits_set=[16 32 64 96];

%% load dataset
fprintf('loading dataset...\n')
set = 'MIRFlickr';
load('MIRFlickr-data.mat');
XTrain = XTrain(1:16000,:);
YTrain = YTrain(1:16000,:);
anchor=XTrain(randsample(2000,1000),:);

%% initialization
fprintf('initializing...\n')
param.alpha = 0.001;
param.beta = 0.1;
param.mu = 10;
param.gama = 0.001;
param.delta = 10;
param.datasets = set;
param.chunk = 2000;
param.lamda = 0.01;
param.paramiter = 10;
param.nq = 200;

%% model training
for bit=1:length(nbits_set)
    nbits=nbits_set(bit);
    Binit = sign(randn(16000, nbits));
    Vinit = randn(16000, nbits);
    Pinit = randn(1000, nbits);
    param.nbits=nbits;
    MAP = train(XTrain,YTrain,param,LTrain,XTest,LTest,anchor,Binit,Vinit,Pinit);
end 


