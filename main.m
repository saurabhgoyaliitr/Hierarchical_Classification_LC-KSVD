% =========================================================================
% An example code for the algorithm proposed in
%
%   Zhuolin Jiang, Zhe Lin, Larry S. Davis.
%   "Learning A Discriminative Dictionary for Sparse Coding via Label 
%    Consistent K-SVD", CVPR 2011.
%
% Author: Zhuolin Jiang (zhuolin@umiacs.umd.edu)
% Date: 10-16-2011
% =========================================================================

function main(inputf,sparsitythres,sqrt_alpha,sqrt_beta,iterations4ini)
clc;
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP
load(strcat('.\trainingdata\features_',inputf,'.mat'),'training_feats','testing_feats', 'H_train','H_test');

%% constant
sparsitythres = str2num(sparsitythres); % sparsity prior
sqrt_alpha = str2num(sqrt_alpha); % weights for label constraint term
sqrt_beta = str2num(sqrt_beta); % weights for classification err term
dictsize = 22; % dictionary size
iterations = 100; % iteration number
iterations4ini = str2num(iterations4ini); % iteration number for initialization

%% dictionary learning process
% get initial dictionary Dinit and Winit
fprintf('\nLC-KSVD initialization... ');
[Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(training_feats,H_train,dictsize,iterations4ini,sparsitythres);
fprintf('done!');

% run LC K-SVD Training (reconstruction err + class penalty)
fprintf('\nDictionary learning by LC-KSVD1...');
[D1,X1,T1,W1] = labelconsistentksvd1(training_feats,Dinit,Q_train,Tinit,H_train,iterations,sparsitythres,sqrt_alpha);
save(strcat('.\trainingdata\dictionarydata1_',inputf,'.mat'),'D1','X1','W1','T1');
fprintf('done!');

% run LC k-svd training (reconstruction err + class penalty + classifier err)
fprintf('\nDictionary and classifier learning by LC-KSVD2...')
[D2,X2,T2,W2] = labelconsistentksvd2(training_feats,Dinit,Q_train,Tinit,H_train,Winit,iterations,sparsitythres,sqrt_alpha,sqrt_beta);
save(strcat('.\trainingdata\dictionarydata2_',inputf,'.mat'),'D2','X2','W2','T2');
fprintf('done!');

%% classification process
%% classification process
[prediction1,accuracy1] = classification(D1, W1, testing_feats, H_test, sparsitythres);
fprintf('\nFinal recognition rate for LC-KSVD1 is : %.03f ', accuracy1);

[prediction2,accuracy2] = classification(D2, W2, testing_feats, H_test, sparsitythres);
fprintf('\nFinal recognition rate for LC-KSVD2 is : %.03f ', accuracy2);
save(strcat('.\trainingdata\prediction_',inputf,'.mat'),'prediction1','prediction2');
