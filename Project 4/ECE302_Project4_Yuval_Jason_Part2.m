%%% Stoch Proj 4
% Yuval Epstain Ofek & Jason Kurian
%% Part 2
clear all;close all;clc

%loading the data
load( 'Iris.mat');
%shuffling data randomly
data = [features labels];
rand_pos = randperm(length(data));
data_shuf = zeros(150,5);
for ii = 1:length(data)
    data_shuf(ii,:) = data(rand_pos(ii),:);
end
% split data 50/50 into training and testing sets
trainset = data_shuf(1:2:end,:);
testset = data_shuf(2:2:end,:);

% split data back into features and labels
trainlabels = trainset(:,5);
trainfeatures = trainset(:,1:4);
testlabels = testset(:,5);
testfeatures = testset(:,1:4);

% implement MAP classifier
mu = zeros(length(testset),4);
var = zeros(4,4,length(testset));
likelihoods = zeros(length(testset),3);
% for each label in the training set, find the sample mean vector and 
% covariance matrices
for ii = 1:3
    mu(ii,:) = mean(trainset(trainlabels==ii,1:4));
    var(:,:,ii) = cov(trainset(trainlabels==ii,1:4));
    % evaluate likelihood of test features for each label
    likelihoods(:,ii) = mvnpdf(testfeatures,mu(ii,:),var(:,:,ii));
end
% for the confusion function, create the targets matrix where each index of 
% 1 indicates which of the test labels is represented
targets = [1:length(testlabels);testlabels';ones(1,length(testlabels))]';
targets = full(spconvert(targets))';
[C,CM,~,~] = confusion(targets,likelihoods')