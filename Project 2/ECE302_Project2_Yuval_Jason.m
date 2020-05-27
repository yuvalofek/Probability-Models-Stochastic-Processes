%%% Jason Kurian & Yuval Epstain Ofek
%%% Stoch Proj 2 - 2020;
%% Scenario 1
% Implement the Bayes MMSE and Linear MMSE estimators from examples 8.5 and
% 8.6. Simulate this system by random draws of Y and W, and then estimating
% Y from the observations X = Y + W. Verify that your simulation is correct
% by comparing theoretical and empirical values of the MSE. Report your 
% results in a table.

clear all; close all; clc;
%%% Estimators: (since both are one liners we left them as anon functions)
%Linear estimator x/5
L_Est = @(x) x/5;  
%Bayes estimator - 
%       (1+x)/2     if x<-1
%       0           if -1<=x<1
%       (-1+x)/2    if 1<=x 
B_Est = @(x) (1+x)/2.*(x<-1)+ 0.*and(x>=-1,x<1) + (-1+x)/2.*(x>=1); 

%Noting down the theoretical MSEs
BMMSE = 1/4;
LMMSE = 4/15;

%Generating Random Vectors
N = 100000000;
W = -2+4*rand(N,1,'double');
Y = -1+2*rand(N,1,'double');
X = Y+W;

%Bayes
yhat = B_Est(X);
BMMSE(2) = mean((Y-yhat).^2);

%linear
Yhat = L_Est(X);
LMMSE(2) = mean((Y-Yhat).^2);


%generate table
t1 = table(BMMSE(:),LMMSE(:),'RowNames', {'Theoretical'; 'Our_Results'}, 'VariableNames', ...
    {'Bayes_MMSE', 'Linear_MMSE'})

%%% We see that our experiment and the theory matched.

%% Scenario 2

% Implement the linear estimator for multiple noisy observations, similar 
% to example 8.8 from the notes. Extend this example so that it works for 
% an arbitrary number of observations. Use Gaussian random variables for Y 
% and R. Set ?y = 1. Experiment with a few different variances for both Y 
% and R. On one plot, show the mean squared error of your simulation 
% compared to the theoretical values for at least 2 different pairs of 
% variances.

clear all;close all;clc;
%%% Implement linear estimator for noisy observations

N = 10000;
Nobs = 20;
NobsVect = 2:Nobs; 

%parameters
ymean = 1;
Y_var = [1, 0.5, 0.1];
R_var = [1, 0.5, 0.1];
Y_sdv = sqrt(Y_var);
R_sdv = sqrt(R_var);

%how many different variances did I specify
Nvars = min(size(Y_sdv,2),size(R_sdv,2));
% MMSE = zeros(size(NobsVect,2), 2,Nvars);
%starting figure here, so I can use the same loop for plotting
figure
hold on
for ii = 1:Nvars
    %run the experiment for the number of different variances specified
    [MMSE(:,:,ii)] = runExp(ymean, Y_sdv(ii), R_sdv(ii),NobsVect,N);
    
    %plotting
    plot(NobsVect, MMSE(NobsVect(1):end,1,ii), 'DisplayName', ['Experiment- \sigma^2_Y =',num2str(Y_sdv(ii)^2), ', \sigma^2_R =',num2str(R_sdv(ii)^2)],'LineWidth', 2)
    plot(NobsVect, MMSE(NobsVect(1):end,2,ii), '--', 'DisplayName', ['Theoretical- \sigma^2_Y =',num2str(Y_sdv(ii)^2), ', \sigma^2_R =',num2str(R_sdv(ii)^2)],'LineWidth', 2)
end
title('Implementing LMMSE for 8.8, with generalized number of observations')
xlabel('Number of observations')
ylabel('MMSE')
legend 
hold off

%%% Summary of Results
% We calculated the theoretical MMSE using eqn. 8.80 scaled for N
% observations, and we used the estimator in eqn. 8.79 to calculate the
% simulated MSSE. The simulated MMSE curve closely follows the theoretical
% curve for a generalized number of observations, as shown in Figure 1.
%
% In general, the number of observations are inversely related to the
% MMSE, while the variance of the random variable Y and the noise is
% directly related to the MMSE. As the number of observations of X increase, 
% the MMSE of the estimator correspondingly decreases. A higher variance in 
% Y or the noise produces a higher MMSE, and a lower variance decrease the 
% MMSE.

%% Functions
function [Yest] = Y_hat(X,Y_sdv,ymean, R_sdv, Nobs)
%finds Yhat from the values of R and Y, given Nobs observations
Yest = 1/(Nobs*Y_sdv^2+R_sdv^2)*(R_sdv^2*ymean + sum(X)*Y_sdv^2);
end

function [MMSE] = runExp(ymean, Y_sdv, R_sdv,NobsVect,N)
%running the experiment for the number of observations in NobsVect and the
%parameters for Y and R provided, with N different elements. 

%Maximum value in NobsVect:
mxObs = max(NobsVect);
%Create random R and Y vectors
Rs = R_sdv*randn(mxObs,N);
Y = ymean + Y_sdv*randn(1,N);
%Make the X vectors
X = ones(mxObs, 1)*Y+Rs;
%loop over the number of observations provided
for jj = NobsVect
    %Estimate y based on the X matrix
    Yest2 = Y_hat(X(1:jj,:),Y_sdv, ymean,R_sdv, jj);
    %Generate an MMSE matrix where the first column is the theoretical and
    %the 2nd is the experimental results. 
    MMSE(jj,1) = Y_sdv^2*R_sdv^2/(jj*Y_sdv^2+R_sdv^2);
    MMSE(jj, 2) = mean((Y-Yest2).^2);
end
end