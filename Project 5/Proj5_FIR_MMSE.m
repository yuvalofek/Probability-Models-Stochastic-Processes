%Stoch Final Assignemnt
% Yuval ,Jason, Theo

%%
clear all; close all;clc;

% We decided to still manually calculate the Rrr and Rsr from the signal
% because we believe it shows a better example of how this type of problem
% will be seen "in the wild". In general, we generated the signal and used
% it to calculate Rrr and Rsr, from which we derive filter coefficients
% and use those coefficients to approximate the input. 

%%% Parameters
var = 1;
N = [1,4,6,10,100];
c = [1, 0.2, 0.4];
Niter = 1e6;

%%% generate s, d, & r
 s = rand(Niter,1);
 s = (s<0.5)-(s>0.5);
 d = sqrt(var)*randn(Niter, 1);
 r = d+filter(c,1,s);
 
%%% Use data to approximate Rrr and Rsr
Rsr = zeros(max(N),1);
for ii = 1:max(N)
    rrr(ii) = mean(r(ii:end).*r(1:end+1-ii));
    Rsr(ii) = mean(s(ii:end).*r(1:end+1-ii));
end
Rrr = toeplitz(rrr);

%%% Solve for h coefficients and use them them to approximate s
h = zeros(max(N),length(N));
for ii = 1:length(N)
    h(1:N(ii),ii) = Rrr(1:N(ii),1:N(ii))\Rsr(1:N(ii));
    s_hat(:,ii) = filter(h(1:N(ii),ii),1,r);
end
% Get MSE without the filters and with filter
MSE_nofilt = mean((s-r).^2).' %Print the MSE without the filter
MSE = mean((s-s_hat).^2).';

%Tabulate
t = table(MSE, 'RowNames', {'N=1', 'N=4', 'N=6', 'N=10', 'N=100'})