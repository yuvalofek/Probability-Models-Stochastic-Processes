% Yuval Ofek and Jason Kurian
% ECE302 Project 1
%% Epic Question 1 (character creation)
clear all;close all; clc;
% Number of tries to perform
N = 100000;
% a) Probability to get 18 from 3d6
% P(Y=18) = P(X1=6,X2=6,X3=6) 
%         = P(X1=6)*P(X2=6)*P(X3=6)
%         = (1/6)^3
%         = 1/216 =~ 0.0046

% Initialize variables 
%Perform 3 rolls of 6 sided dic and sum the result N times
y = sum (randi(6,3,N));    

%Estimate the probability by taking the number of times y=18 and dividing
%by total # of tries
P_y18 = P_est(y, 18)

%% b) generate 3 scores and keep the highest one to get 18
% We want the probability of one or more of the three "scores" to be 18. So
% we actually want to calculate 1- P(none are 18):
% P(Yf = 18) = 1-P(Y1,Y2,Y3!=18) = 1-P(Y1!=18)P(Y2!=18)P(Y3!=18)
%            = 1-(1-[1/6]^3)^3
%            =~ 0.0138
% Note P(Y!=18) = 1-P(X1=6,X2=6,X3=6) = 1-(1/6)^3

Yf = funroll(3,6,3,N);
P_yfun18 = P_est(Yf, 18)

%%
%%% c) Probability of doing this 6 times out of 6
% P(perfect character) = P(Yf1=18,...,Yf6=18)
%       = P(Yf1=18)*...*P(Yf6=18)
%       = 0.0138^6
%       =~ 6.9218e-12
% This number is extremely small, so doing simulations to converge to it
% will take A LOT of iterations and is simply not worth the time... I only
% do so many iterations, so this might seem to be a zero probability
% through the simulation, but it is in fact non-zero but extremely small.

P_perfect = char_type_check(18,N)

%% 
%%% d) Probability of being average with the fun method
% At this point, we omit the comments that show the probability and simply
% go to the simulation

P_average = char_type_check(9,N)

%% Epic Question 2 (Wizards, Trolls and Warriors)
% a) 
% Average hit points of a troll 
troll_hp_avg = mean(get_roll(1,4,N))
% Average fireball damage
fb_avg = mean(get_roll(2,2,N))
% Probability that fireball does greater than 3 points of damage
P_fb3 = sum(get_roll(2,2,N) > 3)/N

%% b) 
% The probability mass function is obtained in the previous section (for
% the last part)
% From our analysis: PFd = [0.25,0.5,0.25];
SHP = 1:4;
P_HP = findprob(get_roll(1,4,N), SHP)
SFd = 2:4;
PFd = findprob(get_roll(2,2,N), SFd)
% From our analysis: P_HP = ones(4,1)/4;

%%% Plot the pmfs
figure
subplot(2,1,1)
stem(SFd,PFd)
title('pmf of the damage of the fireball')
xlabel('Damage (hit points)')
ylabel('Probability')
xticks(2:4)
xlim([1,5])
ylim([0,1])

subplot(2,1,2)
stem(SHP, P_HP)
title('pmf of troll HP')
xlabel('Health Points')
ylabel('Probability')
xticks(1:4)
xlim([0,5])
ylim([0,1])

%% c,d) 
all_slayed = 0;
last_troll_count = 1;
last_troll_hp = zeros(1,N/10);
last_troll_fb = zeros(1,N/10);
for i=1:N
    fb_dmg = get_roll(2,2,1);
    troll_hp = zeros(1,6);
    trolls_slayed = 0;
    for j = 1:6 % generate hp of 6 trolls
        troll_hp(j) = get_roll(1,4,1);
        if troll_hp(j) <= fb_dmg % troll is slayed
            trolls_slayed = trolls_slayed + 1;
        else
            % save position of living troll
            last_troll_pos = j;
        end
    end
    if trolls_slayed == 6 % event where all trolls were slayed
        all_slayed = all_slayed + 1;
    else
        % hp of last troll given five trolls were slayed
        if trolls_slayed == 5
            % save hp of last troll
            last_troll_hp(last_troll_count) = troll_hp(last_troll_pos);
            % save fireball damage that slayed the 5 trolls
            last_troll_fb(last_troll_count) = fb_dmg;
            % index for vector storing above two values
            last_troll_count = last_troll_count + 1;
        end
    end
end
% Probability that fireball slays all trolls
% P(Slay all 6) = P(Fd=4) + P(Fd=3 & HP_all_trolls<=3) + 
% P(Fd=2 & HP_all_trolls<=2)
%       = 1/4 + 1/2*(3/4)^6 + 1/4*(1/2)^6
%       = 2809/8192 =~ 0.3429
P_all_slayed = all_slayed/N
% Expected hitpoints of lone surviving troll  
last_troll_hp = mean(last_troll_hp-last_troll_fb)

%% e) Expected damage done by Shedjam
dmg = zeros(1,N);
for i = 1:N
    if get_roll(1,20,1) >= 11 % if true, can use Sword
        dmg(i) = get_roll(2,6,1);
        if get_roll(1,20,1) >= 11 % if true, can use Hammer
            dmg(i) = dmg(i) + get_roll(1,4,1);
        end
    end
end
% expected damage done by Shedjam
Shedjam_dmg = mean(dmg)

%% Functions
function Yf = funroll(rollcount, sidenum, iter, N)
% roll a sidenum -sided die rollcount times (rollcount-d-sidenum), and 
% take the maximum out of iter times. Repeat for N times. 
% Perform 3 rolls of 6 sided dice
x = randi(sidenum, rollcount,iter,N);
% Sum the result to get a column vector of the individual score 
% (rollcount-d-sidenum), and then take the maximum score from iter
% scores
Yf =zeros(N,1);
Yf(:) = max(sum (x,2));
end

function Prob_est = P_est (Inp, x_req) 
%takes a vector of simulated data Inp and find the probability that the
%input equals x_req
[N,M] = size(Inp);
% We assume that the input is a column or row vector
Prob_est = sum(Inp == (ones(N,M)*x_req))/max(N,M);  
end

function char_prob = char_type_check( ab_score, N)
%counts the number of characters generated by the fun method that have all
%their ability scores equal to ab_score, in N trials, and calculates the
%probability of the character apearing
for i = 1:N
    Yf = funroll(3,6,3,6);
    %Char_count- checks each cell if it is 18 and then sums them up to see if
    %they are all 18 (in which case the sum should be 6)
    Char_count(i) = sum((Yf == ones(6,1)*ab_score))==6; 
    %For storage purposes, we store the result, if the 6 fun-rolls were
    %perfect, rather than the actual roll values. (1/6 the data to store);
end
char_prob = sum(Char_count)/N;
end

function roll = get_roll(rollcount,sidenum,N)
    roll = randi([1,sidenum],rollcount,N);
    % sum columns of roll matrix to add dice values
    roll = sum(roll,1);
end

function P = findprob(x,Sx)
%Find the probability of all the elements of Sx 
%Get important sizes
N = size(x,2);
SxSize = size(Sx,2);
P = zeros(1,SxSize);
for i = 1:SxSize
    P(i) = sum(x == ones(1,N)*Sx(i))/N;
end
end