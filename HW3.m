%% QUESTION 1 

clear all: close all; clc
D=load("icevelocity.txt")
Z=D(:,1);
V=D(:,2);
degree=[0 1 2 3 4];
for n=1:length(degree)
    P=polyfit(Z,V,degree(n)); %fit model 
    vmod(:,n)=polyval(P,Z); %evaluate model
    RMSE(:,n)=sqrt((mean(vmod(:,n)-V).^2)); %model accuracy
end 
plot(Z,V);%,'ko','linewidth',2,'MarkerSize',10)
hold on;
plot(Z,vmod(:,1),'linewidth',2);% Zero degree
plot(Z,vmod(:,2),'linewidth',2);% First degree
plot(Z,vmod(:,3),'linewidth',2);% Second degree
plot(Z,vmod(:,4),'linewidth',2);% Third degree
plot(Z,vmod(:,5),'linewidth',2);% Fourth degree
display(RMSE)
legend({'Raw Data','Zero (0.1249)','First (0.0468)','Second (0.0648)','Third (0.0586)','Fourth (0.0047)'})

%% QUESTION 2

clear all: close all; clc
D=load("icevelocity.txt");

percent=round(91*.9); % Number of rows to sample

degree=[0 1 2 3 4];

for k=1:1000
    for n=1:length(degree)
        r=randsample(91,percent,false); % randomly select .9 rows of 91
        Dsampled=D(r,:); % what are the depths and velocities of the 82 random rows we found
        Z=Dsampled(:,1);
        V=Dsampled(:,2);
        P=polyfit(Z,V,degree(n));
        report(k, n) = P(1); % Store the coefficients      
    end
end

%create a table for mean and std of the 1000 different degrees 
Zero = [nanmean(report(:,1));nanstd(report(:,1))];
First = [nanmean(report(:,2));nanstd(report(:,2))];
Second = [nanmean(report(:,3));nanstd(report(:,3))];
Third = [nanmean(report(:,4));nanstd(report(:,4))];
Fourth = [nanmean(report(:,5));nanstd(report(:,5))];

Parameters = table(Zero,First,Second,Third,Fourth)


%% QUESTION 3 

clear all; close all; clc;

D= load('icevelocity.txt')

pTrain=0.9

for k=1:5
    for n=1:1000
        [trainset,testset]=getTrainTest(D,pTrain);
        ztrain=trainset(:,1);
        vtrain=trainset(:,2);
        P=polyfit(ztrain,vtrain,k);%fit training data with a linear model
        ztest=testset(:,1);
        vtest=testset(:,2);
        vmodel=polyval(P,ztest); %evaluate on test data
        rmsCV(n, k) = sqrt(mean((vmodel - vtest).^2));
    end
end

subplot(1,5,1)
hist(rmsCV(:,1))
subplot(1,5,2)
hist(rmsCV(:,2))
subplot(1,5,3)
hist(rmsCV(:,3))
subplot(1,5,4)
hist(rmsCV(:,4))
subplot(1,5,5)
hist(rmsCV(:,5))

%% QUESTION 4 

clear all; close all; clc;

D= load('icevelocity.txt');

range=1:91;
range = range.';

for n= 1:length(range) % moving window average 
    f3 = mean(D(D(:,1) >= (n - 1) & D(:,1) <= (n + 1), 2)); % mean of all rows column 1 at interval(n) plus or minus 6(aka 3) of column 2
    f10 = mean(D(D(:,1) >= (n - 5) & D(:,1) <= (n + 5), 2));
    f50 = mean(D(D(:,1) >= (n - 25) & D(:,1) <= (n + 25), 2));
    result(n, :) = [n, f3, f10, f50];
end

plot(result(:,1),result(:,2),'linewidth',2);
hold on
plot(result(:,1),result(:,3),'linewidth',2);
hold on
plot(result(:,1),result(:,4),'linewidth',2);
legend({'3','10','50'})

%% QUESTION 5
clear all; close all; clc;

D= load('icevelocity.txt');

range=1:91;
range = range.';

%weights3 = linspace(1, 0, 3); => This creates a variable that spans 1-0 in 7 values of even spacing
%weights3 = weights3 / sum(weights3); => next we need to normalize the weights
%weights3 = weights3.'; => this makes it so that the weights are all in 1 column and not all in 1 row

for n= 1:length(range) % moving window average 
    f3 = mean(D(D(:,1) >= (n - 1) & D(:,1) <= (n + 1), 2)); % mean of all rows column 1 at interval(n) plus or minus 6(aka 3) of column 2
    for k= 1:length(f3)
        weights3 = linspace(1, .1, k); 
        weights3 = weights3 / sum(weights3); 
        weights3 = weights3.';
        result(n,:)= nanmean(f3(:,1).*weights3(:,1));
    end
end

for n= 1:length(range) % moving window average 
    f10 = mean(D(D(:,1) >= (n - 5) & D(:,1) <= (n + 5), 2));
    for k10= 1:length(f10)
        weights10 = linspace(1, .1, k10); 
        weights10 = weights10 / sum(weights10); 
        weights10 = weights10.';
        result10(n,:)= nanmean(f10(:,1).*weights10(:,1));
    end
end

for n= 1:length(range) % moving window average 
     f50 = mean(D(D(:,1) >= (n - 25) & D(:,1) <= (n + 25), 2));
    for k50= 1:length(f50)
        weights50 = linspace(1, .1, k50); 
        weights50 = weights50 / sum(weights50); 
        weights50 = weights50.';
        result50(n,:)= nanmean(f50(:,1).*weights50(:,1));
    end
end

r= 1:1:91;
r=r.';
plot(r,result(:,1),'linewidth',2);
hold on
plot(r,result10(:,1),'linewidth',2);
hold on
plot(r,result50(:,1),'linewidth',2);
legend({'3','10','50'})

%% QUESTION 6

clear all; close all; clc;

D= load('icevelocity.txt');
V=D(:,2);

range=1:91;
range = range.';

for n= 1:length(range) % moving window average 
    f3 = mean(D(D(:,1) >= (n - 1) & D(:,1) <= (n + 1), 2)); % mean of all rows column 1 at interval(n) plus or minus 6(aka 3) of column 2
    for k= 1:length(f3)
        weights3 = linspace(1, .1, k); 
        weights3 = weights3 / sum(weights3); 
        weights3 = weights3.';
        result(n,:)= nanmean(f3(:,1).*weights3(:,1));
    end
end

for n= 1:length(range) % moving window average 
    f10 = mean(D(D(:,1) >= (n - 5) & D(:,1) <= (n + 5), 2));
    for k10= 1:length(f10)
        weights10 = linspace(1, .1, k10); 
        weights10 = weights10 / sum(weights10); 
        weights10 = weights10.';
        result10(n,:)= nanmean(f10(:,1).*weights10(:,1));
    end
end

for n= 1:length(range) % moving window average 
     f50 = mean(D(D(:,1) >= (n - 25) & D(:,1) <= (n + 25), 2));
    for k50= 1:length(f50)
        weights50 = linspace(1, .1, k50); 
        weights50 = weights50 / sum(weights50); 
        weights50 = weights50.';
        result50(n,:)= nanmean(f50(:,1).*weights50(:,1));
    end
end

RMSE(:,1)=sqrt((mean(result(:,1)-V).^2));
RMSE(:,2)=sqrt((mean(result10(:,1)-V).^2));
RMSE(:,3)=sqrt((mean(result50(:,1)-V).^2));

Three = [(RMSE(1,1))];
Ten = [(RMSE(1,2))];
Fifty = [(RMSE(1,3))];

optimum_window_size = table(Three,Ten,Fifty)
 
%% QUESTION 7

clear all; close all; clc;

D = load('icevelocity.txt');
z = D(:,1);
v = D(:,2);

A0range = 20:1:40; % range for A0 from class
N0range = 1:0.25:5; % range for N0, looked it up and it seems like its 3 so we will do 1-5

%runs through values of A and n in the equtation and takes the RMSE for
%each attempted A and n value 
for n= 1:length(A0range)
    A0=A0range(n);
    for f=1:length(N0range)
        N0=N0range(f);
        vmod=(v(1,1))-(A0*(917*9.8*sin(10).^N0)).*(z.^(N0+1)); %equation for ice flow
        RMSE=sqrt((mean(vmod-v).^2));
        RMSE_values_n(n,:) = RMSE;
        RMSE_values_f(f,:) = RMSE;
    end
end


A0range=A0range.';
N0range=N0range.';

OPT_A0=[A0range,RMSE_values_n];
OPT_N0=[N0range,RMSE_values_f];

xOPT_A0=min(OPT_A0)
xOPT_N0=min(OPT_N0)

optimum_A = [(xOPT_A0)];
optimum_n = [(xOPT_N0)];


optimum_values = table(optimum_A,optimum_n)



%% QUESTION 8

clear all; close all; clc;

D = load('icevelocity.txt');
z = D(:,1);
v = D(:,2);

A0range = 20:1:40; % range for A0
N0range = 1:0.25:5; % range for N0


for n= 1:length(A0range)
    A0=A0range(n);
    for f=1:length(N0range)
        N0=N0range(f);
        vmod=(v(1,1))-(A0*(917*9.8*sin(10)).^N0).*(z.^(N0+1));
        RMSE = sqrt(mean(abs(vmod - v) .^ 2)); % must add abs (absolute value) here other wise we get complex numbers and imagec doesn't work
        RMSE_values(n,f) = RMSE;
    end
end
A0range=A0range.';
N0range=N0range.';

imagesc(N0range, A0range, RMSE_values);colorbar


%% QUESTION 9

% when defining 'fun' I thought you could just give it A and n but it seems
% like it needs one varaible, vars in this case, that contains two
% things. To get it to work with two seperate varaibles seems like it would
% require more set up that I couldn't figure out
%vars(1)= A & vars(2)= n

clear all; close all; clc;

D = load('icevelocity.txt');
z = D(:,1);
v = D(:,2);

fun=@(vars) v(1,1)-(vars(1)*(917*9.8*sin(10).^vars(2)))*(z(1,1)^(vars(2)+1));
Start_vars=[20,1]; % we need to give the vars starting points, chose the min of the ranges found above
OPTvars=fminsearch(fun,Start_vars); %we need to hand fminsearch our function and starting points for the variables 
A= OPTvars(1);
n= OPTvars(2);

Optimum_A = [(A)];
Optimum_n = [(n)];


Optimum_Values = table(Optimum_A,Optimum_n)

%% QUESTION 10

%we are only looking for the optimal A value so when using the function
%this time we will just replace n with its optimal value of 1 

clear; close all; clc;
D = load('icevelocity.txt');
for n = 1:1000   
    ninper=round(size(D,1)*0.9); %this tells us what 90% of the length of D is 
    rows=randperm(size(D,1),ninper); % this will select 82 random rows from the length of D
    samp=D(rows,:); %this takes those random rows and finds what values those are within D
    fun=@(A) sqrt(mean((samp(:,2)-(D(1,2)-A*(917*9.8*sin(10))*(samp(:,1).^2))).^2));
    %we use our iceflow equation and run through the random samples of z
    %and find that best A value, then we use that best A value to find our
    %predicted v, then we compare that to the actual v values to find the
    %RMSE
    OPTvars=fminsearch(fun, 20); % since we are only looking for one variable to OPT we can just directly insert the starting value
    Avals(n,:)=OPTvars; %what are the best A values for every sample 
    RMSE(n,:)=fun(OPTvars); %what were every RMSE 
end
subplot(1, 2, 1);
histogram(Avals)
title('OPT A Vals');
subplot(1, 2, 2);
histogram(RMSE)
title('RMSE');

%% QUESTION 11 + 12 + 13

clear; close all; clc;
D = load('icevelocity.txt');
for n = 1:1000   
    ninper=round(size(D,1)*0.9);
    rows=randperm(size(D,1),ninper); 
    samp=D(rows,:); 
    fun=@(A) sqrt(mean((samp(:,2)-(D(1,2)-A*(917*9.8*sin(10))*(samp(:,1).^2))).^2));
    OPTvars=fminsearch(fun, 20);
    Avals(n,:)=OPTvars; 
    RMSE(n,:)=fun(OPTvars);
end

boxchart(Avals)

% QUESTION 12

Astat=[nanmean(Avals),nanstd(Avals)];
RMSEstat=[nanmean(RMSE),nanstd(RMSE)];

%this is how we can generate 1k values using the mu and sd
x = Astat(1,1) + Astat(1,2) *randn(1000,1); %add the ,1 so that it is just 1 column with out that it will make a 1k by 1k
y = RMSEstat(1,1) + RMSEstat(1,2) *randn(1000,1);

%produces 1k simulated values for A as x and for RMSE as y

% QUESTION 13

h = kstest2(Avals,x)





