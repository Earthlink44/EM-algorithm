clc; clear variables; close all;

%% Data generation
N = 10000;                  % N data points
mu = [1 2;-7 -7];          % Means
sigma(:,:,1) = [2 2; 2 3]; % Covariance Matrix model 1
sigma(:,:,2) = [1 0; 0 1]; % Covariacne Matrix model 2

gm = gmdistribution(mu,sigma);
[Y,compIdx] = random(gm,N);
figure;
scatter(Y(:,1),Y(:,2)); title('Raw data');


%% Expectation Maximization algorithm

% Maximum number of iteration
Nmax = 1000;

%initial guess
mu0_1 = [1 1];
mu0_2 = [-2 -2];
sigma0_1 = [5 0; 0 5];
sigma0_2 = [5 0; 0 5];
pro = 0.5;

for i = 1:Nmax
    % E-step 
    res = pro.*mvnpdf(Y,mu0_2,sigma0_2)./(pro.*mvnpdf(Y,mu0_2,sigma0_2)+(1-pro).*mvnpdf(Y,mu0_1,sigma0_1));
    % M-step Maximum likelihood and calculate new 
    mu0_1 = sum((1-res).*Y)/sum(1-res);
    mu0_2 = sum(res.*Y)/sum(res);
    sigma0_1 = (((1-res).*(Y-mu0_1))'* (Y-mu0_1))/sum(1-res);
    sigma0_2 = (((res).*(Y-mu0_2))'* (Y-mu0_2))/sum(res);
    pro = mean(res);
end
