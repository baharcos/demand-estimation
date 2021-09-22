%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EMPIRICAL INDUSTRIAL ORGANIZATION - DEMAND ESTIMATION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%　Group 3　%%%%%%%%
 % Bahar COSKUN  
 % Youssef DHAOUI CHOUAIEB 
 % Mansoor MALIK 
 % Thomas MONNIER 
 % Andrea STRINGHETTI 
 % Naoki TANI 

% This gets you started for the following models
% (1) Logit with and without IV
% (2) Nested Logit

%%%%%%%%%%%%%%%%
%%% SETTINGS %%%
%%%%%%%%%%%%%%%%

clc;
clear;

global y A IV W Kbeta Ktheta ns share TM prods Total T Kgamma price z

DATA       = csvread('Data.csv');
IDmkt      = DATA(:,1);                 % Market identifier
IDprod     = DATA(:,2);                 % Product identifier
share      = DATA(:,3);                 % Market share
A          = DATA(:,4:6);               % Product characteristics
price      = DATA(:,7);                 % Price
z          = DATA(:,8:10);              % Instruments - MC
MC         = [ones(970,1),z];           % Marginal costs (including constant)              
TM         = max(IDmkt);                % Number of markets
group      = DATA(:,11);                % Group identifier for Nested Logit
prods      = zeros(TM,1);               % # of products in each market


for m=1:TM
    prods(m,1) = max(IDprod(IDmkt==m,1));
end
T          = zeros(TM,2);
T(1,1)     = 1; 
T(1,2)     = prods(1,1); 
for i=2:TM
    T(i,1) = T(i-1,2)+1;                % 1st Column market starting point
    T(i,2) = T(i,1)+prods(i,1)-1;       % 2nd Column market ending point
end
Total      = T(TM,2);                   % # of obsevations
TotalProd  = max(prods);                % Max # of products in a given market
Ngroups    = max(group);                % # of groups



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% TRUE PARAMETER VALUES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

betatrue = [3 3 0.5 0.5 -2]';           % True mean tastes 
gamma = [5 .5 .5 .5]';                  % True MC 
Kbeta    = size(betatrue,1);            % # attributes
Kgamma = 4;                             % # cost parameters   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CALCULATE MARKET SHARES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% NB: As there is an outside option, market shares do not sum up to 1!

S_0      = zeros(TM,1); 

for m=1:TM
    S_0(m,1) = 1-sum(share(IDmkt==m,1));
end

y = log(share./S_0(IDmkt,1));
x = [ones(Total,1) A price];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OLS ESTIMATION OF HOMOGENEOUS LOGIT MODEL %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1a. 

[betaOLS,seOLS,~] = OLS(y,x);

% Display results: 

str1    = [betaOLS seOLS betatrue];
disp('*************************');
disp('      OLS estimates:     ');
disp('*************************');
disp(['    Coeff','    ','Std Err','    ','True']);
disp(str1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OLS ESTIMATION OF FIRST STAGE IV %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1b.

z_iv = [ones(Total,1) z];

[IV1S,seIV1S,Rsq,AdjRsq,F] = OLS(price,z_iv);

price_iv = z_iv*IV1S;

% Display results 1st stage: %

str    = [IV1S seIV1S];
disp('******************************');
disp('  First stage IV estimates:   ');
disp('******************************');
disp(['    Coeff','     ','Std Err']);
disp(str);
disp(['Rsq:    ',num2str(Rsq)]);
disp(['AdjRsq: ',num2str(AdjRsq)]);
disp(['F-stat: ',num2str(F)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2SLS ESTIMATION OF HOMOGENEOUS LOGIT MODEL %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_iv = [ones(Total,1) A price_iv];

[beta2SLS,~] = OLS(y,x_iv);

% Structural residuals
u_iv = y - x*beta2SLS;

% Asymptotic covariance
df = Total - Kbeta;
s_hat   = u_iv'*u_iv/df; 

% Estimated covariance matrix
var_hat = s_hat*inv(x_iv'*x_iv); 

% Standard errors
se2SLS  = sqrt(diag(var_hat)); 

% Display results 2nd stage: %

str1     = [beta2SLS se2SLS betatrue];
disp('*************************');
disp('     2SLS estimates:     ');
disp('*************************');
disp(['    Coeff','    ','Std Err','    ','True']);
disp(str1);

%Estimate for alpha is slightly bigger, pointing towards a positive bias:
%negative effect of price is underestimated, due to an omitted variable
%correlated with price and having a positive effect on consumers' utility
%(e.g. overall quality, marketing, etc.).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2SLS ESTIMATION OF HOMOGENEOUS LOGIT MODEL WITH SUPPLY SIDE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 1c. 

rng('default') % I got similar result when I use 'shuffle'.
ns      = 5000;                            

% make instruments and weight matrix
IV      = [ones(Total,1) A z];            
nIV     = size(IV,2);
IV(size(IV,1)+1:2*size(IV,1),size(IV,2)+1:size(IV,2)+1+size(z,2)*2)=[ones(Total,1) z z.^2];
W       = (IV'*IV)\eye(size(IV,2));

%setup initial value of all parameters
x0      = rand(Kbeta+Kgamma,1);

opts    = optimset('display','iter-detailed','Diagnostics','on','TolFun',1e-10,'TolX',1e-10,'GradObj','off','DerivativeCheck','off');
 
tic
[X,fval_rep,exitflag,output,grad,hessian] = fminunc(@GMM,x0,opts);
toc
theta1 = X(1:Kbeta,1);
gamma1 = X(Kbeta+1:end,1);

theta3 = [theta1;gamma1]
stderr1 = sqrt(diag(inv(hessian)));      
stderr2 = stderr1((1:9),:)

 Ttheta      = [betatrue;gamma];
 str1        = [theta3 stderr2 Ttheta];
 disp('****************************');
 disp('   2SLS with supply side:   ');
 disp('****************************');
 disp(['    Coeff','    ','Std Err','    ','True']);
 disp(str1);
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2SLS ESTIMATION OF NESTED LOGIT MODEL %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2. TBC

S_g      = zeros(3,1); 

for i=1:3
    S_g(i,1) = sum(share(group==i,1));
end

lsjg = log(share./S_g(group,1));

[IV1S,~] = OLS(lsjg,z_iv);
lsjg_iv = z_iv*IV1S;

%y = log(share./S_0(IDmkt,1));
x_2 = [ones(Total,1) A price lsjg];
x_iv2 = [ones(Total,1) A price_iv lsjg_iv];

[betaG2SLS,~] = OLS(y,x_iv2);

% Structural residuals
u_iv = y - x_2*betaG2SLS;

% Asymptotic covariance
Gbetatrue = [betatrue;0];
KGbeta    = size(Gbetatrue,1);            % # attributes
df = Total - KGbeta;
s_hat   = u_iv'*u_iv/df; 

% Estimated covariance matrix
var_hat = s_hat*inv(x_iv2'*x_iv2); 

% Standard errors
seG2SLS  = sqrt(diag(var_hat));

str1      = [betaG2SLS seG2SLS Gbetatrue];
disp('**************************');
disp('     G2SLS estimates:     ');
disp('**************************');
disp(['    Coeff','    ','Std Err','    ','True']);
disp(str1);

