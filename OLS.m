function [beta,StdErr,rsqr,arsqr,F] = OLS(y,x)

% function[output] = function_name(inputs) : The first line of the function tells MATLAB 
%                                            (a) the name of the function (OLS), 
%                                            (b) the inputs to the function (y and x), 
%                                            (c) the outputs from the function (Beta and se).

% When we call the function OLS, we do not need to use variables named y and X. 
% It just means that, within the program, the variables we have introduced
% will be locally referred to as y and X. 

% RELEVANT INFO: purpose of the code, dimension and description of inputs
% and outputs

%--------------------------------------------------------------------------
% OLS
% inputs: y: Nx1 dependent variable 
%         x: Nxk independent variable
% output: beta: OLS coefficient vector
%         se: standard error of beta
%--------------------------------------------------------------------------

N = length(y);
[N2,K]=size(x);

if N~=N2
   error('my_ols: length of x and y are different')
end

% Calculate the coefficients
beta = (x'*x)\(x'*y); 

% Calculate the standard errors
yhat = x*beta;  
u = yhat - y;                 % residuals
SSR = sum(u.*u);              % sum of Squared Residuals
sigma = SSR/(N-K);            % variance
v_mat = sigma * inv(x'*x);    % covariance of beta
StdErr = diag(sqrt(v_mat));   % s.d. of beta
T_Stat = beta./StdErr;        % t-statistic of beta
CI = [beta-1.96*StdErr beta+1.96*StdErr];
yc = y - mean(y);            % centered y
RSS = yc'*yc;
rsqr = 1 - SSR/RSS;          % r-squared
arsqr = 1 - (SSR/(N-K))/(RSS/(N-1)); % adjusted r-squared
F = (SSR/K) / (RSS/(N-K-1)); % F-stat
end