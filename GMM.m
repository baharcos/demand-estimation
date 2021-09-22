function f = GMM(x0)

global y A IV W Kbeta Ktheta ns share TM prods Total T Kgamma price z 

%Observed product characteristics parameters
theta1 = x0(1:Kbeta,1);
%Observed cost parameters
gamma1 = x0(Kbeta+1:end,1);             
%Demand side
xi_jm = y - [ones(Total,1) A price]*theta1;

%Supply side
mc_jm = price+1./(theta1(end).*(1-share)); 
w_jm = [ones(Total,1) z];    
omega_jm = mc_jm - w_jm*gamma1; 

%Moment
g = IV'*[xi_jm;omega_jm];    

%GMM objective function
f       = g'*W*g;                                          


end