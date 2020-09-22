function [err,pred] = LapseTermFitRegularisation(p,X,y)
% %model goes here.
%y0 + (1-y0)/(1+exp(-(beta0 + beta1*x1 + beta2*x2 + beta3*x3 etc.)))

%% Setup the function
NTr = size(X,1); %Number of trials
for t=1:size(X,2)
    Betas(t) = p.(['Beta' num2str(t-1)]); %Extract the beta parameters from the input structure
end

%% Add a lapse term to the original logistic regression, and calculate the predictions given current parameters

pred = p.YZERO + (repmat((1 - 2*p.YZERO),NTr,1) ...
    ./   (1+exp(-(X*Betas'))));

%% Calculate the likelhiood given the current parameter set

if exist('y','var')
    
    ProbModelIsCorrect(y==1) = pred(y==1);
    ProbModelIsCorrect(y==0) = 1-pred(y==0);
    Neg_LLofModel = -sum(log(ProbModelIsCorrect));%negative log-likelihood
    
    %% L2 Regularisation
    Lambda = 0.01;
    err = Neg_LLofModel + Lambda*sum(([Betas p.YZERO]).^2);
    
else
    err = NaN;
end
%% Constrain the parameters artificially to prevent implausible parameters
if  p.YZERO<0 | p.YZERO>1 | sum(pred>1)  |    sum(pred<0) ...
        err= 1000000000000000000000000000;
    %Produce a very large cost value if the parameters are implausible
    %(i.e. a lapse parameter >1 or <0; or if a predicted outcome is given a
    %probability >1 or <0.
end
