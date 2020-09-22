function [ BetaOut,TStatOut,POut,FuncOutputSE ] = PvbRegressionAnalysis...
    (EvidenceUnitsA,EvidenceUnitsB,ChosenTarget)
%Input the Evidence for the left option; right option; and the chosen target (1 for left)
%It will output the regression betas, t-statistics, p-values, and standard errors
 %% Design matrix for regression (i.e. equation 5 in Methods section)
    dm = [ mean(EvidenceUnitsA,2)-mean(EvidenceUnitsB,2) ... %difference in mean evidence between left and right options
         nanstd(EvidenceUnitsA,[],2)-nanstd(EvidenceUnitsB,[],2)]; %difference in Std of evidence between left and right options
%% Run the logistic regression

[~,~,Stats] = glmfit(dm,(ChosenTarget==1)',...
    'binomial','link','logit','constant','on');

%% Organise function output
BetaOut =Stats.beta;
TStatOut =Stats.t;
POut = Stats.p; 
FuncOutputSE = Stats.se;
end

