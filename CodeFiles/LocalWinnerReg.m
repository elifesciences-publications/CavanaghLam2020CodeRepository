function [LocalWinsOutput] = LocalWinnerReg(EvidenceUnitsACollapsed_reg,EvidenceUnitsBCollapsed_reg,ChosenTargetCollapsed_reg)
% Equation 7, see Methods section
%% Construct the design matrix
Mu = mean(EvidenceUnitsACollapsed_reg,2) - mean(EvidenceUnitsBCollapsed_reg,2); 
Lw = sum(EvidenceUnitsACollapsed_reg>EvidenceUnitsBCollapsed_reg,2)-sum(EvidenceUnitsACollapsed_reg<EvidenceUnitsBCollapsed_reg,2);
dm = [Mu Lw];
%% Run the regression
[~,~,LocalWinsOutput] = glmfit(dm,ChosenTargetCollapsed_reg'==1,'binomial','link','logit','constant','on');
end

