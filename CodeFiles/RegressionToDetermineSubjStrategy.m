function [RegrOutputs,CollationOfBetas,bestPModel,BootParamSave]...
    = RegressionToDetermineSubjStrategy( EvidenceUnitsA,EvidenceUnitsB,ChosenTarget,BootStrapNo)
%% Design matrix for regression (equation 6 in methods section)
dm = [EvidenceUnitsA(:,1) EvidenceUnitsA(:,end)  mean(EvidenceUnitsA,2)...
    max(EvidenceUnitsA,[],2) min(EvidenceUnitsA,[],2)...
    EvidenceUnitsB(:,1) EvidenceUnitsB(:,end) mean(EvidenceUnitsB,2)...
    max(EvidenceUnitsB,[],2) min(EvidenceUnitsB,[],2)];
%Left first sample evidence; Left last sample evidence; Mean left
%evidence; Max left evidence; Min left evidence; Right first sample
%evidence; Right last sample evidence; Mean right evidence; Max right
%evidence; Min right evidence)
%% Logistic regression
[~,~,RegStats] = glmfit(dm,ChosenTarget',...
    'binomial','link','logit','constant','on');
%% Lapse term model, if performing bootstraps (only for pharmacological sessions)
if ~isempty(BootStrapNo)
    OriginalDM = [ones(size(dm,1),1) dm];
    [CollationOfBetas,bestPModel,BootParamSave,~]= ...
        scFittingFunctionForLapsesF2(OriginalDM,ChosenTarget',BootStrapNo);
end
%% Organise function output
RegrOutputs.RegStats =RegStats;
RegrOutputs.BetaWeights = RegStats.beta;
RegrOutputs.SeOfBetaWeights =RegStats.se;


end

