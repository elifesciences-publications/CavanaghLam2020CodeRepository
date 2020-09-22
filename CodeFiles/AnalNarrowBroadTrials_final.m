function [ NarrowBroadTrials,Errs,StatsOutputForPaper] = AnalNarrowBroadTrials_final ...
    (TrialType,ChosenTarget)
%% Define the Trial Types

BroadLow_Versus_NarrowHigh = TrialType ==18; 
BroadHigh_Versus_NarrowLow = TrialType ==19; 
BroadBalanced_Versus_NarrowBalanced = TrialType==20; 
NarrowHigh_Versus_BroadLow = TrialType ==21; 
NarrowLow_Versus_BroadHigh = TrialType ==22; 
NarrowBalanced_Versus_BroadBalanced = TrialType==23; 
  
%% Calculate choice probabilities for each trial type
NarrowBroadTrials(1) = (sum(ChosenTarget(BroadLow_Versus_NarrowHigh)~=1)...
    + sum(ChosenTarget(NarrowHigh_Versus_BroadLow )==1))/sum(BroadLow_Versus_NarrowHigh...
    +NarrowHigh_Versus_BroadLow);
%Accuracy on NarrowCorrect trials

NarrowBroadTrials(2) = (sum(ChosenTarget(NarrowLow_Versus_BroadHigh )~=1)...
    + sum(ChosenTarget(BroadHigh_Versus_NarrowLow )==1))/sum(NarrowLow_Versus_BroadHigh...
    +BroadHigh_Versus_NarrowLow);
%Accuracy on BroadCorrect trials

NarrowBroadTrials(3) = (sum(ChosenTarget(BroadBalanced_Versus_NarrowBalanced )==1)...
    + sum(ChosenTarget(NarrowBalanced_Versus_BroadBalanced )~=1))/...
    sum(BroadBalanced_Versus_NarrowBalanced...
    +NarrowBalanced_Versus_BroadBalanced);
%Broad preference on Ambiguous trials

%% Calculate error bars
Errs = sqrt((NarrowBroadTrials.*(1-NarrowBroadTrials))./(cat(2,sum(BroadLow_Versus_NarrowHigh...
    +NarrowHigh_Versus_BroadLow),sum(NarrowLow_Versus_BroadHigh...
    +BroadHigh_Versus_NarrowLow),sum(BroadBalanced_Versus_NarrowBalanced...
    +NarrowBalanced_Versus_BroadBalanced))));
%Errorbars calculated using: Square root (    (prob * (1- prob)) / Number of Tr  )
%% Hypothesis testing 
NarrowCorrect_AccurateNum = (sum(ChosenTarget(BroadLow_Versus_NarrowHigh)~=1)...
    + sum(ChosenTarget(NarrowHigh_Versus_BroadLow )==1));

NarrowCorrect_TrNum = sum(BroadLow_Versus_NarrowHigh...
    +NarrowHigh_Versus_BroadLow);

BroadCorrect_AccurateNum = (sum(ChosenTarget(NarrowLow_Versus_BroadHigh )~=1)...
    + sum(ChosenTarget(BroadHigh_Versus_NarrowLow )==1));

BroadCorrect_TrNum = sum(NarrowLow_Versus_BroadHigh...
    +BroadHigh_Versus_NarrowLow);

AmbigTr_BroadChosenNum = (sum(ChosenTarget(BroadBalanced_Versus_NarrowBalanced )==1)...
    + sum(ChosenTarget(NarrowBalanced_Versus_BroadBalanced )~=1));

AmbigTr_TrNum =   sum(BroadBalanced_Versus_NarrowBalanced...
    +NarrowBalanced_Versus_BroadBalanced);

% Compare the proportion correct on NarrowCorrect and BroadCorrect trials
ChiTableInput = [NarrowCorrect_AccurateNum NarrowCorrect_TrNum-NarrowCorrect_AccurateNum;...
    BroadCorrect_AccurateNum BroadCorrect_TrNum-BroadCorrect_AccurateNum];
 
[ChiP,Q] = chi2test(ChiTableInput);

if ChiP==0
    %Work out the max possible p-value, if matlab reports it as 0,
    QvalsToTest = 70:0.1:80;
    pvalReport = 1-gammainc(QvalsToTest/2,0.5);
    min(pvalReport(pvalReport>0));
end

%Compare the broad preference on Ambiguous trials to chance
BinomP = myBinomTest(AmbigTr_BroadChosenNum,AmbigTr_TrNum,0.5);

%% Organise the function output
StatsOutputForPaper.ChiP = ChiP;
StatsOutputForPaper.Q = Q;
StatsOutputForPaper.BinomP = BinomP;

end

