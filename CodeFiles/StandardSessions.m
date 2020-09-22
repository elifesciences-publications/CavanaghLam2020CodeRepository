function [PythonVars] = StandardSessions(SubjectToTest,MainDirectory);
%% Editable Variables:
n_kcv_runs = 5; % How many cross-validation runs to run? Note, in the paper, this was set to 100. It has been set to 10 here to decrease computing time. 
%% Other directory information
CodesFileLocations =     fullfile(MainDirectory,'CodeFiles'); addpath(CodesFileLocations);
DataFileLocations = fullfile(MainDirectory,'DataFiles');

%% Load in the subject data files. 
if strcmp(SubjectToTest,'Both')
    NameOfFileForThisSubj = FuncToLoadInData('Harry',0,CodesFileLocations,DataFileLocations);
    load(NameOfFileForThisSubj,'DataStructureSessions'); %Load the data in for Subject H 
    HarryStruct = DataStructureSessions;
    
    NameOfFileForThisSubj = FuncToLoadInData('Alfie',0,CodesFileLocations,DataFileLocations);
    load(NameOfFileForThisSubj,'DataStructureSessions'); %Load the data in for Subject A 
    AlfieStruct = DataStructureSessions;
    DataStructureSessions = cat(2,HarryStruct,AlfieStruct); %Concatenate the data for both subjects
else
    NameOfFileForThisSubj = FuncToLoadInData(SubjectToTest,0,CodesFileLocations,DataFileLocations);
    load(NameOfFileForThisSubj,'DataStructureSessions'); %Load the data for the chosen subject only 
end
%% Collapse the variables across sessions
EvidenceUnitsACollapsed = cat(1,DataStructureSessions.EvidenceUnitsA); %Evidence values for the Left Option on Completed trials
EvidenceUnitsBCollapsed = cat(1,DataStructureSessions.EvidenceUnitsB); %Evidence values for the Right Option on Completed trials
ChosenTargetCollapsed = cat(2,DataStructureSessions.ChosenTarget); %Chosen target (1=left; 2=right) on Completed trials
LongSampleTrial = cat(2,DataStructureSessions.LongSampleTrial); %Did the trial contain 8 evidence samples on each side (cf. 4 sample trials) - all trials.
HighTrial = cat(2,DataStructureSessions.HighTrial); %Was the subject instructed to choose the series of bars with the higher average height (ChooseTaller trials)? (cf. ChooseShort trials)
RespTr = cat(1,DataStructureSessions.resp_trials); %Did the subject complete the trial by making a behavioural choice?
TrialType = cat(2,DataStructureSessions.CompletedTrialType); %Reference number of the trial type for further anaylses
TrialError = cat(1,DataStructureSessions.TrialError); %MonkeyLogic record of trial outcome (0 = correct; 6 = incorrect; others = imply incomplete)
%% Methods section of paper:
MemoryTr = cat(2,DataStructureSessions.MGS_TrialByCodes);
TrErOnDecisionTr = TrialError(~MemoryTr);
MethodsData.ProportionBreakFixTr = mean(TrErOnDecisionTr==5 | TrErOnDecisionTr==3); %Proportion of break fixation trials
MethodsData.InitFixNotCompleted = mean(TrErOnDecisionTr==8 | TrErOnDecisionTr==4); %Proportion of trials where the subject did not complete the initial fixation at the start of the trial
MethodsData.ProportionNoRespTr = mean(TrialError==1); %Proportion of trials where the subject did not break fixaton or indicate a response.
MethodsData.TotalBreakFixProp =  MethodsData.ProportionBreakFixTr+   MethodsData.InitFixNotCompleted;

PropOfEachTrType = FuncToShowTrTypeProportions(DataStructureSessions,1:length(DataStructureSessions));
MethodsData.RegularTrAsAPropOfCompleteTr = PropOfEachTrType(1);
MethodsData.HalfHalfTrAsAPropOfCompleteTr = PropOfEachTrType(2);
MethodsData.NarrowBroadTrAsAPropOfCompleteTr = PropOfEachTrType(3);
MethodsData.ControlTrAsAPropOfCompleteTr = PropOfEachTrType(4);
MethodsData.Number8SampleTr = sum(  LongSampleTrial(RespTr==1)==1 & (TrialType==1 | (TrialType>17 & TrialType<24)));
%I.e. how many long sample trials did they COMPLETE. Which were either
%regular of NarrowBroad.
PythonVars.MethodsData = MethodsData;
  
%% Make sure all variables have the same dimensions.
TrialErrorComp = TrialError(RespTr); %The monkeylogic trial error codes only for completed trials
LongSampleTrial = LongSampleTrial(RespTr==1); %Reference to 8-sample trials, of those which were completed
HighTrial = HighTrial(RespTr==1); %Reference to ChooseTall trials, of those which were completed
RespTr = RespTr(RespTr==1);
%% Indexes the trials to use in the analyses
%Only use long trials (LongSampleTrial), which are regular or NarrowBroad
IndexOfTrToUse = LongSampleTrial==1 & (TrialType==1 | (TrialType>17 & TrialType<24)) ;
%% Redefine the variables to only reflected included trials
EvidenceUnitsACollapsed = EvidenceUnitsACollapsed(IndexOfTrToUse,:);
EvidenceUnitsBCollapsed = EvidenceUnitsBCollapsed(IndexOfTrToUse,:);
ChosenTargetCollapsed = ChosenTargetCollapsed(IndexOfTrToUse);
HighTrial = HighTrial(IndexOfTrToUse);
RespTr = RespTr(IndexOfTrToUse);
TrialErrorComp = TrialErrorComp(IndexOfTrToUse);
TrialType = TrialType(IndexOfTrToUse);
LongSampleTrial = LongSampleTrial(IndexOfTrToUse);
%%  Make some variables to use if we only want to look at regular tr. (TrialType=1)
IndexOfTrToUseInRegFigs = TrialType==1 ;

EvidenceUnitsACollapsed_reg = EvidenceUnitsACollapsed(IndexOfTrToUseInRegFigs,:);
EvidenceUnitsBCollapsed_reg = EvidenceUnitsBCollapsed(IndexOfTrToUseInRegFigs,:);
ChosenTargetCollapsed_reg = ChosenTargetCollapsed(IndexOfTrToUseInRegFigs);
LongSampleTrial_reg = LongSampleTrial(IndexOfTrToUseInRegFigs);
HighTrial_reg = HighTrial(IndexOfTrToUseInRegFigs);
RespTr_reg = RespTr(IndexOfTrToUseInRegFigs);
TrialErrorComp_reg = TrialErrorComp(IndexOfTrToUseInRegFigs);
TrialType_reg = TrialType(IndexOfTrToUseInRegFigs);

%% Figure 2ab: Psychometrics
%Note, in the paper Fig2ab, this data is analysed separately for each subject.  
[~,P_corr_dx_list,ErrBar_P_corr_dx_list,PythonVars.Fig2.psychometric_params_Subj_non_drug_corr] = ...
    FinalPsychometricsInFunc_Final(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,ChosenTargetCollapsed,...
    TrialErrorComp,'CorrectIncorrect');
%% Figure 2 - figure supplement 1 A,B
%Split the psychometric functions according to whether it was a 'ChooseTall' or 'ChooseShort' trial. 
[~,P_corr_dx_listByContext(:,1),ErrBar_P_corr_dx_listByContext(:,1),PythonVars.Fig2Supp1.ChooseHigh.psychometric_params_Subj_non_drug_corr] = ...
    FinalPsychometricsInFunc_Final(EvidenceUnitsACollapsed(HighTrial==1,:),EvidenceUnitsBCollapsed(HighTrial==1,:),ChosenTargetCollapsed(HighTrial==1),...
    TrialErrorComp(HighTrial==1,:),'CorrectIncorrect');
[~,P_corr_dx_listByContext(:,2),ErrBar_P_corr_dx_listByContext(:,2),PythonVars.Fig2Supp1.ChooseLow.psychometric_params_Subj_non_drug_corr] = ...
    FinalPsychometricsInFunc_Final(EvidenceUnitsACollapsed(HighTrial==0,:),EvidenceUnitsBCollapsed(HighTrial==0,:),ChosenTargetCollapsed(HighTrial==0),...
    TrialErrorComp(HighTrial==0,:),'CorrectIncorrect');

%% Figure 2cd - Psychophysical kernels (temporal weightings)
%Note, in the paper Fig2cd, this data is analysed separately for each subject.  
tmp_PK_GLM = [];                                                          
tmp_PK_GLM(:,9) = transpose(ChosenTargetCollapsed); %Store the subjects' choices in the 9th column of this variable
tmp_PK_GLM(tmp_PK_GLM==2) = 0; %Choices are assigned as 1 (Chose Left); 0 (Chose Right)
tmp_PK_GLM(:,1:8) = 100.*(EvidenceUnitsACollapsed - EvidenceUnitsBCollapsed); %Difference between evidence on the left and right at each timestep
[NonDrugDayPairedData] = RunPkAnalysis(tmp_PK_GLM,[]); 

PythonVars.Fig2.PK_Subj_nondrug =  NonDrugDayPairedData(:,1)'; %Beta weights for each time step
PythonVars.Fig2.PK_Subj_nondrug_errbar =  NonDrugDayPairedData(:,2)'; %Standard error for each time step
%% Figure 2 - figure supplement 1 C,D 
%Split the Psychophysical kernels according to whether it was a 'ChooseTall' or 'ChooseShort' trial. 
[PKSplitByContext(:,:,1)] = RunPkAnalysis(tmp_PK_GLM(HighTrial==1,:),[]);
[PKSplitByContext(:,:,2)] = RunPkAnalysis(tmp_PK_GLM(HighTrial==0,:),[]);

PythonVars.Fig2Supp1.ChooseHigh.PK_Subj_nondrug =  PKSplitByContext(:,1,1)';
PythonVars.Fig2Supp1.ChooseHigh.PK_Subj_nondrug_errbar =  PKSplitByContext(:,2,1)';
PythonVars.Fig2Supp1.ChooseLow.PK_Subj_nondrug =  PKSplitByContext(:,1,2)';
PythonVars.Fig2Supp1.ChooseLow.PK_Subj_nondrug_errbar =  PKSplitByContext(:,2,2)';

%% Figure 3b-c: Narrow-Broad trials.
%Note, the same analyses run with individual subject data are presented as Fig3S1G-J in the paper
[NarrowBroadTrialsCOL(:,:),NarrowBroadTrialsCOL_Errs(:,:),...
    StatsOutputForPaperNarrowBroadTr] =  AnalNarrowBroadTrials_final(TrialType...
    ,ChosenTargetCollapsed); % Probability to choose in narrow-correct, broad-correct, and ambiguous cases.
PythonVars.Fig3bc.ENB_bars_Subj_non_drug = [NarrowBroadTrialsCOL(end) NarrowBroadTrialsCOL(1:2)]; %Choice probabilities
PythonVars.Fig3bc.ENB_bars_err_Subj_non_drug = [NarrowBroadTrialsCOL_Errs(end) NarrowBroadTrialsCOL_Errs(1:2)]; %Standard errors
PythonVars.Fig3bc.StatsOutputForPaperNarrowBroadTr = StatsOutputForPaperNarrowBroadTr; %Stats tests

%% Figure 3 - figure supplement 2 - extend the above to split by ChooseHigh and ChooseLow context trials
[NarrowBroadTrialsCOL_HighLow(:,1),NarrowBroadTrialsCOL_Errs_HighLow(:,1),StatsOutputForPaperNarrowBroadTr_HighLow{1}]...
    =  AnalNarrowBroadTrials_final(TrialType(HighTrial==1),ChosenTargetCollapsed(HighTrial==1));
[NarrowBroadTrialsCOL_HighLow(:,2),NarrowBroadTrialsCOL_Errs_HighLow(:,2),StatsOutputForPaperNarrowBroadTr_HighLow{2}]...
    =  AnalNarrowBroadTrials_final(TrialType(HighTrial==0),ChosenTargetCollapsed(HighTrial==0));

PythonVars.Fig3Sup2.ChooseHigh.ENB_bars_Subj_non_drug = [NarrowBroadTrialsCOL_HighLow(3,1); NarrowBroadTrialsCOL_HighLow(1:2,1)];
PythonVars.Fig3Sup2.ChooseHigh.ENB_bars_err_Subj_non_drug = [NarrowBroadTrialsCOL_Errs_HighLow(3,1); NarrowBroadTrialsCOL_Errs_HighLow(1:2,1)];
PythonVars.Fig3Sup2.ChooseHigh.StatsOutputForPaperNarrowBroadTr = StatsOutputForPaperNarrowBroadTr_HighLow{1};

PythonVars.Fig3Sup2.ChooseLow.ENB_bars_Subj_non_drug = [NarrowBroadTrialsCOL_HighLow(3,2); NarrowBroadTrialsCOL_HighLow(1:2,2)];
PythonVars.Fig3Sup2.ChooseLow.ENB_bars_err_Subj_non_drug = [NarrowBroadTrialsCOL_Errs_HighLow(3,2); NarrowBroadTrialsCOL_Errs_HighLow(1:2,2)];
PythonVars.Fig3Sup2.ChooseLow.StatsOutputForPaperNarrowBroadTr = StatsOutputForPaperNarrowBroadTr_HighLow{2};
%% Extracting the distribution of evidence samples for each trial type (used in Fig4AB, Fig3S1 and Fig4S1)
[n_distribution_regularTr,n_distribution_narrow_high,...
    n_distribution_broad_high,n_distribution_NB_balanced,...
    n_SD_distribution_regularTr]...
    = GetTrialDistributionData_NL_final(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,TrialType);

PythonVars.TrialDistributionFigs.n_distribution_regularTr = n_distribution_regularTr;
PythonVars.TrialDistributionFigs.n_distribution_narrow_high = n_distribution_narrow_high;
PythonVars.TrialDistributionFigs.n_distribution_broad_high = n_distribution_broad_high;
PythonVars.TrialDistributionFigs.n_distribution_NB_balanced = n_distribution_NB_balanced;
PythonVars.TrialDistributionFigs.n_SD_distribution_regression = n_SD_distribution_regularTr;

%% Figure 4c: Non-drug sessions: Pro-variance effects displayed using a psychometric function
%Note, the same analyses run with individual subject data are presented as Fig4S1C;Fig4S1F in the paper
[~,P_corr_Subj_list,ErrBar_P_corr_Subj_list,PythonVars.Fig4c.Psychometric_fit_paramsFig4] = ...
    FinalPsychometricsInFunc_Final(EvidenceUnitsACollapsed_reg,EvidenceUnitsBCollapsed_reg,ChosenTargetCollapsed_reg,...
    TrialErrorComp_reg,'NarrowBroad');
PythonVars.Fig4c.P_corr_Subj_list = P_corr_Subj_list; %Choice probabilities for each bin
PythonVars.Fig4c.ErrBar_P_corr_Subj_list = ErrBar_P_corr_Subj_list; %Standard errors for each bin
%% Figure 4-supplement 2 (panels A, D, G, J): - extend the above to split by ChooseHigh and ChooseLow context trials
%Note, in the paper, these analyses are run separated by subject
[~,P_corr_Subj_list_Hi,ErrBar_P_corr_Subj_list_Hi,PythonVars.Fig4Sup2.ChooseHigh.Psychometric_fit_params] = ...
    FinalPsychometricsInFunc_Final(EvidenceUnitsACollapsed_reg(HighTrial_reg==1,:),EvidenceUnitsBCollapsed_reg(HighTrial_reg==1,:),...
    ChosenTargetCollapsed_reg(HighTrial_reg==1),TrialErrorComp_reg(HighTrial_reg==1),'NarrowBroad');
[~,P_corr_Subj_list_Lo,ErrBar_P_corr_Subj_list_Lo,PythonVars.Fig4Sup2.ChooseLow.Psychometric_fit_params] = ...
    FinalPsychometricsInFunc_Final(EvidenceUnitsACollapsed_reg(HighTrial_reg==0,:),EvidenceUnitsBCollapsed_reg(HighTrial_reg==0,:),...
    ChosenTargetCollapsed_reg(HighTrial_reg==0),TrialErrorComp_reg(HighTrial_reg==0),'NarrowBroad');

PythonVars.Fig4Sup2.ChooseHigh.P_corr_Subj_list = P_corr_Subj_list_Hi;
PythonVars.Fig4Sup2.ChooseHigh.ErrBar_P_corr_Subj_list = ErrBar_P_corr_Subj_list_Hi;
PythonVars.Fig4Sup2.ChooseLow.P_corr_Subj_list = P_corr_Subj_list_Lo;
PythonVars.Fig4Sup2.ChooseLow.ErrBar_P_corr_Subj_list = ErrBar_P_corr_Subj_list_Lo;

%% Figure 4d - Regression for pro-variance effect
%Note, the same analyses run with individual subject data are presented as Fig4S1D;Fig4S1G in the paper
[Betas,TStatOut,POut,ErrCollapsed] = ...
    PvbRegressionAnalysis(100*EvidenceUnitsACollapsed_reg...
    ,100*EvidenceUnitsBCollapsed_reg...
    ,ChosenTargetCollapsed_reg);
 
PythonVars.Fig4d.Reg_bars_Subj_non_drug = Betas'; %Beta weights 
PythonVars.Fig4d.Reg_bars_err_Subj_non_drug = ErrCollapsed'; %Standard errors
PythonVars.Fig4d.PVB_T_Stats = TStatOut; %Stats reporting
PythonVars.Fig4d.PVB_PVals = POut; %Stats reporting

%% Figure 4-supplement 2 (panels B, E, H, K): - extend the above to split by ChooseHigh and ChooseLow context trials
%Note, in the paper, these analyses are run separated by subject
[Betas_HiLo(:,1),Ts_HiLo(:,1),P_HiLo(:,1),Ers_HiLo(:,1)] = ...
    PvbRegressionAnalysis(100*EvidenceUnitsACollapsed_reg(HighTrial_reg==1,:)...
    ,100*EvidenceUnitsBCollapsed_reg(HighTrial_reg==1,:)...
    ,ChosenTargetCollapsed_reg(HighTrial_reg==1));
[Betas_HiLo(:,2),Ts_HiLo(:,2),...
    P_HiLo(:,2),Ers_HiLo(:,2)] = ...
    PvbRegressionAnalysis(100*EvidenceUnitsACollapsed_reg(HighTrial_reg==0,:)...
    ,100*EvidenceUnitsBCollapsed_reg(HighTrial_reg==0,:)...
    ,ChosenTargetCollapsed_reg(HighTrial_reg==0));

PythonVars.Fig4Sup2.ChooseHigh.Reg_bars_Subj_non_drug = Betas_HiLo(:,1)';
PythonVars.Fig4Sup2.ChooseHigh.Reg_bars_err_Subj_non_drug = Ers_HiLo(:,1)';
PythonVars.Fig4Sup2.ChooseLow.Reg_bars_Subj_non_drug = Betas_HiLo(:,2)';
PythonVars.Fig4Sup2.ChooseLow.Reg_bars_err_Subj_non_drug = Ers_HiLo(:,2)';

PythonVars.Fig4Sup2.ChooseHigh.PVB_T_Stats = Ts_HiLo(:,1); %Stats reporting
PythonVars.Fig4Sup2.ChooseHigh.PVB_PVals = P_HiLo(:,1); %Stats reporting
PythonVars.Fig4Sup2.ChooseLow.PVB_T_Stats = Ts_HiLo(:,2); %Stats reporting
PythonVars.Fig4Sup2.ChooseLow.PVB_PVals = P_HiLo(:,2); %Stats reporting

%% Figure 4 - supplement 1E, H: Extended Regression Analysis.
[RegrOutputs] = RegressionToDetermineSubjStrategy( 100*EvidenceUnitsACollapsed_reg,...
    100*EvidenceUnitsBCollapsed_reg...
    ,ChosenTargetCollapsed_reg==1,[]);

PythonVars.Fig4Sup1.Reg_values_Subj_nondrug = RegrOutputs.BetaWeights;
PythonVars.Fig4Sup1.Reg_bars_err_Subj_non_drug = RegrOutputs.SeOfBetaWeights;
%% Figure 4-supplement 2 (panels C, F, I, L): - extend the above to split by ChooseHigh and ChooseLow context trials

for xx=1:2
    tkp = -1*(xx-2);
    [RegrOutputsSplit{xx}] = RegressionToDetermineSubjStrategy( 100*EvidenceUnitsACollapsed_reg(HighTrial_reg==tkp,:),...
        100*EvidenceUnitsBCollapsed_reg(HighTrial_reg==tkp,:)...
        ,ChosenTargetCollapsed_reg(HighTrial_reg==tkp)==1,[]);
end

PythonVars.Fig4Sup2.ChooseHigh.Reg_values_Subj_nondrug = RegrOutputsSplit{1}.BetaWeights;
PythonVars.Fig4Sup2.ChooseHigh.Reg_bars_err_Subj_non_drug = RegrOutputsSplit{1}.SeOfBetaWeights;
PythonVars.Fig4Sup2.ChooseLow.Reg_values_Subj_nondrug = RegrOutputsSplit{2}.BetaWeights;
PythonVars.Fig4Sup2.ChooseLow.Reg_bars_err_Subj_non_drug = RegrOutputsSplit{2}.SeOfBetaWeights;

%% Figure 4- Supplement 3, Local winners
%Note, this analysis is run separated by subject
LocalWinsOutput = ...
    LocalWinnerReg(EvidenceUnitsACollapsed_reg,EvidenceUnitsBCollapsed_reg,ChosenTargetCollapsed_reg);

PythonVars.Fig4Sup3.TVals = LocalWinsOutput.t(2:3);
PythonVars.Fig4Sup3.PVals = LocalWinsOutput.p(2:3);
PythonVars.Fig4Sup3.Betas = LocalWinsOutput.beta(2:3);
PythonVars.Fig4Sup3.Ses = LocalWinsOutput.se(2:3);
 
%% Supplementary Tables 1-3: Model comparison results
PythonVars.SuppTableOutput = Fig4ModelCompare_Final_F(EvidenceUnitsACollapsed_reg,EvidenceUnitsBCollapsed_reg,...
    ChosenTargetCollapsed_reg,...
    n_kcv_runs);
end
