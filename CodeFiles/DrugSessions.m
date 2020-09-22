function [PythonVars] = DrugSessions(SubjectToTest,MainDirectory);
%% Options
AnalysisList = {'PK';'StratReg';'PVB';'SlidingPVB'}; %Lists the different analyses that will be run
BootStrapNo = 10; %Bootstrap number to generate error estimates for parameters in the lapsing models. Note, in the paper, this was set to 10000. It has been set to 10 here to decrease computing time.
noPermutations = 10; %Permutation number to compare ketamine and saline parameter estimates for parameters in the lapsing models. Note, in the paper, this was set to 10000. It has been set to 10 here to decrease computing time.
LoadResultsFromOldHypothesisTests = 1; % When set to 1, the previous permutations and bootstraps will be loaded.
%% Other directory information
CodesFileLocations = fullfile(MainDirectory,'CodeFiles'); addpath(CodesFileLocations);
DataFileLocations = fullfile(MainDirectory,'DataFiles');
DirToSaveBootstraps = fullfile(DataFileLocations,'Bootstrap_Permutation_Data',SubjectToTest);
%% Load in the subject data files.
if strcmp(SubjectToTest,'Both')
    NameOfFileForThisSubj = FuncToLoadInData('Harry',1,CodesFileLocations,DataFileLocations);
    load(NameOfFileForThisSubj,'DrugDayStructure'); %Load the data in for Subject H
    HarryStruct = DrugDayStructure;
    
    NameOfFileForThisSubj = FuncToLoadInData('Alfie',1,CodesFileLocations,DataFileLocations);
    load(NameOfFileForThisSubj,'DrugDayStructure'); %Load the data in for Subject A
    AlfieStruct = DrugDayStructure;
    
    DrugDayStructure = cat(2,HarryStruct,AlfieStruct); %Concatenate the data for both subjects
else
    NameOfFileForThisSubj = FuncToLoadInData(SubjectToTest,1,CodesFileLocations,DataFileLocations);
    load(NameOfFileForThisSubj,'DrugDayStructure'); %Load the data for the chosen subject only
end
 
%% Analyse the data for each individual session
BinnedAnals = nan(1,81,size(DrugDayStructure,2)); %Organise a matrix, to store performance data in each time bin
OutputDMHere = cell(size(DrugDayStructure,2),81); %Organise a cell array to store design matricies for each time point: Number of Sessions x 81 time points

for ses=1:size(DrugDayStructure,2) %Loop across sessions
    WithinSessionStruct = DrugDayStructure(1,ses); %Load the data structure just for that session
    %% Feed the within session data into a function.
    [SessionWiseDMs(ses),BinnedAnals(:,:,ses),OutputDMHere(ses,:)] = ...
        AnalyseWithinSessionData(WithinSessionStruct);
end

%% Code to exclude sessions which do not meet the inclusion criteria (see Methods)
ExclusionCriteria = WorkOutSessionsToExcludeS(SessionWiseDMs,DrugDayStructure);
%% Drug Session Numbers
PythonVars.Methods.TotalRecordedSessions = length(ExclusionCriteria);
PythonVars.Methods.TotalRecordedKetamineSessions = sum(cat(1,DrugDayStructure.DrugDay));
PythonVars.Methods.TotalRecordedSalineSessions = sum(~cat(1,DrugDayStructure.DrugDay));

PythonVars.Methods.TotalExcludedSessions = sum(ExclusionCriteria);
PythonVars.Methods.TotalExcludedKetamineSessions = sum(ExclusionCriteria & cat(1,DrugDayStructure.DrugDay));
PythonVars.Methods.TotalExcludedSalineSessions = sum(ExclusionCriteria & ~cat(1,DrugDayStructure.DrugDay));

PythonVars.Methods.TotalIncludedSessions = sum(ExclusionCriteria==0);
PythonVars.Methods.TotalIncludedKetamineSessions = sum(ExclusionCriteria==0 & cat(1,DrugDayStructure.DrugDay));
PythonVars.Methods.TotalIncludedSalineSessions = sum(ExclusionCriteria==0 & ~cat(1,DrugDayStructure.DrugDay));

KetamineSes = (find(cat(1,DrugDayStructure.DrugDay))); %Index ketamine sessions
SalineSes = (find(cat(1,DrugDayStructure.DrugDay)==0)); %Index saline sessions

[~,ia,~] = intersect(KetamineSes,find(ExclusionCriteria));
KetamineSes(ia) = []; %Remove excluded ketamine sessions from the indexing array
[~,ia,~] = intersect(SalineSes,find(ExclusionCriteria));
SalineSes(ia) = []; %Remove excluded saline sessions from the indexing array
%% Drug database trial numbers
PythonVars.Methods.TotalOnDrugKetamineTr = size(cat(1,SessionWiseDMs(KetamineSes).OnDrugTemporalWeightsDM),1);
PythonVars.Methods.TotalOnDrugSalineTr = size(cat(1,SessionWiseDMs(SalineSes).OnDrugTemporalWeightsDM),1);
%% Figure 8A: Performance over time (average across sessions)
%Note, the same analyses run with individual subject data are presented as Fig8S1A, F in the paper

% Average the performance in each time bin, relative to injection, across sessions
PythonVars.Fig8A.Pcorr_t_mean_list_ketamine_Subj = nanmean(permute(BinnedAnals(1,:,KetamineSes),[3 2 1]));
ketamine_performance_sd = nanstd(permute(BinnedAnals(1,:,KetamineSes),[3 2 1]));
PythonVars.Fig8A.Pcorr_t_se_list_ketamine_Subj = ketamine_performance_sd/sqrt(length(KetamineSes));

PythonVars.Fig8A.Pcorr_t_mean_list_saline_Subj = nanmean(permute(BinnedAnals(1,:,SalineSes),[3 2 1])); %
saline_performance_sd = nanstd(permute(BinnedAnals(1,:,SalineSes),[3 2 1]));
PythonVars.Fig8A.Pcorr_t_se_list_saline_Subj = saline_performance_sd/sqrt(length(SalineSes));

%% Extract some variables, collapsed across sessions
%Extract variables for ketamine trials, collapsed across sessions
EviA_EviB_Choice_OnDrug = cat(1,SessionWiseDMs(KetamineSes).OnDrugTemporalWeightsDM);
EvidenceUnitsACollapsed_Ket = EviA_EviB_Choice_OnDrug(:,1:6);
EvidenceUnitsBCollapsed_Ket = EviA_EviB_Choice_OnDrug(:,7:12);
ChosenTargetCollapsed_Ket = EviA_EviB_Choice_OnDrug(:,13);
TrialErrorComp_Ket = cat(1,SessionWiseDMs(KetamineSes).CompTr_TrialError);

%Extract variables for saline trials, collapsed across sessions
EviA_EviB_Choice_Sal = cat(1,SessionWiseDMs(SalineSes).OnDrugTemporalWeightsDM);
EvidenceUnitsACollapsed_Sal = EviA_EviB_Choice_Sal(:,1:6);
EvidenceUnitsBCollapsed_Sal = EviA_EviB_Choice_Sal(:,7:12);
ChosenTargetCollapsed_Sal = EviA_EviB_Choice_Sal(:,13);
TrialErrorComp_Sal = cat(1,SessionWiseDMs(SalineSes).CompTr_TrialError);

%% Figure 8B,C: Psychometric functions
%Note, the same analyses run with individual subject data are presented as Fig8S1B, G in the paper

[~,PythonVars.Fig8BC.Ket.P_corr_Subj_list,PythonVars.Fig8BC.Ket.ErrBar_P_corr_Subj_list,PythonVars.Fig8BC.Ket.Psychometric_fit_params] = ...
    FinalPsychometricsInFunc_Final(EvidenceUnitsACollapsed_Ket,EvidenceUnitsBCollapsed_Ket,(2-ChosenTargetCollapsed_Ket)',...
    TrialErrorComp_Ket,'NarrowBroad');

[~,PythonVars.Fig8BC.Sal.P_corr_Subj_list,PythonVars.Fig8BC.Sal.ErrBar_P_corr_Subj_list,PythonVars.Fig8BC.Sal.Psychometric_fit_params] = ...
    FinalPsychometricsInFunc_Final(EvidenceUnitsACollapsed_Sal,EvidenceUnitsBCollapsed_Sal,(2-ChosenTargetCollapsed_Sal)',...
    TrialErrorComp_Sal,'NarrowBroad');

%% Analyses of Pro-variance Bias under Ketamine and Saline: Figure 8D,E,F.
%Note, the same analyses run with individual subject data are presented as Fig8S1C, H in the paper
%Note, the analyses which control for lapsing, Figure 8-figure supplement2A, D, are reproduced here also

if sum(strcmp(AnalysisList,'PVB'))>0
    
    [PythonVars.Fig8def] = ...
        PVB_drugdayanalysis(KetamineSes,SalineSes,SessionWiseDMs,...
        BootStrapNo,noPermutations,DirToSaveBootstraps,LoadResultsFromOldHypothesisTests);
    
end

%% Analysis of Subject Decision Strategy (Mean/Max/Min/First/Last Regression)
%Note, this analysis is included separated by subjects, as Figure 8-figure supplement 1D, I
%Note, this analyses with control for lapsing, Figure 8-figure supplement2B, E, are reproduced here also

if sum(strcmp(AnalysisList,'StratReg'))>0
    
    [PythonVars.Fig8Supp1di,~] = StratDrugFunctionF(noPermutations,BootStrapNo,EviA_EviB_Choice_Sal,EvidenceUnitsACollapsed_Ket,...
        EvidenceUnitsBCollapsed_Ket,ChosenTargetCollapsed_Ket,DirToSaveBootstraps,LoadResultsFromOldHypothesisTests);
    
end
%% Analysis of Subject's Temporal weights (Psychophysical kernel aka. PK)
%Note, this analysis is included collapsed across subjects, as Figure8G
%Note, this analysis is included separated by subjects, as Figure 8-figure supplement 1E, J
%Note, this analyses with control for lapsing, Figure 8-figure supplement2C, F, are reproduced here also
if sum(strcmp(AnalysisList,'PK'))>0
    
    [PythonVars.Fig8g,~] = PKDrugF(noPermutations,BootStrapNo,ChosenTargetCollapsed_Ket,EvidenceUnitsACollapsed_Ket,...
        EvidenceUnitsBCollapsed_Ket,EviA_EviB_Choice_Sal,DirToSaveBootstraps,LoadResultsFromOldHypothesisTests);
    
end
%% Figure 8-figure supplement 3. Time course of ketamine?s influence on pro-variance bias.
if sum(strcmp(AnalysisList,'SlidingPVB'))>0
    
    [PythonVars.Fig8Supp3] =...
        SlidingPVBIndexF(OutputDMHere,SalineSes,KetamineSes,noPermutations,DirToSaveBootstraps,LoadResultsFromOldHypothesisTests);
    
end

 








