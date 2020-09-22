function [PythonOutputs,StratReg_RevisionFunctionData] = StratDrugFunctionF(PermutationNo,BootStrapNo,EviA_EviB_Choice_Sal,EvidenceUnitsACollapsed_Ket,...
    EvidenceUnitsBCollapsed_Ket,ChosenTargetCollapsed_Ket,DirToSaveBootstraps,TryToLoadOldPermTests)

%% Try to load the old tests.
if TryToLoadOldPermTests
    cd(DirToSaveBootstraps);
    files=dir(fullfile(DirToSaveBootstraps,'StratReg_RevisionFunctionData*.mat'));
    load(files(end).name);
    RunNewPermTests = 0;
else
    RunNewPermTests = 1;
end
%%
if RunNewPermTests
    %% Run the strategy analyses including the lapse term - equation 10 (see Methods)
    [~,CollationOfBetas_Strat(:,:,1),bestPModelSaline,BootParamSave{1}] ...
        = RegressionToDetermineSubjStrategy(EviA_EviB_Choice_Sal(:,1:6),...
        EviA_EviB_Choice_Sal(:,7:12)...
        ,EviA_EviB_Choice_Sal(:,13)'==1,BootStrapNo);
    
    [~,CollationOfBetas_Strat(:,:,2),bestPModelKetamine,BootParamSave{2}] = ...
        RegressionToDetermineSubjStrategy( EvidenceUnitsACollapsed_Ket,...
        EvidenceUnitsBCollapsed_Ket...
        ,ChosenTargetCollapsed_Ket'==1,BootStrapNo);
    
    %% Run permutation tests (to compare saline and ketamine parameters)
    
    Ysaline = EviA_EviB_Choice_Sal(:,13)'==1;
    YKetamine = ChosenTargetCollapsed_Ket'==1;
    
    SalineDM = [ones(length(EviA_EviB_Choice_Sal),1) EviA_EviB_Choice_Sal(:,1) EviA_EviB_Choice_Sal(:,6)  mean(EviA_EviB_Choice_Sal(:,1:6),2)...
        max(EviA_EviB_Choice_Sal(:,1:6),[],2) min(EviA_EviB_Choice_Sal(:,1:6),[],2)...
        EviA_EviB_Choice_Sal(:,7) EviA_EviB_Choice_Sal(:,12) mean(EviA_EviB_Choice_Sal(:,7:12),2)...
        max(EviA_EviB_Choice_Sal(:,7:12),[],2) min(EviA_EviB_Choice_Sal(:,7:12),[],2)];
    
    EviA_EviB_Choice_Ket = cat(2,EvidenceUnitsACollapsed_Ket,EvidenceUnitsBCollapsed_Ket,ChosenTargetCollapsed_Ket);
    KetamineDM = [ones(length(EviA_EviB_Choice_Ket),1) EviA_EviB_Choice_Ket(:,1) EviA_EviB_Choice_Ket(:,6)  mean(EviA_EviB_Choice_Ket(:,1:6),2)...
        max(EviA_EviB_Choice_Ket(:,1:6),[],2) min(EviA_EviB_Choice_Ket(:,1:6),[],2)...
        EviA_EviB_Choice_Ket(:,7) EviA_EviB_Choice_Ket(:,12) mean(EviA_EviB_Choice_Ket(:,7:12),2)...
        max(EviA_EviB_Choice_Ket(:,7:12),[],2) min(EviA_EviB_Choice_Ket(:,7:12),[],2)];
    
    [pValsKetvSal] = ...
        PermTestsWithLapseFitting(Ysaline, ...
        YKetamine,PermutationNo,SalineDM,KetamineDM);
    %% Outputs organised and saved
    StratReg_RevisionFunctionData.CollationOfBetas = CollationOfBetas_Strat;
    StratReg_RevisionFunctionData.bestPModelSaline = bestPModelSaline;
    StratReg_RevisionFunctionData.bestPModelKetamine = bestPModelKetamine;
    StratReg_RevisionFunctionData.BootParamSave = BootParamSave;
    StratReg_RevisionFunctionData.pValsKetvSal = pValsKetvSal;
    
    FN = ['StratReg_RevisionFunctionData' datestr(datetime,'dd mmm yyyy')];
    save(fullfile(DirToSaveBootstraps,FN),'StratReg_RevisionFunctionData');
    
end

%% Run the logistic regression analysis
RegStatsSaline = RegressionToDetermineSubjStrategy(EviA_EviB_Choice_Sal(:,1:6),...
    EviA_EviB_Choice_Sal(:,7:12)...
    ,EviA_EviB_Choice_Sal(:,13)'==1,[]);

RegStatsKetamine = RegressionToDetermineSubjStrategy( EvidenceUnitsACollapsed_Ket,...
    EvidenceUnitsBCollapsed_Ket...
    ,ChosenTargetCollapsed_Ket'==1,[]);
%% Organise the output for python
%From the logistic equation (e.g. Fig8S1D, I)
PythonOutputs.SalineWeights_Logistic = RegStatsSaline.BetaWeights;
PythonOutputs.KetamineWeights_Logistic = RegStatsKetamine.BetaWeights;

PythonOutputs.SalineErs_Logistic = ...
    [RegStatsSaline.BetaWeights-RegStatsSaline.SeOfBetaWeights RegStatsSaline.BetaWeights+RegStatsSaline.SeOfBetaWeights];
PythonOutputs.KetamineErs_Logistic = ...
    [RegStatsKetamine.BetaWeights-RegStatsKetamine.SeOfBetaWeights RegStatsKetamine.BetaWeights+RegStatsKetamine.SeOfBetaWeights];

%From the lapse equation (e.g. Fig8S2B, E)
PythonOutputs.SalineWeights_LapseModel =StratReg_RevisionFunctionData.CollationOfBetas(:,2,1);
PythonOutputs.KetamineWeights_LapseModel =StratReg_RevisionFunctionData.CollationOfBetas(:,2,2);

PythonOutputs.SalineErs_LapseModel = prctile(StratReg_RevisionFunctionData.BootParamSave{1}{1}{1},[2.5 97.5]);
PythonOutputs.KetamineErs_LapseModel = prctile(StratReg_RevisionFunctionData.BootParamSave{2}{1}{1},[2.5 97.5]);

KetamineBootstraps = StratReg_RevisionFunctionData.BootParamSave{2}{1}{1};
SalineBootstraps = StratReg_RevisionFunctionData.BootParamSave{1}{1}{1};

PythonOutputs.KetamineErs_LapseModel_CollapsedAcrossLR = prctile(cat(1,KetamineBootstraps(:,[2:6]),-KetamineBootstraps(:,[7:11])),[2.5 97.5]);
PythonOutputs.SalineErs_LapseModel_CollapsedAcrossLR = prctile(cat(1,SalineBootstraps(:,[2:6]),-SalineBootstraps(:,[7:11])),[2.5 97.5]);
%First row is 2.5th percentile; bottom row is 97.5th percentile
%Columns: First sample (1); Last sample (2); Mean evidence (3); Max evidence (4); Min evidence (5)
%% P VALUES OUT FOR PLOTTING
PythonOutputs.pValsKetvSal = StratReg_RevisionFunctionData.pValsKetvSal;
PythonOutputs.pValsKetvSal_CollapsedAcrossLR  = mean(reshape(PythonOutputs.pValsKetvSal(2:11),[5 2]),2);
%Average the p-value across the Left and Right regressors.
%Columns: First sample (1); Last sample (2); Mean evidence (3); Max evidence (4); Min evidence (5)
end

