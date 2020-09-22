function [PythonOutputs,PK_RevisionFunctionData] = PKDrugF(NoPerms,BootStrapNo,ChosenTargetCollapsed_Ket,EvidenceUnitsACollapsed_Ket,...
    EvidenceUnitsBCollapsed_Ket,EviA_EviB_Choice_Sal,DirToSaveBootstraps,TryToLoadOldPermTests)
%% Try to load the old tests.
if TryToLoadOldPermTests
    cd(DirToSaveBootstraps);
    files=dir(fullfile(DirToSaveBootstraps,'PkData*.mat'));
    load(files(end).name);
    RunNewPermTests = 0;
else
    RunNewPermTests = 1;
end
 
%%
if RunNewPermTests
    %Calculate the temporal weights for the ketamine data (equation 8)
    tmp_PK_GLM = [];
    tmp_PK_GLM(:,7) = transpose(ChosenTargetCollapsed_Ket);
    tmp_PK_GLM(:,1:6) = EvidenceUnitsACollapsed_Ket - EvidenceUnitsBCollapsed_Ket;
    [~,CollationOfBetas(:,:,2),BootstrapedData{2},bestPModel{2}]...
        = RunPkAnalysis(tmp_PK_GLM,BootStrapNo);
    
    %Calculate the temporal weights for the saline data
    tmp_PK_GLM = [];
    tmp_PK_GLM(:,7) = transpose(EviA_EviB_Choice_Sal(:,13));
    tmp_PK_GLM(:,1:6) = EviA_EviB_Choice_Sal(:,1:6) - EviA_EviB_Choice_Sal(:,7:12);
    [~,CollationOfBetas(:,:,1),BootstrapedData{1},bestPModel{1}]...
        = RunPkAnalysis(tmp_PK_GLM,BootStrapNo);
    %% Run permutation tests to compare ketamine and saline
    YSaline= transpose(EviA_EviB_Choice_Sal(:,13));
    YKetamine = transpose(ChosenTargetCollapsed_Ket);
    SalineDM =   [ ones(length(EviA_EviB_Choice_Sal),1) EviA_EviB_Choice_Sal(:,1:6) - EviA_EviB_Choice_Sal(:,7:12)];
    KetamineDM = [ones(length(EvidenceUnitsACollapsed_Ket),1) EvidenceUnitsACollapsed_Ket - EvidenceUnitsBCollapsed_Ket];
    [pValsKetvSal] = ...
        PermTestsWithLapseFitting(YSaline, ...
        YKetamine,NoPerms,SalineDM,KetamineDM);
    
    %%
    PK_RevisionFunctionData.CollationOfBetas = CollationOfBetas(:,2,:);
    PK_RevisionFunctionData.bestPModelSaline = bestPModel{2};
    PK_RevisionFunctionData.bestPModelKetamine = bestPModel{1};
    PK_RevisionFunctionData.BootParamSave = BootstrapedData;
    PK_RevisionFunctionData.pValsKetvSal = pValsKetvSal;
     
    FN = ['PkData' datestr(datetime,'dd mmm yyyy')];
    save(fullfile(DirToSaveBootstraps,FN),'PK_RevisionFunctionData');
    
end
%% Run the logistic analyses alone
%Calculate the temporal weights for the ketamine data
tmp_PK_GLM = [];
tmp_PK_GLM(:,7) = transpose(ChosenTargetCollapsed_Ket);
tmp_PK_GLM(:,1:6) = EvidenceUnitsACollapsed_Ket - EvidenceUnitsBCollapsed_Ket;
[KetaminePairedData]...
    = RunPkAnalysis(tmp_PK_GLM,[]);

%Calculate the temporal weights for the saline data
tmp_PK_GLM = [];
tmp_PK_GLM(:,7) = transpose(EviA_EviB_Choice_Sal(:,13));
tmp_PK_GLM(:,1:6) = EviA_EviB_Choice_Sal(:,1:6) - EviA_EviB_Choice_Sal(:,7:12);
[SalinePairedData]...
    = RunPkAnalysis(tmp_PK_GLM,[]);

%% Organise the variables to be output into python

%From the logistic equation (e.g. Fig8G; Fig8S1E, J)
PythonOutputs.SalinePK_LogisticModel = SalinePairedData(:,1);
PythonOutputs.KetaminePK_LogisticModel= KetaminePairedData(:,1);

PythonOutputs.SalineErs_LogisticModel = [SalinePairedData(:,1)-SalinePairedData(:,2) SalinePairedData(:,1)+SalinePairedData(:,2)];
PythonOutputs.KetamineErs_LogisticModel = [KetaminePairedData(:,1)-KetaminePairedData(:,2) KetaminePairedData(:,1)+KetaminePairedData(:,2)];

%From the lapse equation (e.g. Fig8S2C, F)
PythonOutputs.SalinePK_LapseModel = PK_RevisionFunctionData.CollationOfBetas(2:end,:,1);
PythonOutputs.KetaminePK_LapseModel = PK_RevisionFunctionData.CollationOfBetas(2:end,:,2);

PythonOutputs.SalineErs_LapseModel = prctile(PK_RevisionFunctionData.BootParamSave{1}{1}{1}(:,2:7),[2.5 97.5]);
PythonOutputs.KetamineErs_LapseModel = prctile(PK_RevisionFunctionData.BootParamSave{2}{1}{1}(:,2:7),[2.5 97.5]);

%Permutation test of the parameters from the lapse equation
PythonOutputs.KetvSalinePermutationTest = PK_RevisionFunctionData.pValsKetvSal(2:7);

end

