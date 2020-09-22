function [OutputForPython] = PVB_drugdayanalysis(KetSes,...
    SalineSes,SessionWiseDMs,BootStrapNo,noPermutations,DirToSaveBootstraps,TryToLoadOldPermTests)
%Explanation of outputs:
%KeyOutputsToSave.CollationOfBetas (Rows - beta: constant,Mu,Std); (columns-models: logistic, lapse term); (pages - saline; ketamine)
%KeyOutputsToSave.bestPModelSaline (Structure - fitted parameters of the lapse model to all the saline data)
%KeyOutputsToSave.bestPModelKetamine (Structure - fitted parameters of the lapse model to all the ketamine data)
%KeyOutputsToSave.BootParamSave (Structure)(First level is {1} = saline; {2} = ketamine)(Second level is the different models {1}Lapse)(Third level is different info {1}Bootstraps (iterations x params) 
%% Load the old tests, if applicable
if TryToLoadOldPermTests
    cd(DirToSaveBootstraps); %Change folder to where the data should be stored
    files=dir(fullfile(DirToSaveBootstraps,'PVB_RevisionFunctionData*.mat'));
    load(files(end).name);
    RunNewPermTests = 0;
else
    RunNewPermTests = 1;
end

%% Run the PVB analyses with the logistic equation 5 (see Methods)
%SalineSessions:
FullDM = cat(1,SessionWiseDMs(SalineSes).ProVar9RegModel); %Load the stored design matrix and collapse across sessions
OriginalDM = FullDM(FullDM(:,2)==1,[2 5 8]); %Limit the design matrix to only 'OnDrug' trials
YVar_saline = FullDM(FullDM(:,2)==1,[10]); %Limit the y-variable, chose left, to 'OnDrug' trials
[~,~,GLMstats_saline] = glmfit(OriginalDM,YVar_saline,'binomial','link','logit','constant','off');

%KetamineSessions:
FullDM = cat(1,SessionWiseDMs(KetSes).ProVar9RegModel);
OriginalDM = FullDM(FullDM(:,2)==1,[2 5 8]);
YVar_Ketamine = FullDM(FullDM(:,2)==1,[10]);
[~,~,GLMstats_ketamine] = glmfit(OriginalDM,YVar_Ketamine,'binomial','link','logit','constant','off');

%% Permutation tests for equation 5
n_perm_here = 1000;     %Note in the paper, this was set to 1,000,000. It has been reduced here to speed up computation time
% Note that minimum p-value is ~1/n_perm.
PVB_Ketamine = GLMstats_ketamine.beta(3)/GLMstats_ketamine.beta(2); %Provariance bias index (PVB) for ketamine data
PVB_Saline = GLMstats_saline.beta(3)/GLMstats_saline.beta(2);
PVB_Diff = PVB_Ketamine - PVB_Saline; %Difference in PVB between ketamine and saline conditions
Mean_Diff = GLMstats_ketamine.beta(2)-GLMstats_saline.beta(2); %Difference in the Mean evidence regression coefficient
SD_Diff =GLMstats_ketamine.beta(3)-GLMstats_saline.beta(3); %Difference in the Std evidence regression coefficient

EviA_EviB_Choice_Saline = cat(1,SessionWiseDMs(SalineSes).OnDrugTemporalWeightsDM);
EviA_EviB_Choice_OnDrug = cat(1,SessionWiseDMs(KetSes).OnDrugTemporalWeightsDM);

%Run the permutation test within this function
[p_perm_mean_beta, p_perm_SD_beta, p_perm_pvbindex] = PermTestBetasRatio_Aug2020sc(YVar_saline', EviA_EviB_Choice_Saline(:,1:6), EviA_EviB_Choice_Saline(:,7:12),...
    YVar_Ketamine', EviA_EviB_Choice_OnDrug(:,1:6), EviA_EviB_Choice_OnDrug(:,7:12),...
    Mean_Diff,SD_Diff,PVB_Diff, n_perm_here);

ArOfPermResults = [p_perm_mean_beta p_perm_SD_beta p_perm_pvbindex];
for ii=1:3 %Loop across the 3 parameters
    if ArOfPermResults(ii)==0
        PermutationTextReportHere{ii} = [' < ' num2str(1/n_perm_here)];
    else
        PermutationTextReportHere{ii}  = [' = ' num2str(round(ArOfPermResults(ii),3,'significant'))];
    end
end

OutputForPython.Permutationtest_Logistic_MeanBeta = PermutationTextReportHere(1);
OutputForPython.Permutationtest_Logistic_SDbeta= PermutationTextReportHere(2);
OutputForPython.Permutationtest_Logistic_PVB = PermutationTextReportHere(3);

%% Run the PVB analyses including the lapse term - equation 9 (see Methods)
if RunNewPermTests
    %SalineSessions:
    FullDM = cat(1,SessionWiseDMs(SalineSes).ProVar9RegModel); %Load the stored design matrix and collapse across sessions
    OriginalDM_saline = FullDM(FullDM(:,2)==1,[2 5 8]); %Limit the design matrix to only 'OnDrug' trials
    YVarSaline = FullDM(FullDM(:,2)==1,[10]); %Limit the y-variable, chose left, to 'OnDrug' trials
    [CollationOfBetas(:,:,1),bestPModelSaline,BootParamSave{1}]= ...
        scFittingFunctionForLapsesF2(OriginalDM_saline,YVarSaline,BootStrapNo);
    
    %KetamineSessions:
    FullDM = cat(1,SessionWiseDMs(KetSes).ProVar9RegModel);
    OriginalDM_Ket = FullDM(FullDM(:,2)==1,[2 5 8]);
    YVarKetamine = FullDM(FullDM(:,2)==1,[10]);
    [CollationOfBetas(:,:,2),bestPModelKetamine,BootParamSave{2}]= ...
        scFittingFunctionForLapsesF2(OriginalDM_Ket,YVarKetamine,BootStrapNo);
    
    %Run permutation tests (to compare saline and ketamine parameters)
    [pValsKetvSal] = ...
        PermTestsWithLapseFitting_extend4pvb(YVarSaline', ...
        YVarKetamine',noPermutations,OriginalDM_saline,OriginalDM_Ket);
    
    %Gather the key information to save
    KeyOutputsToSave.CollationOfBetas = CollationOfBetas;
    KeyOutputsToSave.bestPModelSaline = bestPModelSaline;
    KeyOutputsToSave.bestPModelKetamine = bestPModelKetamine;
    KeyOutputsToSave.BootParamSave = BootParamSave;
    KeyOutputsToSave.pValsKetvSal = pValsKetvSal;
    
    FN = ['PVB_RevisionFunctionData' datestr(datetime,'dd mmm yyyy')];
    save(fullfile(DirToSaveBootstraps,FN),'KeyOutputsToSave');
    
end

%% Saving the output neatly
%From the logistic equation (e.g. Fig8D,E,F; Fig8S1C, H)
OutputForPython.Reg_bars_Subj_ketamine_logistic = GLMstats_ketamine.beta;
OutputForPython.Reg_bars_Subj_saline_logistic = GLMstats_saline.beta;

OutputForPython.Reg_bars_err_Subj_ketamine_logistic = [GLMstats_ketamine.beta'-GLMstats_ketamine.se'; GLMstats_ketamine.beta'+GLMstats_ketamine.se'];
OutputForPython.Reg_bars_err_Subj_saline_logistic = [GLMstats_saline.beta'-GLMstats_saline.se'; GLMstats_saline.beta'+GLMstats_saline.se'];

OutputForPython.PVB_Index_Subj_ketamine_logistic = GLMstats_ketamine.beta(3)/GLMstats_ketamine.beta(2);
OutputForPython.PVB_Index_Subj_saline_logistic = GLMstats_saline.beta(3)/GLMstats_saline.beta(2);

%From the lapse equation (e.g. Fig8S2A, D)
OutputForPython.Reg_bars_Subj_ketamine_lapse = KeyOutputsToSave.CollationOfBetas(:,2,2);
OutputForPython.Reg_bars_Subj_saline_lapse  = KeyOutputsToSave.CollationOfBetas(:,2,1);

SalineBootstrapDistribution = KeyOutputsToSave.BootParamSave{1}{1}{1};
KetamineBootstrapDistribution = KeyOutputsToSave.BootParamSave{2}{1}{1};

OutputForPython.Reg_bars_err_Subj_ketamine_lapse = prctile(KetamineBootstrapDistribution(:,1:3),[2.5 97.5]);
OutputForPython.Reg_bars_err_Subj_saline_lapse = prctile(SalineBootstrapDistribution(:,1:3),[2.5 97.5]);

OutputForPython.PVB_Index_Subj_ketamine_lapse = OutputForPython.Reg_bars_Subj_ketamine_lapse(3)/OutputForPython.Reg_bars_Subj_ketamine_lapse(2);
OutputForPython.PVB_Index_Subj_saline_lapse = OutputForPython.Reg_bars_Subj_saline_lapse(3)/OutputForPython.Reg_bars_Subj_saline_lapse(2);

OutputForPython.PVB_Index_err_Subj_ketamine_lapse =prctile(KetamineBootstrapDistribution(:,3)./KetamineBootstrapDistribution(:,2),[2.5 97.5]);
OutputForPython.PVB_Index_err_Subj_saline_lapse = prctile(SalineBootstrapDistribution(:,3)./SalineBootstrapDistribution(:,2),[2.5 97.5]);

%% Report results of comparison of lapse rates, for main text of results section:
OutputForPython.LapseResultsReporting = ['Lapse(Saline) = ' num2str(KeyOutputsToSave.bestPModelSaline{1}.YZERO) ...
    '; Lapse(Ketamine) = ' num2str(KeyOutputsToSave.bestPModelKetamine{1}.YZERO) ...
    '; Permutation test, p = ' num2str(KeyOutputsToSave.pValsKetvSal(4)) ];

%% Report results of permutations terms, comparing ketamine and saline parameters in the lapse equation, for Fig8S2 legend
Pts = [2 3 5]; %Parameters of interest (e.g. Mean Evidence; SD; PVB_

for ii=1:3 %Loop across the 3 parameters
    if KeyOutputsToSave.pValsKetvSal(Pts(ii))==0
        PermutationTextReport2{ii} = [' < ' num2str(1/10000)];
    else
        PermutationTextReport2{ii}  = [' = ' num2str(round(KeyOutputsToSave.pValsKetvSal(Pts(ii)),3,'significant'))];
    end
end

OutputForPython.Figure8sup2legendReport_ProseType1 = ...
    ['(Left) The coefficient for mean evidence, under injection of saline or ketamine. Ketamine significantly reduces the coefficient (permutation test, p' ...
    PermutationTextReport2{1} ') reflecting a drop in choice accuracy.'...
    ' (Middle) The coefficient for evidence standard deviation, under injection of saline or ketamine. Ketamine does not significantly reduce the coefficient (permutation test, p'...
    PermutationTextReport2{2} ...
    '). (Right) Ketamine increases the PVB index (permutation test, p' PermutationTextReport2{3} ...
    ') consistent with the model prediction of the lowered E/I circuit.'];

OutputForPython.Figure8sup2legendReport_ProseType2 = ...
    ['Ketamine significantly reduces the coefficient for mean evidence (permutation test, p ' PermutationTextReport2{1}...
    '), does not significantly reduce the coefficient for evidence standard deviation (permutation test, p' PermutationTextReport2{2}...
    '), and significantly increases the PVB index (permutation test, p' PermutationTextReport2{3} '.'];

end

