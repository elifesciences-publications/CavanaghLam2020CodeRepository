function [PythonVars] = StandardSessions_model_lapse_downstream(model_condition,MainDirectory, model_folder, flag_lapse_downstream,lapse_rate,n_lapse_shuff);
%% Editable Variables:
n_kcv_runs = 2; % How many cross-validation runs to run? Note, in the paper, this was set to 100. It has been set to 10 here to decrease computing time. 
%% Other directory information
CodesFileLocations =     fullfile(MainDirectory,'CodeFiles'); addpath(CodesFileLocations);
DataFileLocations = fullfile(MainDirectory,'DataFiles\Model_Data\');

%% Load in the data files corresponding to the model condition. 
load(strcat(DataFileLocations, model_folder, '\', model_condition, '_model_data.mat'))

%% Collapse the variables across sessions
EvidenceUnitsACollapsed = cat(1,model_data.EvidenceUnitsACollapsed); %Evidence values for the Left Option on Completed trials
EvidenceUnitsBCollapsed = cat(1,model_data.EvidenceUnitsBCollapsed); %Evidence values for the Right Option on Completed trials
ChosenTargetCollapsed = cat(2,model_data.ChosenTargetCollapsed); %Chosen target (1=left; 2=right) on Completed trials

%% Incoporate lapse downstream of choice data, if flag_lapse_downstream==1.
%%% Note that this only incoporates one instance of shuffling. The process should be repeated multiple times with the average taken.

if flag_lapse_downstream==1
    n_to_shuff = round(length(ChosenTargetCollapsed)*2*lapse_rate);         % Number of trials to shuffle as lapse effect.
        
    % First run once.
    ChosenTargetCollapsed_lapse_shuff = ChosenTargetCollapsed;
    lapsed_trials = sort(randsample(length(ChosenTargetCollapsed),n_to_shuff))';
    ChosenTargetCollapsed_lapse_shuff(lapsed_trials) = randi(2,length(lapsed_trials),1);
    
    %% Figures

    % %% Figure 5c,d: Narrow-Broad trials (control model).
    % %Note, the same analyses run with individual subject data are presented as Fig3S1G-J in the paper
    % [NarrowBroadTrialsCOL(:,:),NarrowBroadTrialsCOL_Errs(:,:),...
    %     StatsOutputForPaperNarrowBroadTr] =  AnalNarrowBroadTrials_final(TrialType...
    %     ,ChosenTargetCollapsed_lapse_shuff); % Probability to choose in narrow-correct, broad-correct, and ambiguous cases.
    % PythonVars.Fig3bc.ENB_bars_Subj_non_drug = [NarrowBroadTrialsCOL(end) NarrowBroadTrialsCOL(1:2)]; %Choice probabilities
    % PythonVars.Fig3bc.ENB_bars_err_Subj_non_drug = [NarrowBroadTrialsCOL_Errs(end) NarrowBroadTrialsCOL_Errs(1:2)]; %Standard errors
    % PythonVars.Fig3bc.StatsOutputForPaperNarrowBroadTr = StatsOutputForPaperNarrowBroadTr; %Stats tests

    %% Figure 5e: Non-drug sessions: Pro-variance effects displayed using a psychometric function
    %Note, the same analyses run with individual subject data are presented as Fig4S1C;Fig4S1F in the paper
    [~,P_corr_Subj_list,ErrBar_P_corr_Subj_list,PythonVars.Fig4c.Psychometric_fit_paramsFig4] = ...
        FinalPsychometricsInFunc_Model(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,ChosenTargetCollapsed_lapse_shuff,'NarrowBroad');
    PythonVars.Fig4c.P_corr_Subj_list = P_corr_Subj_list/n_lapse_shuff; %Choice probabilities for each bin
    PythonVars.Fig4c.ErrBar_P_corr_Subj_list = ErrBar_P_corr_Subj_list/n_lapse_shuff; %Standard errors for each bin

    %% Figure 5f - Regression for pro-variance effect
    %Note, the same analyses run with individual subject data are presented as Fig4S1D;Fig4S1G in the paper
    [Betas,TStatOut,POut,ErrCollapsed] = ...
        PvbRegressionAnalysis(100*EvidenceUnitsACollapsed...
        ,100*EvidenceUnitsBCollapsed...
        ,ChosenTargetCollapsed_lapse_shuff);

    PythonVars.Fig4d.Reg_bars_Subj_non_drug = Betas'/n_lapse_shuff; %Beta weights 
    PythonVars.Fig4d.Reg_bars_err_Subj_non_drug = ErrCollapsed'/n_lapse_shuff; %Standard errors
    PythonVars.Fig4d.PVB_T_Stats = TStatOut/n_lapse_shuff; %Stats reporting
    PythonVars.Fig4d.PVB_PVals = POut/n_lapse_shuff; %Stats reporting



    %% Figure 5g - Psychophysical kernels (temporal weightings)  (control model).
    %Note, in the paper Fig2cd, this data is analysed separately for each subject.  
    tmp_PK_GLM = [];                                                          
    tmp_PK_GLM(:,9) = transpose(ChosenTargetCollapsed_lapse_shuff); %Store the subjects' choices in the 9th column of this variable
    tmp_PK_GLM(tmp_PK_GLM==2) = 0; %Choices are assigned as 1 (Chose Left); 0 (Chose Right)
    tmp_PK_GLM(:,1:8) = 100.*(EvidenceUnitsACollapsed - EvidenceUnitsBCollapsed); %Difference between evidence on the left and right at each timestep
    [NonDrugDayPairedData] = RunPkAnalysis(tmp_PK_GLM,[]); 

    PythonVars.Fig2.PK_Subj_nondrug =  NonDrugDayPairedData(:,1)'/n_lapse_shuff; %Beta weights for each time step
    PythonVars.Fig2.PK_Subj_nondrug_errbar =  NonDrugDayPairedData(:,2)'/n_lapse_shuff; %Standard error for each time step


    %% Figure 5 - supplement 1: Extended Regression Analysis.
    [RegrOutputs] = RegressionToDetermineSubjStrategy( 100*EvidenceUnitsACollapsed,...
        100*EvidenceUnitsBCollapsed...
        ,ChosenTargetCollapsed_lapse_shuff==1,[]);

    PythonVars.Fig4Sup1.Reg_values_Subj_nondrug = RegrOutputs.BetaWeights/n_lapse_shuff;
    PythonVars.Fig4Sup1.Reg_bars_err_Subj_non_drug = RegrOutputs.SeOfBetaWeights/n_lapse_shuff;

%         %% Supplementary Tables 1-3: Model comparison results
%         PythonVars.SuppTableOutput = Fig4ModelCompare_Final_F(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,...
%             ChosenTargetCollapsed_lapse_shuff,...
%             n_kcv_runs);


    % Then loop
    for i_lapse_shuff = 2:n_lapse_shuff
        ChosenTargetCollapsed_lapse_shuff = ChosenTargetCollapsed;
        lapsed_trials = sort(randsample(length(ChosenTargetCollapsed),n_to_shuff))';
        ChosenTargetCollapsed_lapse_shuff(lapsed_trials) = randi(2,length(lapsed_trials),1);
        
        %% Figures

        % %% Figure 5c,d: Narrow-Broad trials (control model).
        % %Note, the same analyses run with individual subject data are presented as Fig3S1G-J in the paper
        % [NarrowBroadTrialsCOL(:,:),NarrowBroadTrialsCOL_Errs(:,:),...
        %     StatsOutputForPaperNarrowBroadTr] =  AnalNarrowBroadTrials_final(TrialType...
        %     ,ChosenTargetCollapsed_lapse_shuff); % Probability to choose in narrow-correct, broad-correct, and ambiguous cases.
        % PythonVars.Fig3bc.ENB_bars_Subj_non_drug = [NarrowBroadTrialsCOL(end) NarrowBroadTrialsCOL(1:2)]; %Choice probabilities
        % PythonVars.Fig3bc.ENB_bars_err_Subj_non_drug = [NarrowBroadTrialsCOL_Errs(end) NarrowBroadTrialsCOL_Errs(1:2)]; %Standard errors
        % PythonVars.Fig3bc.StatsOutputForPaperNarrowBroadTr = StatsOutputForPaperNarrowBroadTr; %Stats tests

        %% Figure 5e: Non-drug sessions: Pro-variance effects displayed using a psychometric function
        %Note, the same analyses run with individual subject data are presented as Fig4S1C;Fig4S1F in the paper
        [~,P_corr_Subj_list,ErrBar_P_corr_Subj_list,PythonVars.Fig4c.Psychometric_fit_paramsFig4] = ...
            FinalPsychometricsInFunc_Model(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,ChosenTargetCollapsed_lapse_shuff,'NarrowBroad');
        PythonVars.Fig4c.P_corr_Subj_list = PythonVars.Fig4c.P_corr_Subj_list + P_corr_Subj_list/n_lapse_shuff; %Choice probabilities for each bin
        PythonVars.Fig4c.ErrBar_P_corr_Subj_list = PythonVars.Fig4c.ErrBar_P_corr_Subj_list + ErrBar_P_corr_Subj_list/n_lapse_shuff; %Standard errors for each bin

        %% Figure 5f - Regression for pro-variance effect
        %Note, the same analyses run with individual subject data are presented as Fig4S1D;Fig4S1G in the paper
        [Betas,TStatOut,POut,ErrCollapsed] = ...
            PvbRegressionAnalysis(100*EvidenceUnitsACollapsed...
            ,100*EvidenceUnitsBCollapsed...
            ,ChosenTargetCollapsed_lapse_shuff);

        PythonVars.Fig4d.Reg_bars_Subj_non_drug = PythonVars.Fig4d.Reg_bars_Subj_non_drug + Betas'/n_lapse_shuff; %Beta weights 
        PythonVars.Fig4d.Reg_bars_err_Subj_non_drug = PythonVars.Fig4d.Reg_bars_err_Subj_non_drug + ErrCollapsed'/n_lapse_shuff; %Standard errors
        PythonVars.Fig4d.PVB_T_Stats = PythonVars.Fig4d.PVB_T_Stats + TStatOut/n_lapse_shuff; %Stats reporting
        PythonVars.Fig4d.PVB_PVals = PythonVars.Fig4d.PVB_PVals + POut/n_lapse_shuff; %Stats reporting



        %% Figure 5g - Psychophysical kernels (temporal weightings)  (control model).
        %Note, in the paper Fig2cd, this data is analysed separately for each subject.  
        tmp_PK_GLM = [];                                                          
        tmp_PK_GLM(:,9) = transpose(ChosenTargetCollapsed_lapse_shuff); %Store the subjects' choices in the 9th column of this variable
        tmp_PK_GLM(tmp_PK_GLM==2) = 0; %Choices are assigned as 1 (Chose Left); 0 (Chose Right)
        tmp_PK_GLM(:,1:8) = 100.*(EvidenceUnitsACollapsed - EvidenceUnitsBCollapsed); %Difference between evidence on the left and right at each timestep
        [NonDrugDayPairedData] = RunPkAnalysis(tmp_PK_GLM,[]); 

        PythonVars.Fig2.PK_Subj_nondrug =  PythonVars.Fig2.PK_Subj_nondrug + NonDrugDayPairedData(:,1)'/n_lapse_shuff; %Beta weights for each time step
        PythonVars.Fig2.PK_Subj_nondrug_errbar =  PythonVars.Fig2.PK_Subj_nondrug_errbar + NonDrugDayPairedData(:,2)'/n_lapse_shuff; %Standard error for each time step


        %% Figure 5 - supplement 1: Extended Regression Analysis.
        [RegrOutputs] = RegressionToDetermineSubjStrategy( 100*EvidenceUnitsACollapsed,...
            100*EvidenceUnitsBCollapsed...
            ,ChosenTargetCollapsed_lapse_shuff==1,[]);

        PythonVars.Fig4Sup1.Reg_values_Subj_nondrug = PythonVars.Fig4Sup1.Reg_values_Subj_nondrug + RegrOutputs.BetaWeights/n_lapse_shuff;
        PythonVars.Fig4Sup1.Reg_bars_err_Subj_non_drug = PythonVars.Fig4Sup1.Reg_bars_err_Subj_non_drug + RegrOutputs.SeOfBetaWeights/n_lapse_shuff;

%         %% Supplementary Tables 1-3: Model comparison results
%         PythonVars.SuppTableOutput = Fig4ModelCompare_Final_F(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,...
%             ChosenTargetCollapsed_lapse_shuff,...
%             n_kcv_runs);


    end


elseif flag_lapse_downstream==0
    %% Figures

    % %% Figure 5c,d: Narrow-Broad trials (control model).
    % %Note, the same analyses run with individual subject data are presented as Fig3S1G-J in the paper
    % [NarrowBroadTrialsCOL(:,:),NarrowBroadTrialsCOL_Errs(:,:),...
    %     StatsOutputForPaperNarrowBroadTr] =  AnalNarrowBroadTrials_final(TrialType...
    %     ,ChosenTargetCollapsed); % Probability to choose in narrow-correct, broad-correct, and ambiguous cases.
    % PythonVars.Fig3bc.ENB_bars_Subj_non_drug = [NarrowBroadTrialsCOL(end) NarrowBroadTrialsCOL(1:2)]; %Choice probabilities
    % PythonVars.Fig3bc.ENB_bars_err_Subj_non_drug = [NarrowBroadTrialsCOL_Errs(end) NarrowBroadTrialsCOL_Errs(1:2)]; %Standard errors
    % PythonVars.Fig3bc.StatsOutputForPaperNarrowBroadTr = StatsOutputForPaperNarrowBroadTr; %Stats tests

    %% Figure 5e: Non-drug sessions: Pro-variance effects displayed using a psychometric function
    %Note, the same analyses run with individual subject data are presented as Fig4S1C;Fig4S1F in the paper
    [~,P_corr_Subj_list,ErrBar_P_corr_Subj_list,PythonVars.Fig4c.Psychometric_fit_paramsFig4] = ...
        FinalPsychometricsInFunc_Model(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,ChosenTargetCollapsed,'NarrowBroad');
    PythonVars.Fig4c.P_corr_Subj_list = P_corr_Subj_list; %Choice probabilities for each bin
    PythonVars.Fig4c.ErrBar_P_corr_Subj_list = ErrBar_P_corr_Subj_list; %Standard errors for each bin

    %% Figure 5f - Regression for pro-variance effect
    %Note, the same analyses run with individual subject data are presented as Fig4S1D;Fig4S1G in the paper
    [Betas,TStatOut,POut,ErrCollapsed] = ...
        PvbRegressionAnalysis(100*EvidenceUnitsACollapsed...
        ,100*EvidenceUnitsBCollapsed...
        ,ChosenTargetCollapsed);

    PythonVars.Fig4d.Reg_bars_Subj_non_drug = Betas'; %Beta weights 
    PythonVars.Fig4d.Reg_bars_err_Subj_non_drug = ErrCollapsed'; %Standard errors
    PythonVars.Fig4d.PVB_T_Stats = TStatOut; %Stats reporting
    PythonVars.Fig4d.PVB_PVals = POut; %Stats reporting



    %% Figure 5g - Psychophysical kernels (temporal weightings)  (control model).
    %Note, in the paper Fig2cd, this data is analysed separately for each subject.  
    tmp_PK_GLM = [];                                                          
    tmp_PK_GLM(:,9) = transpose(ChosenTargetCollapsed); %Store the subjects' choices in the 9th column of this variable
    tmp_PK_GLM(tmp_PK_GLM==2) = 0; %Choices are assigned as 1 (Chose Left); 0 (Chose Right)
    tmp_PK_GLM(:,1:8) = 100.*(EvidenceUnitsACollapsed - EvidenceUnitsBCollapsed); %Difference between evidence on the left and right at each timestep
    [NonDrugDayPairedData] = RunPkAnalysis(tmp_PK_GLM,[]); 

    PythonVars.Fig2.PK_Subj_nondrug =  NonDrugDayPairedData(:,1)'; %Beta weights for each time step
    PythonVars.Fig2.PK_Subj_nondrug_errbar =  NonDrugDayPairedData(:,2)'; %Standard error for each time step


    %% Figure 5 - supplement 1: Extended Regression Analysis.
    [RegrOutputs] = RegressionToDetermineSubjStrategy( 100*EvidenceUnitsACollapsed,...
        100*EvidenceUnitsBCollapsed...
        ,ChosenTargetCollapsed==1,[]);

    PythonVars.Fig4Sup1.Reg_values_Subj_nondrug = RegrOutputs.BetaWeights;
    PythonVars.Fig4Sup1.Reg_bars_err_Subj_non_drug = RegrOutputs.SeOfBetaWeights;

    %% Supplementary Tables 1-3: Model comparison results
    PythonVars.SuppTableOutput = Fig4ModelCompare_Final_F(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,...
        ChosenTargetCollapsed,...
        n_kcv_runs);
end
end
