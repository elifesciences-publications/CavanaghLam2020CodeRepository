function  Output = Fig4ModelCompare_Final_F(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,ChosenTargetCollapsed,n_kcv_runs)
%% Design matricies for 9 different models, based off of equation 6 (see Methods), which are compared to produce Supplementary Tables 1-3
%The comments above each model indicate which predictors are included. 
  
 % dm: first, last, mean, max, min
dm_no_sd = [EvidenceUnitsACollapsed(:,1) EvidenceUnitsACollapsed(:,end)  mean(EvidenceUnitsACollapsed(:,1:end),2)...
    max(EvidenceUnitsACollapsed(:,1:end),[],2) min(EvidenceUnitsACollapsed(:,1:end),[],2)...
    EvidenceUnitsBCollapsed(:,1) EvidenceUnitsBCollapsed(:,end) mean(EvidenceUnitsBCollapsed(:,1:end),2) ...
     max(EvidenceUnitsBCollapsed(:,1:end),[],2) min(EvidenceUnitsBCollapsed(:,1:end),[],2)];

% dm: first, last, mean, SD, max, min
dm_all = [EvidenceUnitsACollapsed(:,1) EvidenceUnitsACollapsed(:,end)  mean(EvidenceUnitsACollapsed(:,1:end),2)  std(EvidenceUnitsACollapsed(:,1:end),[],2)...
    max(EvidenceUnitsACollapsed(:,1:end),[],2) min(EvidenceUnitsACollapsed(:,1:end),[],2)...
    EvidenceUnitsBCollapsed(:,1) EvidenceUnitsBCollapsed(:,end) mean(EvidenceUnitsBCollapsed(:,1:end),2) std(EvidenceUnitsBCollapsed(:,1:end),[],2) ...
     max(EvidenceUnitsBCollapsed(:,1:end),[],2) min(EvidenceUnitsBCollapsed(:,1:end),[],2)];

% dm: first, last, mean, SD
dm_no_max_min = [EvidenceUnitsACollapsed(:,1) EvidenceUnitsACollapsed(:,end)  mean(EvidenceUnitsACollapsed(:,1:end),2)  std(EvidenceUnitsACollapsed(:,1:end),[],2)...
    EvidenceUnitsBCollapsed(:,1) EvidenceUnitsBCollapsed(:,end) mean(EvidenceUnitsBCollapsed(:,1:end),2) std(EvidenceUnitsBCollapsed(:,1:end),[],2)];

% dm: mean, SD, max, min
dm_no_first_last = [mean(EvidenceUnitsACollapsed(:,1:end),2)  std(EvidenceUnitsACollapsed(:,1:end),[],2)...
    max(EvidenceUnitsACollapsed(:,1:end),[],2) min(EvidenceUnitsACollapsed(:,1:end),[],2)...
    mean(EvidenceUnitsBCollapsed(:,1:end),2) std(EvidenceUnitsBCollapsed(:,1:end),[],2) ...
     max(EvidenceUnitsBCollapsed(:,1:end),[],2) min(EvidenceUnitsBCollapsed(:,1:end),[],2)];

% dm: mean, SD
dm_mean_sd = [mean(EvidenceUnitsACollapsed(:,1:end),2)  std(EvidenceUnitsACollapsed(:,1:end),[],2)...
     mean(EvidenceUnitsBCollapsed(:,1:end),2) std(EvidenceUnitsBCollapsed(:,1:end),[],2)];

% dm: mean, max, min
dm_mean_max_min = [mean(EvidenceUnitsACollapsed(:,1:end),2)  ...
    max(EvidenceUnitsACollapsed(:,1:end),[],2) min(EvidenceUnitsACollapsed(:,1:end),[],2)...
    mean(EvidenceUnitsBCollapsed(:,1:end),2) ...
     max(EvidenceUnitsBCollapsed(:,1:end),[],2) min(EvidenceUnitsBCollapsed(:,1:end),[],2)];

% dm: first, last, mean
dm_mean_first_last = [EvidenceUnitsACollapsed(:,1) EvidenceUnitsACollapsed(:,end)  mean(EvidenceUnitsACollapsed(:,1:end),2)  ...
    EvidenceUnitsBCollapsed(:,1) EvidenceUnitsBCollapsed(:,end) mean(EvidenceUnitsBCollapsed(:,1:end),2)];

% dm: mean only
dm_mean = [mean(EvidenceUnitsACollapsed(:,1:end),2)  ...
    mean(EvidenceUnitsBCollapsed(:,1:end),2)];

% dm: no mean
dm_no_mean = [EvidenceUnitsACollapsed(:,1) EvidenceUnitsACollapsed(:,end)  std(EvidenceUnitsACollapsed(:,1:end),[],2)...
    max(EvidenceUnitsACollapsed(:,1:end),[],2) min(EvidenceUnitsACollapsed(:,1:end),[],2)...
    EvidenceUnitsBCollapsed(:,1) EvidenceUnitsBCollapsed(:,end) std(EvidenceUnitsBCollapsed(:,1:end),[],2) ...
     max(EvidenceUnitsBCollapsed(:,1:end),[],2) min(EvidenceUnitsBCollapsed(:,1:end),[],2)];

%% k-fold cross-validation
ChosenTarget_kcv = (ChosenTargetCollapsed(:)==1);
k_kcv = 10; %This is the number of splits of the data. 

%Store all the models in a cell array. 
ModzInArray= {dm_no_sd; dm_all; dm_no_max_min;dm_no_first_last; dm_mean_sd; ...
    dm_mean_max_min; dm_mean_first_last; dm_mean; dm_no_mean;};
LL_forloop = zeros(size(ModzInArray,1),1); %Log-likelihood for each model


%% Run the analysis
for i_kcv_runs = 1:n_kcv_runs %Number of runs refers to how many times the analysis is repeated
    indices_kcv = crossvalind('Kfold', ChosenTarget_kcv, k_kcv); %Divide the trials into k-splits of the data. 
for i_kcv = 1:k_kcv 
    %Loops across the splits in the data, performing cross-validation for
    %each model
    ind_validation = (indices_kcv == i_kcv); %Trials to use for the validation set
    ind_training = ~ind_validation; %Trials to use for the training set
   
    for ModNum=1:size(ModzInArray,1); %Loop across the different models. 
        dm_here = ModzInArray{ModNum};
        fitglm_kcv_training_forloop = fitglm(dm_here(ind_training,:),ChosenTarget_kcv(ind_training), 'linear', 'distribution','binomial', 'link','logit'); % 'constant', 'on' <=> 'intercept', 'true' (default)
        betas_kcv_training_forloop  = fitglm_kcv_training_forloop .Coefficients.Estimate; %Beta coefficient estimates from the training data
        ChosenTarget_kcv_validation_estimated_forloop = ...
            betas_kcv_training_forloop(1) + dm_here(ind_validation,:) * betas_kcv_training_forloop(2:end); 
        pred_LL_forloop = exp(ChosenTarget_kcv_validation_estimated_forloop) ./ (1 + (exp(ChosenTarget_kcv_validation_estimated_forloop))); %Predicted data from the validation set
        LL_forloop(ModNum) = LL_forloop(ModNum) + sum(log(1-abs(pred_LL_forloop - ChosenTarget_kcv(ind_validation))))/n_kcv_runs; %Calculate the log-likelihood
    end
end
    fprintf(['Completed ' num2str(i_kcv_runs) ' cross-validation runs of ' num2str(n_kcv_runs) '\n'])
end

%% Supplementary Tables Output: Table 1
%Difference in log-likelihood of Full regression model (mean, SD, max, min, first, last of evidence values; equation 6 in Methods) vs reduced model
 ComparisonOfLikelihoods = [LL_forloop(2)-LL_forloop(9);...
   LL_forloop(2)-LL_forloop(4);...
   LL_forloop(2)-LL_forloop(1);...
   LL_forloop(2)-LL_forloop(3)];
Output.Table1 = cat(2,{'Mean';'First & Last';'SD';'Max and Min'},num2cell(ComparisonOfLikelihoods))';

%% Supplementary Tables Output: Table 2
%Difference in log-likelihood of regression models including either evidence standard deviation (SD) or both maximum and minimum evidence (Max & Min) as regressors, for each monkey and the circuit model
ComparisonOfLikelihoods = [LL_forloop(5)-LL_forloop(6);...
    LL_forloop(3)-LL_forloop(1)];
Output.Table2 = cat(2,{'Mean';'Mean; and First & Last'},num2cell(ComparisonOfLikelihoods))';

%% Supplementary Table Output: Table 3
% Increase in log-likelihood of various regression models (regressors in column labels) due to inclusion of evidence standard deviation as a regressor
ComparisonOfLikelihoods = [LL_forloop(5)-LL_forloop(8);...
    LL_forloop(3)-LL_forloop(7);...
    LL_forloop(4)-LL_forloop(6);...
    LL_forloop(2)-LL_forloop(1)];
 
Output.Table3 = cat(2,{'Mean';'Mean, First, & Last'; 'Mean, Max, & Min';'Mean, Max, Min, First, & Last'},num2cell(ComparisonOfLikelihoods))';

end

