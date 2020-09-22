function [pValsKetvSal] = ...
    PermTestsWithLapseFitting(Ysaline, ...
    YKetamine,n_perm,SalineDM,KetamineDM)

%% Initialization
n_trials_saline = length(Ysaline);                             % Number of Saline trials.
n_trials_ketamine = length(YKetamine);                              % Number of Ketamine trials.
range_n_trial_all = 1:(n_trials_saline+n_trials_ketamine);                 % Numbered indices of saline and ketmaine trials.

ChosenTarget_all = [Ysaline, YKetamine]; %Stack the chosen target across saline and ketamine datasets
DM_all = [SalineDM; KetamineDM];%Stack the regression design matricies.

%% Calculate the true fitted values from Equation 8 or 10 (see Methods)

[~,~,~,CollationOfAllTermsInLapseModelsSAL] = scFittingFunctionForLapsesF2(SalineDM,Ysaline',[]);
%Order of rows: Beta0 (Constant Term); Beta1 - BetaN; Y0 (Lapse Term)
[~,~,~,CollationOfAllTermsInLapseModelsKET] = scFittingFunctionForLapsesF2(KetamineDM,YKetamine',[]);


%% Calculate True difference between ketamine and saline
TrueKetSalineDiff = CollationOfAllTermsInLapseModelsKET-CollationOfAllTermsInLapseModelsSAL;

%% Shuffle data and run regression models.
PermVals_SalVKet = nan([size(TrueKetSalineDiff) n_perm]);
fprintf('Running permutation test for lapse-corrected regression using parfor (may take a while) \n');

parfor i_perm = 1:n_perm
    
    indices_saline = range_n_trial_all(sort(randperm(numel(range_n_trial_all), n_trials_saline)));         % Randomly choose a shuffled subset as saline trials.
    indices_ket = setdiff(range_n_trial_all, indices_saline);                                              % The rest are ketamine trials.
    Ysaline_shuffled = ChosenTarget_all(indices_saline);
    YKetamine_shuffled = ChosenTarget_all(indices_ket);
    
    SalineDM_Shuf = DM_all(indices_saline,:); %Run the analyses on the shuffled data assigned to the saline category
    [~,~,~,CollationOfAllTermsInLapseModelsSAL] = scFittingFunctionForLapsesF2(SalineDM_Shuf,Ysaline_shuffled',[]);
    
    KetamineDM_Shuf = DM_all(indices_ket,:); %Run the analyses on the shuffled data assigned to the ketamine category
    [~,~,~,CollationOfAllTermsInLapseModelsKET] = scFittingFunctionForLapsesF2(KetamineDM_Shuf,YKetamine_shuffled',[]);
    
    %Calculate the difference between the parameters in the two groups
    PermVals_SalVKet(:,:,i_perm) = CollationOfAllTermsInLapseModelsKET-CollationOfAllTermsInLapseModelsSAL;
end

fprintf('Completed permutation test for lapse-corrected regression using parfor \n');

%% Calculate the p-values by comparing true and shuffled data
pValsKetvSal =   (sum(abs(PermVals_SalVKet)>abs(TrueKetSalineDiff),3))/n_perm;
end


