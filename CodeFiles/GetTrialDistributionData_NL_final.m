function [n_distribution_regularTr,n_distribution_narrow_high,n_distribution_broad_high,n_distribution_NB_balanced, n_SD_distribution_regularTr]...
    = GetTrialDistributionData_NL_final(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,TrialType)
%% Info
%This function produces grids to reflect the evidence distributions acrosstrial types
%%
dx_distribution = 0.0005; %Width of the bins used. 
x_distribution_list = 0.0:dx_distribution:1; %List of bins used

n_distribution_regularTr = zeros(length(x_distribution_list),length(x_distribution_list));      % axis is (narrow, broad)
n_distribution_narrow_high = zeros(length(x_distribution_list),length(x_distribution_list));      % axis is (narrow, broad)
n_distribution_broad_high = zeros(length(x_distribution_list),length(x_distribution_list));      % axis is (narrow, broad)
n_distribution_NB_balanced = zeros(length(x_distribution_list),length(x_distribution_list));      % axis is (narrow, broad)

dx_SD_distribution = 0.01; %Width of the bins used.
x_SD_distribution_list = 0.:dx_SD_distribution:0.3; %List of bins used
n_SD_distribution_regularTr = zeros(length(x_SD_distribution_list),length(x_SD_distribution_list));      % axis is (narrow, broad)

for i_trial_type = 1:length(TrialType)
    if TrialType(i_trial_type) ==1                                         % Regular trial
        mean_x_A_temp = nanmean(EvidenceUnitsACollapsed(i_trial_type,:)); %Mean evidence for left option
        mean_x_B_temp = nanmean(EvidenceUnitsBCollapsed(i_trial_type,:)); %Mean evidence for right option
        std_x_A_temp = nanstd(EvidenceUnitsACollapsed(i_trial_type,:)); %STD of evidence for left option
        std_x_B_temp = nanstd(EvidenceUnitsBCollapsed(i_trial_type,:)); %STD of evidence for right option
        if std_x_A_temp >= std_x_B_temp                                     % if A is broader than B  
            n_distribution_regularTr(ceil(mean_x_B_temp/dx_distribution), ceil(mean_x_A_temp/dx_distribution)) = n_distribution_regularTr(ceil(mean_x_B_temp/dx_distribution), ceil(mean_x_A_temp/dx_distribution)) + 1;
            n_SD_distribution_regularTr(ceil(std_x_B_temp/dx_SD_distribution), ceil(std_x_A_temp/dx_SD_distribution)) = n_SD_distribution_regularTr(ceil(std_x_B_temp/dx_SD_distribution), ceil(std_x_A_temp/dx_SD_distribution)) + 1;
        elseif std_x_A_temp < std_x_B_temp                                     % if B is broader than A  
            n_distribution_regularTr(ceil(mean_x_A_temp/dx_distribution), ceil(mean_x_B_temp/dx_distribution)) = n_distribution_regularTr(ceil(mean_x_A_temp/dx_distribution), ceil(mean_x_B_temp/dx_distribution)) + 1;
            n_SD_distribution_regularTr(ceil(std_x_A_temp/dx_SD_distribution), ceil(std_x_B_temp/dx_SD_distribution)) = n_SD_distribution_regularTr(ceil(std_x_A_temp/dx_SD_distribution), ceil(std_x_B_temp/dx_SD_distribution)) + 1;
        end
    elseif TrialType(i_trial_type) ==18                                    % (Narrow High (B))
        mean_x_A_temp = nanmean(EvidenceUnitsACollapsed(i_trial_type,:));
        mean_x_B_temp = nanmean(EvidenceUnitsBCollapsed(i_trial_type,:));
        n_distribution_narrow_high(ceil(mean_x_B_temp/dx_distribution), ceil(mean_x_A_temp/dx_distribution)) = n_distribution_narrow_high(ceil(mean_x_B_temp/dx_distribution), ceil(mean_x_A_temp/dx_distribution)) + 1;
    elseif TrialType(i_trial_type) ==21                                    % (Narrow High (A))
        mean_x_A_temp = nanmean(EvidenceUnitsACollapsed(i_trial_type,:));
        mean_x_B_temp = nanmean(EvidenceUnitsBCollapsed(i_trial_type,:));
        n_distribution_narrow_high(ceil(mean_x_A_temp/dx_distribution), ceil(mean_x_B_temp/dx_distribution)) = n_distribution_narrow_high(ceil(mean_x_A_temp/dx_distribution), ceil(mean_x_B_temp/dx_distribution)) + 1;
    elseif TrialType(i_trial_type) ==19                                    % (Broad High (B))
        mean_x_A_temp = nanmean(EvidenceUnitsACollapsed(i_trial_type,:));
        mean_x_B_temp = nanmean(EvidenceUnitsBCollapsed(i_trial_type,:));
        n_distribution_broad_high(ceil(mean_x_B_temp/dx_distribution), ceil(mean_x_A_temp/dx_distribution)) = n_distribution_broad_high(ceil(mean_x_B_temp/dx_distribution), ceil(mean_x_A_temp/dx_distribution)) + 1;
    elseif TrialType(i_trial_type) ==22                                    % (Broad High (A))
        mean_x_A_temp = nanmean(EvidenceUnitsACollapsed(i_trial_type,:));
        mean_x_B_temp = nanmean(EvidenceUnitsBCollapsed(i_trial_type,:));
        n_distribution_broad_high(ceil(mean_x_A_temp/dx_distribution), ceil(mean_x_B_temp/dx_distribution)) = n_distribution_broad_high(ceil(mean_x_A_temp/dx_distribution), ceil(mean_x_B_temp/dx_distribution)) + 1;
    elseif TrialType(i_trial_type) ==20                                    % (Balanced/ Equal Means (B))
        mean_x_A_temp = nanmean(EvidenceUnitsACollapsed(i_trial_type,:));
        mean_x_B_temp = nanmean(EvidenceUnitsBCollapsed(i_trial_type,:));
        n_distribution_NB_balanced(ceil(mean_x_B_temp/dx_distribution), ceil(mean_x_A_temp/dx_distribution)) = n_distribution_NB_balanced(ceil(mean_x_B_temp/dx_distribution), ceil(mean_x_A_temp/dx_distribution)) + 1;
    elseif TrialType(i_trial_type) ==23                                    % (Balanced/ Equal Means (A))
        mean_x_A_temp = nanmean(EvidenceUnitsACollapsed(i_trial_type,:));
        mean_x_B_temp = nanmean(EvidenceUnitsBCollapsed(i_trial_type,:));
        n_distribution_NB_balanced(ceil(mean_x_A_temp/dx_distribution), ceil(mean_x_B_temp/dx_distribution)) = n_distribution_NB_balanced(ceil(mean_x_A_temp/dx_distribution), ceil(mean_x_B_temp/dx_distribution)) + 1;
    end
end

end

