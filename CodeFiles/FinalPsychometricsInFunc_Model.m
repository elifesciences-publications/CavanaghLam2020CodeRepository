function [dx_list,P_corr_Subj_list,ErrBar_P_corr_Subj_list,Psychometric_fit_params] = FinalPsychometricsInFunc_Final(EvidenceUnitsACollapsed,EvidenceUnitsBCollapsed,ChosenTargetCollapsed,Method)
%% Define some important variables
dx_pos_log_list = logspace(log(0.02)/log(10), log(0.4)/log(10), 10);  %Initial bin spacing for evidence values                                                       % x for evidence
BestOptionIndicator = nan(size(EvidenceUnitsACollapsed,1),1);
BestOptionIndicator(nanmean(EvidenceUnitsACollapsed,2)<nanmean(EvidenceUnitsBCollapsed,2)) = 2;
BestOptionIndicator(nanmean(EvidenceUnitsACollapsed,2)>nanmean(EvidenceUnitsBCollapsed,2)) = 1;
BestOptionIndicator(nanmean(EvidenceUnitsACollapsed,2)==nanmean(EvidenceUnitsBCollapsed,2)) = 0;
 
%% Set up the bins for the psychometric function

if strcmp(Method,'CorrectIncorrect')
    dx_list = sort([0, 0.01, dx_pos_log_list]); %Modify the bin spacing                                                         % x for evidence (2nd element in P_corr_Subj_list is for small (< ~0.017) but not exactly 0 data. 0.01 is a placeholder)
elseif strcmp(Method,'NarrowBroad');
    %Modify the bin spacing, including separating exactly 0 vs small but non-zero net evidence.
    dx_list = sort([-dx_pos_log_list, -0.01, 0, 0.01, dx_pos_log_list]);                                                         % x for evidence (elements right before and after x=0 in P_corr_Subj_list is for small (< ~0.017) but not exactly 0 data. 0.01 is a placeholder)
end

P_corr_Subj_list = zeros(length(dx_list), 1);
n_dx_list = zeros(length(dx_list), 1);

%% Indexed groupings
if strcmp(Method,'CorrectIncorrect') 
    %Bin the trials according to their difficulty (i.e. difference in mean evidence)
    i_dx_pos_log_Collapsed = length(dx_pos_log_list)-round((log(abs(nanmean(EvidenceUnitsACollapsed,2) -nanmean(EvidenceUnitsBCollapsed,2))) - log(dx_pos_log_list(end))) / (log(dx_pos_log_list(1)) - log(dx_pos_log_list(end)))*(length(dx_pos_log_list)-1));   % Log-Spaced. Ignoring signs and only map to positive log-space (from 1 to length(dx_pos_log_list)).
    i_dx_Collapsed = i_dx_pos_log_Collapsed+2;
    i_dx_Collapsed(i_dx_pos_log_Collapsed<1) =2;                                                                                       % <1 in i_dx_pos_log_Collapsed => absolute diff in evidence is less than the minimum for dx_pos_log_list(1)=0.02 (~0.017).
    i_dx_Collapsed(nanmean(EvidenceUnitsACollapsed,2) == nanmean(EvidenceUnitsBCollapsed,2)) =1;                                       % Exactly 0 net evidence.
    
elseif strcmp(Method,'NarrowBroad');

    is_A_broad = 2*heaviside((nanstd(EvidenceUnitsACollapsed,[],2) -nanstd(EvidenceUnitsBCollapsed,[],2)))-1;        % 1 if A is broad (or equally broad), -1 if not.
    is_A_broad(is_A_broad==0)=1;    
    
    % If we separate exactly 0 vs small but non-zero net evidence.
    i_dx_pos_log_Collapsed_NB = length(dx_pos_log_list)-round((log(abs(nanmean(EvidenceUnitsACollapsed,2) -nanmean(EvidenceUnitsBCollapsed,2))) - log(dx_pos_log_list(end))) / (log(dx_pos_log_list(1)) - log(dx_pos_log_list(end)))*(length(dx_pos_log_list)-1));   % Log-Spaced. Ignoring signs and only map to positive log-space (from 1 to length(dx_pos_log_list)).
    i_dx_Collapsed = (i_dx_pos_log_Collapsed_NB+1);                              % Now maps [2,length(dx_pos_log_list)+1] to dx_list.
    i_dx_Collapsed(i_dx_pos_log_Collapsed_NB<1) =1;                              % <1 in i_dx_pos_log_Collapsed => absolute diff in evidence is less than the minimum for dx_pos_log_list(1)=0.02 (~0.017).
    i_dx_Collapsed(nanmean(EvidenceUnitsACollapsed,2) == nanmean(EvidenceUnitsBCollapsed,2)) =0;                                       % Exactly 0 net evidence.
    i_dx_Collapsed = sign((nanmean(EvidenceUnitsACollapsed,2) -nanmean(EvidenceUnitsBCollapsed,2))) .* (i_dx_Collapsed);          % Now also sign-dependent (from -length(dx_pos_log_list)-1 to length(dx_pos_log_list)+1).
    i_dx_Collapsed = i_dx_Collapsed.*is_A_broad + length(dx_pos_log_list)+2;                                                             % If narrow/broad
    
    
end
%% Y variable
if strcmp(Method,'CorrectIncorrect') %Y variable is whether the correct option was chosen
    Correct_Option_Chosen_Collapsed = (BestOptionIndicator==(ChosenTargetCollapsed'));                                               % Works poorly. Seems to be due to sometimes best-option =0 instead of 1,2, while chosen-target always =1,2
%     Correct_Option_Chosen_Collapsed(nanmean(EvidenceUnitsACollapsed,2)==nanmean(EvidenceUnitsBCollapsed,2)) = TrialErrorComp(nanmean(EvidenceUnitsACollapsed,2)==nanmean(EvidenceUnitsBCollapsed,2))==0;
    YVariable = Correct_Option_Chosen_Collapsed; 

elseif strcmp(Method,'NarrowBroad');
    Broad_Option_Chosen_Collapsed = 0.5*(is_A_broad+1).*(2-transpose(ChosenTargetCollapsed)) ...
        + -0.5*(is_A_broad-1).*(1-(2-transpose(ChosenTargetCollapsed)));
    YVariable = Broad_Option_Chosen_Collapsed; 
end


%% For Loop across bins to work out the accuracy in that bin
 
    for i_dx_temp = 1:length(dx_list)
        P_corr_Subj_list(i_dx_temp) = sum((i_dx_Collapsed==i_dx_temp) .* YVariable) / sum(i_dx_Collapsed==i_dx_temp);
        n_dx_list(i_dx_temp) = sum(i_dx_Collapsed==i_dx_temp);
    end
    
%% Calculate errorbars
ErrBar_P_corr_Subj_list = sqrt(P_corr_Subj_list.*(1-P_corr_Subj_list) ./ n_dx_list);
    
    %% Fitting the psychometric function to the data
    
    if strcmp(Method,'CorrectIncorrect')
evidence_list = abs((nanmean(EvidenceUnitsACollapsed,2) -nanmean(EvidenceUnitsBCollapsed,2)));
StartPoint = [0.05, 1];             
% Psychometric function without shift parameter. Used to fit correct psychometric function (figure 2).
Psychometric_fit_function_list = @(Psychometric_params)...
    0.5 + 0.5.*(1. - exp(-(evidence_list./Psychometric_params(1)).^Psychometric_params(2)));

    elseif strcmp(Method,'NarrowBroad');
evidence_list = (nanmean(EvidenceUnitsACollapsed,2) -nanmean(EvidenceUnitsBCollapsed,2)) .* is_A_broad;
StartPoint = [0.05, 1, 0];
% Psychometric function with shift parameter, to fit for both positive and negative x.
Psychometric_fit_function_list = @(Psychometric_params)...
    0.5 + 0.5*sign(evidence_list+Psychometric_params(3)).*(1. - exp(-(abs(evidence_list+Psychometric_params(3))./Psychometric_params(1)).^Psychometric_params(2)));
        
    end
    
    
    
    %% MLE using Matlab's fmninsearch

% (Negative) log-likelihood function to minimize
Psychometric_fit_LL = @(Psychometric_params)...
    -sum(log( Psychometric_fit_function_list(Psychometric_params).*YVariable + (1-Psychometric_fit_function_list(Psychometric_params)).*(1-YVariable)));

Psychometric_fit_params = fminsearch(Psychometric_fit_LL, StartPoint);

end

