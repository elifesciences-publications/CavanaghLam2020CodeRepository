function [LogisticOutput,CollationOfBetas,BootParamSave,bestPModel] = RunPkAnalysis(Dm_input,BootStrapNo)
%% Inputs:

%Dm_input: design matrix for equation 4 (see Methods). Trials x (Number of Samples) + 1 matrix.
%The Number of Samples (NoS) will be 8 on Standard Sessions, and 6 on Pharmacological sessions.
%The first NoS columns, will the Left-Right evidence values at each timestep.
%The final column will be a logical of whether the Monkey chose Left

%BootstrapNo - this is the number of iterations used when using a bootstrap method to 
%generate parameter confidence intervals (only used when a lapse parameter is also 
%incorporated - i.e. equation 8, see Methods)  
%% Run the logistic regression model, and store the output
[b,~,stats] = glmfit(Dm_input(:,1:end-1),Dm_input(:,end),'binomial','link','logit','constant','on'); 
%Equation 4 in the paper (see Methods). Logistic regression using
%difference in evidence at each timestep to predict choice. 
BetasForEachTimeStep= b(2:end); ErrorsForEachTimeStep = stats.se(2:end);
LogisticOutput = cat(2,BetasForEachTimeStep,ErrorsForEachTimeStep);
%% Pharmacological sessions, where bootstrapping analyses produce extra outputs
if ~isempty(BootStrapNo)
    OriginalDM = [ones(size(Dm_input,1),1) Dm_input(:,1:end-1)];
    YVar = Dm_input(:,end);
    [CollationOfBetas,bestPModel,BootParamSave]= ...
        scFittingFunctionForLapsesF2(OriginalDM,YVar,BootStrapNo);
else
    % Where these analyses are not run, allocate their output as NaN
    CollationOfBetas = nan;
    bestPModel = nan;
    BootParamSave = nan; 
end
end

