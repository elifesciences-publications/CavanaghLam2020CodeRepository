function [CollationOfBetas,bestPModel,BootParamSave,CollationOfAllTermsInLapseModels]=...
    scFittingFunctionForLapsesF2(OriginalDM,YVar,BootStrapNo) 
%This is a function which can add lapse terms to an existing logistic
%regression model. The inputs are: 1) OriginalDM: a Tr x Predictors matrix,
%which includes any constant terms. 2) YVar: The binary y variable. 
%The function then uses optimisation algorithms to get the coefficients. 
%4) Number of bootstrap iterations
 
%Outputs:
%CollationOfBetas - output contains 2 columns: 1) Betas from Logit Reg 2) Betas from a
%regression with a lapse term.
%bestPModel - all of the parameters fitted by the lapse-term model to the true data
%BootParamSave - The full bootstrap distributions for both the lapsing model
%% Define the outputs so none are left un-allocated.
CollationOfBetas = nan;
bestPModel = {};
BootParamSave = {}; 
%% Run a logistic regression, without a lapse term, with the inputted design matrix
[bLOG,~,~] = glmfit(OriginalDM,YVar,'binomial','link','logit','constant','off');

%% Assign the parameters from the logistic as starting parameters for the optimisation algorithm
Starting_parameters = [];
for t = 1:size(OriginalDM,2)
    NameOfField{t} = ['Beta' num2str(t-1)];
    Starting_parameters.(NameOfField{t}) = bLOG(t);
    %Starting parameters for the optimisation algorithm fitting of the lapse equation are equal to the logistic regression results.
end

%% Fit the model with the lapse term to the true data
NameOfField = cat(2,NameOfField,{'YZERO'});
% % . y0 + (1-y0)/(1+exp(-(beta0 + beta1*x1 + beta2*x2 etc.)))

Starting_parameters.YZERO = 0.1; %Starting estimate for the lapse term

[bestPModel{1},~,~, ~] = fitSC('LapseTermFitRegularisation',...
    Starting_parameters, NameOfField,...
    OriginalDM,+YVar);
%Function which interfaces with Matlab's fminsearch algorithm. 
  
%% Bootstrap for lapse parameter model
if ~isempty(BootStrapNo)
fprintf(['Beginning a bootstrap procedure with ' num2str(BootStrapNo) ' iterations\n']);
Starting_parameters.YZERO = 0.1; 
    [BootParamSave{1}{1}] = SCBootstrapCodeF2('LapseTermFitRegularisation',BootStrapNo,OriginalDM,YVar,...
        NameOfField,Starting_parameters);
fprintf(['Finished a bootstrap procedure with ' num2str(BootStrapNo) ' iterations\n']);
end
%% Save relevant output from the lapse-fitted model
    for Xs=1:size(OriginalDM,2)
        ExtractBetasFromModels(Xs,1) = bestPModel{1}.(['Beta' num2str(Xs-1)]);
    end
%%
CollationOfBetas = cat(2,bLOG,ExtractBetasFromModels); 
%First column are the parameters from the logistic regression, second
%column from the lapse-adjusted fitting. 

CollationOfAllTermsInLapseModels = cat(1,ExtractBetasFromModels,...
[bestPModel{1}.YZERO]); %Rows are the betas, then the last row is the lapse term
end

