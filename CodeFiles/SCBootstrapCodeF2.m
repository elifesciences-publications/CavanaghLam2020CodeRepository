function [BootParamSave] = SCBootstrapCodeF2(StrNameOfFunc,BootStrapNo,OriginalDM,YVar,...
    NameOfField,Starting_parameters)
%%
%This function is similar to scFittingFunctionForLapsesF2, but it will fit
%the model to a bootstrap distributions of trials.
%% Preassign variable before parfor
exitflagstore = nan(BootStrapNo,1);
BootParamSave = nan(BootStrapNo,length(NameOfField));

%% Parfor across iteration numbers
parfor i =1:BootStrapNo
    exitflagstore(i) =5; ErrorCount =0; ErrorThreshold = 2;
    %When set as above, these variables ensure that if the algorithm fails to converge, 
    %a new trial set can be chosen up to twice. 
    while exitflagstore(i)~=1 & ErrorCount<ErrorThreshold
        %I.e. run this iteration until a suitable trialset which coverges
        %is found; or there are 2 attempts. 
        
        Inds = datasample(1:length(YVar),length(YVar),'Replace',true); %Trials to run the analysis on
        [~,~,exitflagstore(i), ~,BootParamSave(i,:)] = fitSC(StrNameOfFunc,...
            Starting_parameters, NameOfField,...
            OriginalDM(Inds,:),+YVar(Inds));
        ErrorCount = ErrorCount+1;
    end
    
    
end

end

