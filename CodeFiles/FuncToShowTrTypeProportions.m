function [Output] = FuncToShowTrTypeProportions(SessionStruct,SessionsToUse)
% The purpose of this function is to calculate the proportions of different
% trial types

%Reference of the meaning of different trial type numbers:
%BroadLow_Versus_NarrowHigh = TrialType ==18;
%BroadHigh_Versus_NarrowLow = TrialType ==19;
%BroadBalanced_Versus_NarrowBalanced = TrialType==20;
%NarrowHigh_Versus_BroadLow = TrialType ==21;
%NarrowLow_Versus_BroadHigh = TrialType ==22;
%NarrowBalanced_Versus_BroadBalanced = TrialType==23;

PosTrTypes = [1 16 17 18:23 45];
PosTrTypeNames = {'Regular Tr';...
    'HalfHalf:LeftHighLow RightLowHigh';'HalfHalf:LeftLowHigh RightHighLow';...
    'BroadLow v NarrowHigh';'BroadHigh v NarrowLow'; 'BroadBalanced v NarrowBalanced';
    'NarrowHigh v BroadLow';'NarrowLow v BroadHigh'; 'NarrowBalanced v BroadBalanced';...
    'Control:NonIntegrateTr'};
TrTypeCollapsed = cat(2,SessionStruct(SessionsToUse).CompletedTrialType); 
%Collapse trial type data across sessions

%% Loop across different trial types to calculate total and proportion
for tt=1:length(PosTrTypes)
    SumHere(tt) = sum(TrTypeCollapsed==PosTrTypes(tt));
    ProportionsHere(tt) = SumHere(tt)/length(TrTypeCollapsed);
end

Output = [ProportionsHere(1) ProportionsHere(2)+ProportionsHere(3) sum(ProportionsHere(4:9)) ProportionsHere(10)];
%1st entry is the proportion of Regular trials; 
%2nd entry is the proportion of 'HalfHalf' trials;
%3rd entry is the proportion of 'NarrowBroad' trials;
%4th entry is the proportion of 'Control' trials
end

