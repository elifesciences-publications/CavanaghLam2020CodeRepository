function [ExclusionCriteria] = WorkOutSessionsToExcludeS(SessionWiseDMs,DrugDayStructure)
%% How many on drug trials were completed in each session?
for ses=1:length(SessionWiseDMs)
    CompDrugTrDefinition(ses) = sum(SessionWiseDMs(ses).resp_trials  & SessionWiseDMs(ses).NewVarForTrRef==2);
end
%%
AllSessions = cat(1,DrugDayStructure.BhvFileName); %Names of all sessions
SubjNameForEachSession = AllSessions(:,14); %Work out the subject performing each session

%Find the minimum number of 'OnDrug' saline trials the subject completed.
%They must reach this threshold on all 'OnDrug' ketamine sessions for the
%session to be included (see Methods). 
TrialNumNeededAlfie = min(CompDrugTrDefinition(~cat(1,DrugDayStructure.DrugDay) & SubjNameForEachSession=='A'));
TrialNumNeededHarry = min(CompDrugTrDefinition(~cat(1,DrugDayStructure.DrugDay) & SubjNameForEachSession=='H'));
 
ExclusionCriteria = zeros(size(DrugDayStructure,2),1);

TrialNumThreshComb = zeros(length(CompDrugTrDefinition),1);
TrialNumThreshComb(SubjNameForEachSession=='H') = TrialNumNeededHarry; %The number of trials needed for that session to be eligible
TrialNumThreshComb(SubjNameForEachSession=='A') = TrialNumNeededAlfie; %The number of trials needed for that session to be eligible

ExclusionCriteria(CompDrugTrDefinition<TrialNumThreshComb',1) = 1; %Low Tr number



end

