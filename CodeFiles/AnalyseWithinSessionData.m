function [SessionWiseDMs,BinnedAnals,OutputDMHere] = AnalyseWithinSessionData(WithinSessionStruct)
%% Unpack the input structure into variables
resp_trials= WithinSessionStruct.resp_trials; %Did the subject complete the trial by making a behavioural choice?
TrialError= WithinSessionStruct.TrialError; %MonkeyLogic record of trial outcome (0 = correct; 6 = incorrect; others = imply incomplete)
DrugGivenTime= WithinSessionStruct.DrugGivenTime; % Time in HH; MM that the subject was injected
TrialStarTimeHours= WithinSessionStruct.TrialStarTimeHours; %Time each trial began - hours
TrialStarTimeMins= WithinSessionStruct.TrialStarTimeMins; %Time each trial began - minutes
TrialStarTimeSecs= WithinSessionStruct.TrialStarTimeSecs; %Time each trial began - seconds
TrialType= WithinSessionStruct.CompletedTrialType; %Reference number of the trial type for further anaylses
ChosenTarget= WithinSessionStruct.ChosenTarget; %Chosen target (1=left; 2=right) on Completed trials
EvidenceUnitsA= WithinSessionStruct.EvidenceUnitsA; %Evidence values for the Left Option on Completed trials
EvidenceUnitsB= WithinSessionStruct.EvidenceUnitsB; %Evidence values for the Right Option on Completed trials

%% Work out the times of injection, Drug ON (5 mins post injection), and Drug OFF (30 mins post injection).
X = datestr(((DrugGivenTime(1)*60)+DrugGivenTime(2))/((24*60))+0.0035, 'HH:MM:SS');
t = datetime(num2str(X),'InputFormat','HH:mm:ss');
[hON_drug5min,mON_drug5min] = hms(t); %Get the time of 'DrugOn' in Hours and minutes

Xnew = datestr(((DrugGivenTime(1)*60)+DrugGivenTime(2))/((24*60)), 'HH:MM:SS');
tnew = datetime(num2str(Xnew),'InputFormat','HH:mm:ss');
[hON_Inj,mON_Inj] = hms(tnew); %Get the time of 'Injection' in Hours and minutes

X = datestr(((DrugGivenTime(1)*60)+DrugGivenTime(2))/((24*60))+(1/48), 'HH:MM:SS'); %Adds 30 mins on to the time from when the drug was given
t = datetime(num2str(X),'InputFormat','HH:mm:ss');
[hOFF,mOFF] = hms(t); %Get the time of 'DrugOff' in Hours and minutes

clear X t Xnew tnew; %clear used variables

%% Define what is a drug trial and what is a pre drug trial
HMS_TrialStart = cat(2,TrialStarTimeHours,TrialStarTimeMins,TrialStarTimeSecs);
DrugTrials = zeros(size(TrialStarTimeMins,1),1);
PreDrugTrials = zeros(size(TrialStarTimeMins,1),1);
PostDrugTrials = zeros(size(TrialStarTimeMins,1),1);

%% Define On Drug Trials
IndexForFirstOnDrugTr = find(HMS_TrialStart(:,1)==hON_drug5min & HMS_TrialStart(:,2)==mON_drug5min,1); %Find the trial index of the first drug trial
if isempty(IndexForFirstOnDrugTr)
    %If no trials are started in that minute, look for the first trial started in the next minute
    IndexForFirstOnDrugTr =  find(HMS_TrialStart(:,1)==hON_drug5min & HMS_TrialStart(:,2)==mON_drug5min+1,1);
end

if mOFF==0
    IndexForLastOnDrugTr = find(HMS_TrialStart(:,1)==hOFF-1 & HMS_TrialStart(:,2)==59,1,'last');
else
    %Use the last trial in the minute before drug OFF.
    IndexForLastOnDrugTr = find(HMS_TrialStart(:,1)==hOFF & HMS_TrialStart(:,2)==mOFF-1,1,'last');
end

DrugTrials(IndexForFirstOnDrugTr:IndexForLastOnDrugTr) = 1; %Index from the first trial started in the minute 5 mins after injection to  the last trial starting within the minute 30 mins after injection

%% Define Pre-drug trials

if mON_Inj~=0
    PreDrugTrials(1:find(HMS_TrialStart(:,1)==hON_Inj & HMS_TrialStart(:,2)==mON_Inj-1,1,'last')) = 1;
else
    %Drug was given exactly on the hour. So run up to the last trial on 59 mins past the previous hour.
    PreDrugTrials(1:find(HMS_TrialStart(:,1)==hON_Inj-1 & HMS_TrialStart(:,2)==59,1,'last')) = 1;
end

%% Define post-drug trials
PostDrugTrials(find(DrugTrials,1,'last')+1:length(DrugTrials)) = 1;

%% Select the trials to include in the analyses (Responded AND trials of a certain trial type).
% Responsed trials only.
DrugTrialsResp = DrugTrials(resp_trials); %On drug trials, where the subject responded
PreDrugTrialsResp = PreDrugTrials(resp_trials); %Pre drug trials, where the subject responded
PostDrugTrialsResp = PostDrugTrials(resp_trials);
TrialErrorResp = TrialError(resp_trials); %Trial error, just on responded trials

TrialsToIncludeInAnalysesForDrugDays = TrialType==1 | TrialType==16 | TrialType==17 | TrialType==20 | TrialType==23;
% Analyse regular trials (Type 1); Half-Half trials (Types 16-17); NarrowBroad (Types 20, 23)

DrugTrialsResp_IncTrials = DrugTrialsResp(TrialsToIncludeInAnalysesForDrugDays);
PreDrugTrialsResp_IncTrials = PreDrugTrialsResp(TrialsToIncludeInAnalysesForDrugDays);
PostDrugTrialsResp_Reg_Trials = PostDrugTrialsResp(TrialsToIncludeInAnalysesForDrugDays);

%Redefine these variables to just include elements from included trials
ChosenTarget_IncTrials = ChosenTarget(TrialsToIncludeInAnalysesForDrugDays);
EvidenceUnitsA_IncTrials = EvidenceUnitsA((TrialsToIncludeInAnalysesForDrugDays),:);
EvidenceUnitsB_IncTrials = EvidenceUnitsB((TrialsToIncludeInAnalysesForDrugDays),:);
TrialErrorResp_IncTrials = TrialErrorResp((TrialsToIncludeInAnalysesForDrugDays));

DrugTrialsResp_TrialError = TrialErrorResp(DrugTrialsResp' & TrialsToIncludeInAnalysesForDrugDays);

%% 9 regerssors model to analyse pro-variance bias (3 constant terms, 3 regressors with evidence mean, 3 regressors with STD information)
%Store information about trials in a regression design matrix.

DM_Here = cat(2,PreDrugTrialsResp_IncTrials,DrugTrialsResp_IncTrials,PostDrugTrialsResp_Reg_Trials,nanmean(EvidenceUnitsA_IncTrials,2)-nanmean(EvidenceUnitsB_IncTrials,2)...
    ,nanstd(EvidenceUnitsA_IncTrials,[],2)-nanstd(EvidenceUnitsB_IncTrials,[],2));
%Up to this point, columns 1 to 3 are constant terms; %4 is the Mean Evidence difference between Left
%and Right options; %5 is the STD difference between Left and Right Options

DM_Here(:,6) = DM_Here(:,1).*DM_Here(:,4); %6 is Mean Evidence on Pre-drug trials
DM_Here(:,9) = DM_Here(:,1).*DM_Here(:,5); %9 is STD Evidence on Pre-drug trials
DM_Here(:,7) = DM_Here(:,2).*DM_Here(:,4); %7 is Mean Evidence  on Drug trials
DM_Here(:,10) = DM_Here(:,2).*DM_Here(:,5);%10 is STD Evidence on Drug trials
DM_Here(:,8) = DM_Here(:,3).*DM_Here(:,4); %8 is Mean Evidence  post drug trials
DM_Here(:,11) = DM_Here(:,3).*DM_Here(:,5); %11 is STD Evidence post drug trials

DM_Here(:,4:5) = []; %Remove the columns for Mean Evidence and STD evidence which were not split by trial period.

SessionWiseDMs.ProVar9RegModel = cat(2,DM_Here,(ChosenTarget_IncTrials==1)'); %Also store the choice data

%% Store a regression model to look at the temporal weights
SessionWiseDMs.OnDrugTemporalWeightsDM = cat(2,cat(2,EvidenceUnitsA_IncTrials(find(DrugTrialsResp_IncTrials),:),EvidenceUnitsB_IncTrials(find(DrugTrialsResp_IncTrials),:))...
    ,(ChosenTarget_IncTrials(find(DrugTrialsResp_IncTrials))==1)');
%Columns 1-6 are the evidence on the left hand side, during drug trials
%Columns 7-12 are the evidence on the right hand side, during drug trials
%Column 13 is the choice responses.

%% Store information about the trial errors on completed trials
SessionWiseDMs.CompTr_TrialError = DrugTrialsResp_TrialError; %Store the trial error information on completed trials

%% Store trial information in sliding bins relative to the time of injection

ProVar_FullDM =    [nanmean(EvidenceUnitsA_IncTrials,2)-nanmean(EvidenceUnitsB_IncTrials,2)...
    ,nanstd(EvidenceUnitsA_IncTrials,[],2)-nanstd(EvidenceUnitsB_IncTrials,[],2)]; %Pro-variance regression model, for all completed trials
ChosenTargetMaker =(ChosenTarget_IncTrials==1)'; %Chosen target, for all completed trials
 
BinWidthMins = 6;
StepSize = 1; %In minutes 
StartPoint = -20; %Minutes relative to injection to start the analysis
EndPoint = 60; %Minutes relative to injection to end the analysis
ConvertDrugGivenTime = 60*((hON_Inj*60)+mON_Inj); %Seconds from midnight drug was given
AllBins = StartPoint:StepSize:EndPoint;
ConvertTrStartTIme = 60*(HMS_TrialStart(:,1)*60+HMS_TrialStart(:,2))+HMS_TrialStart(:,3);
BinCounter = 1; %For use in for loop below

for Bn=AllBins %Loop across bins
    BinStartTime = ConvertDrugGivenTime+(60*Bn)-(60*BinWidthMins/2); %In seconds from midnight
    BinEndTime = ConvertDrugGivenTime+(60*Bn)+(60*BinWidthMins/2);
    
    TrInThisBin = (ConvertTrStartTIme>BinStartTime & ConvertTrStartTIme<BinEndTime); %Find trials started in this bin
    TrInThisBinRelevant = TrInThisBin(resp_trials); %Find completed trials started in this bin
    TrInThisBinRelevant = TrInThisBinRelevant(TrialsToIncludeInAnalysesForDrugDays); %Find completed included trials
    
    % Performance accuracy in this bin
    BinnedAnals(1,BinCounter) = sum(TrialErrorResp_IncTrials(TrInThisBinRelevant)==0)/(sum(TrialErrorResp_IncTrials(TrInThisBinRelevant)==0)...
        +sum(TrialErrorResp_IncTrials(TrInThisBinRelevant)==6));
    
    % Output the pro-variance bias regression design matrix here
    OutputDMHere{1,BinCounter} = [ProVar_FullDM(TrInThisBinRelevant,:) ChosenTargetMaker(TrInThisBinRelevant)];
    
    BinCounter = BinCounter+1;
end
%% Reference all trials - according to whether they are pre, on, or post drug
NewVarForTrRef = nan(length(DrugTrials),1);
NewVarForTrRef(find(PreDrugTrials)) = 1;
NewVarForTrRef(find(DrugTrials)) = 2;
NewVarForTrRef(find(PostDrugTrials)) = 3;
SessionWiseDMs.NewVarForTrRef = NewVarForTrRef;

%% Save resp_trials as an output
SessionWiseDMs.resp_trials = resp_trials;
end

