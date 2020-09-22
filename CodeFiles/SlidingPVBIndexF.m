function [PythonOutputs] ...
    = SlidingPVBIndexF(OutputDMHere,SalineSes,KetamineSes,n_perm,DirToSaveBootstraps,TryToLoadOldPermTests)
%OutputDMHere (an input to this function) is a sessions x time points cell array. Each entry is the
%design matrix for the pro-variance regression. This function calculates
%the PVB index at each time point - then runs some significance tests.

%% Try loading existing results of permutation test
if TryToLoadOldPermTests
    cd(DirToSaveBootstraps);
    files=dir(fullfile(DirToSaveBootstraps,'SlidingPVBdata*.mat'));
    load(files(end).name);
    RunNewPermTests = 0;
else
    RunNewPermTests = 1;
end

%%
if RunNewPermTests
    %% Calculate the true PVB index at each timepoint, relative to drug injection.
    KetTrueAcrossTime = nan(5,size(OutputDMHere,2));
    SalineTrueAcrossTime = nan(5,size(OutputDMHere,2));
    TrueDiff = nan(5,size(OutputDMHere,2));
    
    for tp=1:size(OutputDMHere,2) %Loop across timepoints
        
        %Collapse all saline trials, at this timepoint, across sessions into a single design matrix for regression
        FullDM = cat(1,OutputDMHere{SalineSes,tp});
        SalineDM = [ones(size(FullDM,1),1) FullDM(:,1:2)]; %Constant Term; Difference in Mean; Difference in STD (equation 5; or equation 9 as it will be fitted with lapse term)
        SalineYVar = FullDM(:,3); %Logical for whether the subject chose the left option
        
        %As above for ketamine trials
        FullDM = cat(1,OutputDMHere{KetamineSes,tp});
        KetamineDM = [ones(size(FullDM,1),1) FullDM(:,1:2)];
        KetamineYVar = FullDM(:,3);
        
        %Run the regression analysis, with a lapse term incorporated
        [~,~,~,KetTrue]=...
            scFittingFunctionForLapsesF2(KetamineDM,KetamineYVar,[]);
        %KetTrue: Rows are the different parameters: ConstantTerm; Mean Regressor; STD regressor; Lapse term
        [~,~,~,SalineTrue]=...
            scFittingFunctionForLapsesF2(SalineDM,SalineYVar,[]);
        %SalineTrue: Rows are the different parameters: ConstantTerm; Mean Regressor; STD regressor; Lapse term
        
        TrueDiff(:,tp) = [KetTrue-SalineTrue;
            [KetTrue(3,:)./KetTrue(2,:)   -    SalineTrue(3,:)./SalineTrue(2,:) ]];
        %Calculate the true difference between the ketamine and saline parameters
        %(also add an additional row to calculate the difference in PVB)
        
        %Store the regression output for this timepoint, along with the PVB
        KetTrueAcrossTime(:,tp) = [KetTrue;  KetTrue(3,:)./KetTrue(2,:)];
        SalineTrueAcrossTime(:,tp)= [SalineTrue;   SalineTrue(3,:)./SalineTrue(2,:)];
        
        if sum(tp==linspace(1,size(OutputDMHere,2),11)) & tp>1
            fprintf(['Calculating PVB at each timepoint - ' num2str(round(10*(tp/size(OutputDMHere,2)))*10) ' percent complete \n']);
        end
    end
    
    %% RUN the same analysis on permuted data
    
    FakeDiff = nan(5,size(OutputDMHere,2),n_perm);
    OutPutDMIncludedSessions = OutputDMHere([SalineSes' KetamineSes'],:); %Collapse the design matrix across ketamine and saline sessions
    SalSessIncl = 1:length(SalineSes);
    KetSessIncl = length(SalineSes)+1:length(SalineSes)+length(KetamineSes);
    noSessions = length(SalineSes) + length(KetamineSes);
    
    fprintf('Running permutation test for calculation of PVB at each timepoint using parfor (may take a while) \n');
    
    parfor pp=1:n_perm
        %Permute the order of sessions, hence randomising whether a session will be indexed as 'Ketamine' or 'Saline'
        OutputDMHereShuffled = OutPutDMIncludedSessions(randperm(noSessions),:);
        
        for tp=1:81 %Loop across timepoints, and perform the same analysis above, but on the shuffled data
            FullDM = cat(1,OutputDMHereShuffled{SalSessIncl,tp});
            SalineDM = [ones(size(FullDM,1),1) FullDM(:,1:2)];
            SalineYVar = FullDM(:,3);
            
            FullDM = cat(1,OutputDMHereShuffled{KetSessIncl,tp});
            KetamineDM = [ones(size(FullDM,1),1) FullDM(:,1:2)];
            KetamineYVar = FullDM(:,3);
            
            [~,~,~,KetFake]=...
                scFittingFunctionForLapsesF2(KetamineDM,KetamineYVar,[]);
            
            [~,~,~,SalineFake]=...
                scFittingFunctionForLapsesF2(SalineDM,SalineYVar,[]);
            
            FakeDiff(:,tp,pp) = [KetFake-SalineFake;
                [KetFake(3,:)./KetFake(2,:)   -    SalineFake(3,:)./SalineFake(2,:) ]];
        end
    end
    
    %% Organise the output and save it
    SlidingPVB_RevisionFunctionData.TrueDiff =TrueDiff;
    SlidingPVB_RevisionFunctionData.FakeDiff =FakeDiff;
    SlidingPVB_RevisionFunctionData.KetTrueAcrossTime = KetTrueAcrossTime;
    SlidingPVB_RevisionFunctionData.SalineTrueAcrossTime = SalineTrueAcrossTime;
    
    FN = ['SlidingPVBdata' datestr(datetime,'dd mmm yyyy')];
    save(fullfile(DirToSaveBootstraps,FN),'SlidingPVB_RevisionFunctionData');
    
end
%% Analyse the permuted data with a cluster-based permutation test
 
ClusterFormingThreshold = 0.15;
RealDiff_PVB = SlidingPVB_RevisionFunctionData.TrueDiff(5,:); %Difference between Saline and Ketamine PVB
FakeDiff_PVB = squeeze(SlidingPVB_RevisionFunctionData.FakeDiff(5,:,:))'; %Difference between the two shuffled datasets, for each permutation

%Look for clusters of time, where the |Saline PVB - Ketamine PVB|>ClusterFormingThreshold, in the real data.
TrueClusters  = SCFuncToFindClusters(RealDiff_PVB,ClusterFormingThreshold);
%TrueClusters is a three column matrix. Column 1 is the start of the
%cluster, column 2 is the end. Column 3 is if the statistic was positive or
%negative within the cluster.

for ii=1:size(FakeDiff_PVB,1) %Loop across each permutation of the shuffled data
    ThisPermNullData = FakeDiff_PVB(ii,:);
    FakeClustersH = SCFuncToFindClusters(ThisPermNullData,ClusterFormingThreshold);
    %Find clusters in the permuted data
    
    %Record the longest cluster (i.e. number of time bins) found in this permutation of shuffled data
    if length(FakeClustersH)
        FakeClusterLength(ii) = max(1+(FakeClustersH(:,2)-FakeClustersH(:,1)));
    else
        FakeClusterLength(ii) = 0;
    end
end

%Compare the clusters in the true data, to the null distribution of cluster
%lenghts built from the permuted data
PValH = nan(1,size(TrueClusters,1));
for tc = 1:size(TrueClusters,1) %Loop across the number of clusters in the true data
    FakeLonger = sum(FakeClusterLength>=(1+(TrueClusters(tc,2)-TrueClusters(tc,1))));
    %How many entries in the null distribution are longer clusters
    PValH(tc) = FakeLonger/size(FakeDiff_PVB,1); %Calculate the p-value.
end

[~,ind_max] = max(TrueClusters(:,2)-TrueClusters(:,1)); %Indexes longest cluster
BinNumbersOfLongestCluster = TrueClusters(ind_max,1:2); %Time bins in longest cluster
PValueOfLongestCluster = PValH(ind_max); %Significance value of longest cluster

%% Store the output for figure plotting in python
PythonOutputs.PVBIndexs_Ketamine =   SlidingPVB_RevisionFunctionData.KetTrueAcrossTime(5,:);
PythonOutputs.PVBIndexs_Saline =   SlidingPVB_RevisionFunctionData.SalineTrueAcrossTime(5,:);
PythonOutputs.P_ValueForLongestCluster =   PValueOfLongestCluster;
PythonOutputs.BinNumbersOfLongestCluster =   BinNumbersOfLongestCluster;
