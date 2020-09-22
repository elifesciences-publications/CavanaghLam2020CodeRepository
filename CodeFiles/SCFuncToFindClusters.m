function [ ClusterTable ] = SCFuncToFindClusters( TrueStat,ClusterFormingThreshold )
%ClusterTable is a three column matrix. Column 1 is the start of the
%cluster, column 2 is the end. Column 3 is if the statistic was positive or
%negative within the cluster. 
    trueClusterLengthPOS_Stat = TrueStat>ClusterFormingThreshold; %is the true statistic above the threshold, and positive?
    trueClusterLengthNEG_Stat = -TrueStat>ClusterFormingThreshold; %is the true statistic above the threshold, and negative?
    
    isPOS=find(diff([0 trueClusterLengthPOS_Stat])==1);
    iePOS=find(diff([trueClusterLengthPOS_Stat 0])==-1);  
        
    isNEG=find(diff([0 trueClusterLengthNEG_Stat])==1);
    ieNEG=find(diff([trueClusterLengthNEG_Stat 0])==-1);  
    
    ClusterTable = cat(2,cat(2,isPOS,isNEG)',cat(2,iePOS,ieNEG)'...
        ,cat(1,ones(length(isPOS),1),-1*ones(length(isNEG),1)));
  
end

