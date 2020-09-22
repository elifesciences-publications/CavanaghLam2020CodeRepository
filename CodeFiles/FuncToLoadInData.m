function [NameOfFileForThisSubj] = FuncToLoadInData(SubjectToTest,DrugDay,CodesFileLocations,DataFileLocations)
%Input some information, and this function will tell you which data file you should load
%Inputs: 
%SubjectToTest = Which Subject to Run the analysis on?
%DrugDay: 1 if it is the pharmacological experiment; 0 if it is a standard session
%CodesFileLocations
%DataFileLocations
%% Add the function code files to the path
restoredefaultpath; %Restore default matlab path, so matlab doesn't look elsewhere for the functions called
addpath(CodesFileLocations);
%% Find the data files within the data folder
cd(DataFileLocations);

if DrugDay %I.e. pharmacological experiment session (Figure 8)
    files=dir('DrugDay*.mat');
    PointToLookInFileNameForSubjInitial = 28;
else %I.e. standard experiment session (Figure 2-4)
    files=dir('NonDrugDay*.mat');
    PointToLookInFileNameForSubjInitial = 32;
end

%% Extract info (date and subject) about relevant files

for t=1:length(files) %Loop across all potentially eligible files
    DatesOfFiles(t,:) = files(t).date;
    DateNumber(t,:) = datenum(DatesOfFiles(t,:));
    if strcmp(files(t).name(PointToLookInFileNameForSubjInitial),'H');
        Subject{t} = 'Harry';
    elseif strcmp(files(t).name(PointToLookInFileNameForSubjInitial),'A');
        Subject{t} = 'Alfie';
    end
end
%% Find the file name that should be loaded
PosFiles = strcmp(Subject,SubjectToTest); %Which files match the subject entered as an input to the function.
PosFilesFiles = files(PosFiles);
[~,i] = max(DateNumber(PosFiles)); %Choose the most recent file. 
NameOfFileForThisSubj = PosFilesFiles(i).name;

end

