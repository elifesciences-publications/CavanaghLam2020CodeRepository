clearvars; close all;
%% Path locations of the matlab codes and data files (Please edit)
% MainDirectory = '/Users/seancavanagh/Downloads/CavanaghLam2020Repository';
MainDirectory = 'C:/Users/nhl8/Desktop/Murray Lab/Bar Task/CavanaghLam2020Repository';
%% Which Subject to Run the analysis on?
SubjectToTest = 'Both'; %This will pool the data for both subjects, as in the main figures
%SubjectToTest = 'Alfie'; %This will show the data for Subject A, as in the supplementary figures
%SubjectToTest = 'Harry'; %This will show the data for Subject H, as in the supplementary figures
%% Run the analyses
addpath(fullfile(MainDirectory,'CodeFiles'))

% This function reproduces all of the results for Figures 2-4
[PythonVars_SS] = StandardSessions(SubjectToTest,MainDirectory); 

% This function reproduces all of the results for Figure 8
[PythonVars_DS] = DrugSessions(SubjectToTest,MainDirectory); 




