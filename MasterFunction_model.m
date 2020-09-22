clearvars; close all;
%% Path locations of the matlab codes and data files (Please edit)
% MainDirectory = '/Users/seancavanagh/Downloads/CavanaghLam2020Repository';
MainDirectory = 'C:/Users/nhl8/Desktop/Murray Lab/Bar Task/CavanaghLam2020Repository';
%% Run the analyses
addpath(fullfile(MainDirectory,'CodeFiles'))


%%% Run default spiking circuit models (main figures). For Figures 5&7.
model_folder = 'Default_Models';

model_condition = 'Control';
[PythonVars_model_control] = StandardSessions_model(model_condition, MainDirectory, model_folder); 

model_condition = 'Lowered_EI';
[PythonVars_model_lowered_EI] = StandardSessions_model(model_condition, MainDirectory, model_folder); 

model_condition = 'Elevated_EI';
[PythonVars_model_elevated_EI] = StandardSessions_model(model_condition, MainDirectory, model_folder); 

model_condition = 'Sensory_deficit';
[PythonVars_model_sensory_deficit] = StandardSessions_model(model_condition, MainDirectory, model_folder); 

%%% Run control mean-field models (main figures). For Figure 6.
model_condition = 'Control_MF';
[PythonVars_model_control_MF] = StandardSessions_model(model_condition, MainDirectory, model_folder); 



%%% For fig 5S2: more vs less net total evidence.
model_folder = 'Fig5S2';
model_condition = 'Control_upper_half_total_evi';
[PythonVars_model_control_more_total_evi_MF] = StandardSessions_model(model_condition, MainDirectory, model_folder); 

model_condition = 'Control_lower_half_total_evi';
[PythonVars_model_control_less_total_evi_MF] = StandardSessions_model(model_condition, MainDirectory, model_folder); 

%%% Parameterized Sensory deficit
model_folder = 'Parameterized_sensory_deficit';
sensory_coeff_precent = 80;             % Specify what percent of sensory deficit to use. 100 <=> control, 80 <=> inputs are only 80% as strong as control, etc.
model_condition = sprintf('Sensory_deficit_sens_coeff=0.%d', sensory_coeff_precent);
[PythonVars_model_parameterized_sensory_deficit] = StandardSessions_model(model_condition, MainDirectory, model_folder); 

%%% Parameterized E/I perturbation (gNMDA_E->E/I)
model_folder = 'Parameterized_EI_pert';
gNMDA_i_pert_rel = 0.;% Specify what percent of gNMDA_I (NMDA conductance to I-cells) to use. 0 <=> no perturbation.
gNMDA_e_pert_rel = 0.5;% Specify what percent of gNMDA_I (NMDA conductance to I-cells) to use. 0 <=> no perturbation.
model_condition = sprintf('EI_pert_gNMDAI_pert_rel=%.2f_gNMDAE_pert_rel=%.2f', gNMDA_i_pert_rel, gNMDA_e_pert_rel);
[PythonVars_model_parameterized_EI_pert] = StandardSessions_model(model_condition, MainDirectory, model_folder); 


%%% Note: For analysis with lapse incoporated downstream of choice data, please see MasterFunction_model_lapse_downstream for downstream lapse.












