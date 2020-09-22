import numpy as np
import matplotlib.pyplot as plt
import functions_MFmodel_Sim
import copy

########################################################################################################################
## Load params and functions
execfile('functions_MFmodel_Sim.py')
pars = copy.copy(default_pars())
expt_pars = copy.copy(default_expt_pars())
## Modify parameters
pars['mu1'] = pars['mu0']
pars['mu2'] = pars['mu0']
pars['coh'] = 0.*2.
expt_pars['Ntrials'] = 20000


fluc_Input1_list = [0., 2.*0.12, 2.*0.24*2.]
fluc_Input2_0 = 0.
pars['fluc_Input1'] = fluc_Input1_list[0]
pars['fluc_Input2'] = fluc_Input1_list[2]


results = run_sim(pars,expt_pars,expt='DM') # expt = 'WM' for working memory, 'DM' for decision making'


# ## Plot trajectories
# fig_sim_traj = plt.figure(figsize=(8,10.5))
# ax_1 = fig_sim_traj.add_subplot(211)
# for i in xrange(expt_pars['Ntrials']):
#     ax_1.plot(results['t'], results['r1smooth'][i], c='r', alpha=0.5)
#     ax_1.plot(results['t'], results['r2smooth'][i], c='b', alpha=0.5)
# ax_1.set_xlabel('Time [s]')
# ax_1.set_ylabel('Firing rate [Hz]')
#
# fig_sim_traj.savefig('Sim_Traj.pdf')


## Sort by the higher firing rate at the last 0.1s
win_120_count_list = results['win_120_list']/expt_pars['Ntrials']       # ans = 1 or 2 or 0
# win_120_count_list = np.zeros(3)       # ans = 1 or 2 or 0
# t_mean_end = np.maximum(expt_pars['Tstim'] + expt_pars['Tdur'] + 2., expt_pars['Ttotal'])
# t_mean_dur = 0.1
# t_list_to_mean = t_vec_list[((t_mean_end-t_mean_dur)/dt_vec): (t_mean_end/dt_vec)]
# ## If compute choices from trajs here.
# for i_trial in range((expt_pars['Ntrials'])):
#     r_Pe1_end_temp_mean = np.mean(results['r1smooth'][i_trial][int((t_mean_end-t_mean_dur)/expt_pars['dt_smooth']): int(t_mean_end/expt_pars['dt_smooth'])])
#     r_Pe2_end_temp_mean = np.mean(results['r2smooth'][i_trial][int((t_mean_end-t_mean_dur)/expt_pars['dt_smooth']): int(t_mean_end/expt_pars['dt_smooth'])])
#     if   r_Pe1_end_temp_mean > r_Pe2_end_temp_mean:
#         win_120_count_list[0] += 1./float(expt_pars['Ntrials'])
#     elif r_Pe1_end_temp_mean < r_Pe2_end_temp_mean:
#         win_120_count_list[1] += 1./float(expt_pars['Ntrials'])
#     elif r_Pe1_end_temp_mean ==r_Pe2_end_temp_mean:
#         win_120_count_list[2] += 1./float(expt_pars['Ntrials'])
## If computed choices in the function.
# win_120_count_list[0] = float(results['win_120_list'].count(0))/float(expt_pars['Ntrials'])
# win_120_count_list[1] = float(results['win_120_list'].count(1))/float(expt_pars['Ntrials'])
# win_120_count_list[2] = float(results['win_120_list'].count(2))/float(expt_pars['Ntrials'])


#-----------------------------------------------------------------------------------------------------------------------### Plots
figPK = plt.figure(figsize=(8,10.5))
# variable_list = ['Control', 'gEI-3p', 'gEE-2p']         #Manually used variable.
variable_list = ['Control']         #Manually used variable.
color_list    = ['r', 'g', 'b']

plt.subplot(311)
index_bar = np.arange(len(variable_list))
bar_width = 0.23
bar_opacity = 0.8

rects1 = plt.bar(index_bar, win_120_count_list[0]-0.5, bar_width, alpha=bar_opacity, color='b', label='Narrow')
rects2 = plt.bar(index_bar+bar_width, win_120_count_list[1]-0.5, bar_width, alpha=bar_opacity, color='g', label='Broad')
# rects3 = plt.bar(index_bar+2.*bar_width, win_120_count_list[2], bar_width, alpha=bar_opacity, color='r', label='Undecided')
#figPK.ylim([-1.,1.])
# plt.xlabel('Conditions/Perturbations')
plt.ylabel('Choice Probability - 0.5')
plt.title('Choice Probability - 0.5')
# axPK1.set_yscale('log')
plt.xticks(index_bar + 1.*bar_width, variable_list)
plt.legend(loc=4)

figPK.savefig('MF_Model_Sim_NB.pdf')















