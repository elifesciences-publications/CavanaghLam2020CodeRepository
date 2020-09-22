import numpy as np
import matplotlib.pyplot as plt

# Let's first define parameters for our model, using dictionaries:

def default_pars():
    '''
    Return: dictionary of default parameters for Wong & Wang (2006) reduced model.
    '''

    z = {}
    ### Stimulus Parameters ###
    z['Jext'] = 2.243e-4 # Stimulus input strength [nA/Hz]

    # Working memory (WM) stimulus parameters
    z['mu1'] = 30. # Strength of stimulus 1 [dimensionless]
    z['mu2'] = 0. # Strength of stimulus 2 [dimensionless]

    # Decision making (DM) stimulus parameters
    z['coh'] = 100. # Stimulus coherence [%]
    # z['mu0'] = 20. # Stimulus firing rate [spikes/sec]                                                                  # Wong and Wang used mu0=30Hz, though even robert's code uses mu0 = 20Hz
    z['mu0'] = 30. # Stimulus firing rate [spikes/sec]                                                                  # Wong and Wang used mu0=30Hz, though even robert's code uses mu0 = 20Hz
    z['mu0_slope'] = 50. # Stimulus firing rate [spikes/sec], coherent-dependence proportion                                                                  # Wong and Wang used mu0=30Hz, though even robert's code uses mu0 = 20Hz
    z['fluc_Input1'] = 0. # Stimulus firing rate [spikes/sec]                                                                  # Wong and Wang used mu0=30Hz, though even robert's code uses mu0 = 20Hz
    z['fluc_Input2'] = 0. # Stimulus firing rate [spikes/sec]                                                                  # Wong and Wang used mu0=30Hz, though even robert's code uses mu0 = 20Hz

    ### Network Parameters ###
    # z['JE'] = 0.1561 # self-coupling NMDA strength [nA]
    # z['JI'] = -0.0264 # cross-coupling NMDA strength [nA]
    # z['JE'] =  (1. - 1.*3.9/3.*0.0724) *0.1561 # self-coupling strength [nA]                                            # JE pert only. Maintains spont state but make it more shallow (so ~ the case in spiking circuit model)
    # z['JI'] = -(1. + 0.*3.5/3.*0.2209) *0.0264 # cross-coupling strength [nA]                                            # JI pert only.
    z['JE'] =  (1. - 0.*1.5/3.*0.0724) *0.1561 # self-coupling strength [nA]                                             # default gENMDA x0.97. Circuit Model perturbation x0.9825
    z['JI'] = -(1. + 0.*1.5/3.*0.2209) *0.0264 # cross-coupling strength [nA]                                            # default gENMDA x0.97. Circuit Model perturbation x0.9825
    # z['JE'] =  (1. + 1.*1.5/3.*0.04222542159) *0.1561 # self-coupling strength [nA]                                      # default gINMDA x0.97. Circuit Model perturbation x0.965
    # z['JI'] = -(1. - 1.*1.5/3.*0.25170116478) *0.0264 # cross-coupling strength [nA]                                     # default gINMDA x0.97. Circuit Model perturbation x0.965
    z['BE'] = 9.9026e-4 # self-coupling AMPA strength [nC]
    z['BI'] = -6.5177e-5 # cross-coupling AMPA strength [nC]
    # z['I0'] = 0.34959142025 # background current [nA] (Wong2006 Model value (0.2346) cannot be rederived)
    # z['I0'] =  (1. - 1.*1.75/3.*0.00876)* 0.35268067531 # background current [nA]                                         # default gENMDA x0.97
    # z['I0'] =  (1. - 1.*1.75/3.*0.00876)* 0.35 # background current [nA]                                                 # Modified control I0 to tune the circuit PVBE effects to more reasonable ranges.
    z['I0'] =  (1. - 1.*1.5/3.*0.00876)* 0.351 # background current [nA]                                                 # Modified control I0 to tune the circuit PVBE effects to more reasonable ranges.
    # z['I0'] =  (1. + 0.*3.5/3.*0.00994892396)* 0.35268067531 # background current [nA]                                   # default gINMDA x0.97
    # z['I0'] =  (1. + 1.*1.5/3.*0.00994892396)* 0.351 # background current [nA]                                   # default gINMDA x0.97
    # z['I0'] =  (1. + 1.*2./3.*0.00994892396)* 0.35 # background current [nA]                                   # Modified control I0 to tune the circuit PVBE effects to more reasonable ranges.
    # z['I0'] =  (1. - 2.*0.01317) * 0.34959142025 # background current [nA]
    # z['I0'] = 0.98683 * 0.34959142025 # background current [nA] (Wong2006 Model value (0.2346) cannot be rederived)
    z['tauS'] = 0.1 # Synaptic (NMDA) time constant [sec]
    z['gamma'] = 0.641 # Saturation factor for gating variable
    z['tau0'] = 0.002 # Noise (AMPA) time constant [sec]
    z['sigma'] = 0.007 # Noise magnitude [nA]           (eqns.pdf use 0.006...)
    return z

def default_expt_pars():
    '''
    Return: dictionary of parameters related to experimental simulation.
    '''

    z = {}
    z['Ntrials'] = 5 # Total number of trials
    # z['Ttotal'] = 5. # Total duration of simulation [sec]
    z['Ttotal'] = 3.5 # Total duration of simulation [sec]
    z['Tstim'] = 1. # Time of stimulus 1 onset [sec]
    z['Tdur'] = 2. # Duration of stimulus 1 [sec]
    z['dt'] = 0.0005 # Simulation time step [sec]
    z['dt_smooth'] = 0.02 # Temporal window size for smoothing [sec]
    z['S1_init'] = 0.1 # initial condition for dimension-less gating variable S1
    z['S2_init'] = 0.1 # initial condition for dimension-less gating variable S2
    return z

# Let's define the transfer function for our cells, the firing rate as a function of input current, known as the F-I curve:

def F(x1, x2, pars):#, a=270., b=108., d=0.154):
    '''
    Transfer function: Firing rate as a function of input.

    Parameters:
    I : Input current
    a, b, d : parameters of F(I) curve

    Return: F(I) for vector I
    '''

    a = 239400.*pars['BE'] + 270.
    b =  97000.*pars['BE'] + 108.
    d =    -30.*pars['BE'] + 0.1540
    return (a*x1 - pars['BI']*(-0.0276*x2+0.0106)*(0.5*(np.sign(x2-0.4)+1)) - b)/(1. - np.exp(-d*(a*x1 - pars['BI']*(-0.0276*x2+0.0106)*(0.5*(np.sign(x2-0.4)+1)) - b)))
    # return (a*x1 - pars['BI']*(-0.0276*x2+0.0106)*(0.5*(np.sign(x2-0.4)+1)) - b)*((a*x1 - pars['BI']*(-0.0276*x2+0.0106)*(0.5*(np.sign(x2-0.4)+1)) - b) > 0.4)

def Fp_x20(x1, pars):#, a=270., b=108., d=0.154):
    '''
    First derivative of Transfer function (Firing rate as a function of input).
    Assumes x2=0, for simplicity.

    Parameters:
    I : Input current
    a, b, d : parameters of F(I) curve

    Return: F(I) for vector I
    '''

    a = 239400.*pars['BE'] + 270.
    b =  97000.*pars['BE'] + 108.
    d =    -30.*pars['BE'] + 0.1540
    return a/(1. - np.exp(-d*(a*x1 - b))) - (a*x1 - b)*a*d*np.exp(-d*(a*x1 - b))/(1. - np.exp(-d*(a*x1 - b)))**2

def Fpp_x20(x1, pars):#, a=270., b=108., d=0.154):
    '''
    Second derivative of Transfer function (Firing rate as a function of input).
    Assumes x2=0, for simplicity.

    Parameters:
    I : Input current
    a, b, d : parameters of F(I) curve

    Return: F(I) for vector I
    '''

    a = 239400.*pars['BE'] + 270.
    b =  97000.*pars['BE'] + 108.
    d =    -30.*pars['BE'] + 0.1540
    return a**2*d*np.exp(-d*(a*x1 - b)) / (1. - np.exp(-d*(a*x1 - b)))**3 \
        * (-2.*(1.-np.exp(-d*(a*x1-b))) + (a*x1 - b)*d*(1.+np.exp(-d*(a*x1-b))))

# Now let's make a function that simulates the model in time, for multiple trials:

def run_sim(pars,expt_pars,expt, verbose=False, probability_only=1):
    '''
    Run simulation, for multiple trials.

    Parameters:
    pars : circuit model parameters
    expt_pars : other parameters
    expt: Experimental paradigm: 'WM' (for working memory ) or 'DM' (for decision making)

    Return: dictionary with activity traces
    '''

    # Make lists to store firing rate (r) and gating variable (s)
    S1_traj = []
    S2_traj = []
    r1_traj = []
    r2_traj = []
    r1smooth_traj = []
    r2smooth_traj = []
    # win_120_list = []
    win_120_list = np.zeros(3)

    for i in xrange(expt_pars['Ntrials']): #Loop through trials

        if verbose and (i % 10 == 0):
            print "trial # ", i+1, 'of', expt_pars['Ntrials']

        # ##Set random seed
        # np.random.seed(i)

        #Initialize
        r1smooth = []
        r2smooth = []

        NT = int(expt_pars['Ttotal']/expt_pars['dt'])

        Ieta1 = np.zeros(NT+1)
        Ieta2 = np.zeros(NT+1)
        S1 = np.zeros(NT+1)
        S2 = np.zeros(NT+1)
        r1 = np.zeros(NT)
        r2 = np.zeros(NT)
        dt_input = 0.25
        input_SD_narrow_list = pars['fluc_Input1'] *np.random.randn(int(expt_pars['Tdur']/dt_input))# U(-0.25, 0.25)
        input_SD_broad_list  = pars['fluc_Input2'] *np.random.randn(int(expt_pars['Tdur']/dt_input))# U(-0.25, 0.25)


        # Initialize S1, S2
        S1[0] = expt_pars['S1_init']
        S2[0] = expt_pars['S2_init']

        for t in xrange(0,NT): #Loop through time for a trial

            #---- Stimulus------------------------------------------------------

            if expt == 'WM':
                Istim1 = \
                    ((expt_pars['Tstim']/expt_pars['dt'] < t) & \
                     (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                    ) * (pars['Jext']*pars['mu1']) # To population 1
                Istim2 = \
                    ((expt_pars['Tstim']/expt_pars['dt'] < t) & \
                     (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                    ) * (pars['Jext']*pars['mu2']) # To population 1

            elif expt == 'DM':
                ## mu0=mu0_slope
                # Istim1 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                #           (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                #          ) * (pars['Jext']*pars['mu0']*(1+pars['coh']/100. + input_SD_narrow_list[max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))])) # To population 1
                #          # ) * (pars['Jext']*pars['mu0']*(1+pars['coh']/100.)) # To population 1
                # Istim2 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                #           (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                #          ) * (pars['Jext']*pars['mu0']*(1-pars['coh']/100. + input_SD_broad_list[ max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))])) # To population 2
                #          # ) * (pars['Jext']*pars['mu0']*(1-pars['coh']/100.)) # To population 2
                ## mu0_slope=50Hz, for any mu0
                Istim1 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                          (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                         ) * (pars['Jext']*(pars['mu0']+pars['mu0_slope']*pars['coh']/100. + pars['mu0_slope']*input_SD_narrow_list[max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))])) # To population 1
                         # ) * (pars['Jext']*pars['mu0']*(1+pars['coh']/100.)) # To population 1
                Istim2 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                          (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                         ) * (pars['Jext']*(pars['mu0']-pars['mu0_slope']*pars['coh']/100. + pars['mu0_slope']*input_SD_broad_list[ max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))])) # To population 2
                         # ) * (pars['Jext']*pars['mu0']*(1-pars['coh']/100.)) # To population 2

            # Total synaptic input
            Isyn1 = pars['JE']*S1[t] + pars['JI']*S2[t] + Istim1 + Ieta1[t]
            Isyn2 = pars['JI']*S1[t] + pars['JE']*S2[t] + Istim2 + Ieta2[t]
            #Isyn1, Isyn2 = currents_WM(S1[t], S2[t], pars)

            # Transfer function to get firing rate

            r1[t]  = F(Isyn1, Isyn2, pars)
            r2[t]  = F(Isyn2, Isyn1, pars)

            #---- Dynamical equations -------------------------------------------

            # Mean NMDA-mediated synaptic dynamics updating
            S1[t+1] = S1[t] + expt_pars['dt']*(-S1[t]/pars['tauS'] + (1-S1[t])*pars['gamma']*r1[t]);
            S2[t+1] = S2[t] + expt_pars['dt']*(-S2[t]/pars['tauS'] + (1-S2[t])*pars['gamma']*r2[t]);

            # Ornstein-Uhlenbeck generation of noise in pop1 and 2
            Ieta1[t+1] = Ieta1[t] + \
                            (expt_pars['dt']/pars['tau0']) * (pars['I0']-Ieta1[t]) + \
                            np.sqrt(expt_pars['dt']/pars['tau0'])*pars['sigma']*np.random.randn()
            Ieta2[t+1] = Ieta2[t] + \
                            (expt_pars['dt']/pars['tau0']) * (pars['I0']-Ieta2[t]) + \
                            np.sqrt(expt_pars['dt']/pars['tau0'])*pars['sigma']*np.random.randn()

        smooth_wind = int(expt_pars['dt_smooth']/expt_pars['dt'])

        r1smooth = np.array([np.mean(r1[j:j+smooth_wind]) for j in xrange(NT)])
        r2smooth = np.array([np.mean(r2[j:j+smooth_wind]) for j in xrange(NT)])

        # Compute committed choice here instead of in the simulation page, if we do not want to output any trajs.
        # t_mean_end = np.minimum(expt_pars['Tstim'] + expt_pars['Tdur'] + 2., expt_pars['Ttotal'])
        # t_mean_dur = 0.1
        # t_list_to_mean = t_vec_list[((t_mean_end-t_mean_dur)/dt_vec): (t_mean_end/dt_vec)]
        # r_Pe1_end_temp_mean = np.mean(r1smooth[int((t_mean_end-t_mean_dur)/expt_pars['dt']): int(t_mean_end/expt_pars['dt'])])
        # r_Pe2_end_temp_mean = np.mean(r2smooth[int((t_mean_end-t_mean_dur)/expt_pars['dt']): int(t_mean_end/expt_pars['dt'])])
        # print r_Pe1_end_temp_mean - r_Pe2_end_temp_mean
        # if   r_Pe1_end_temp_mean > r_Pe2_end_temp_mean:
        #     win_120_temp = 0
        # elif r_Pe1_end_temp_mean < r_Pe2_end_temp_mean:
        #     win_120_temp = 1
        # elif r_Pe1_end_temp_mean ==r_Pe2_end_temp_mean:
        #     win_120_temp = 2
        t_DM = np.minimum(expt_pars['Tstim'] + expt_pars['Tdur'] + 0.*2. - 2.*expt_pars['dt_smooth'], expt_pars['Ttotal'])
        if   r1smooth[t_DM/expt_pars['dt']] >  r2smooth[t_DM/expt_pars['dt']]:
            # win_120_temp = 0                                                                                          # If I record and report data from each trial
            win_120_list[0] +=1
        elif r1smooth[t_DM/expt_pars['dt']] <  r2smooth[t_DM/expt_pars['dt']]:
            # win_120_temp = 1                                                                                          # If I record and report data from each trial
            win_120_list[1] +=1
        elif r1smooth[t_DM/expt_pars['dt']] == r2smooth[t_DM/expt_pars['dt']]:
            # win_120_temp = 2                                                                                          # If I record and report data from each trial
            win_120_list[2] +=1
        else:
            print "ERROR! no winner is decided!"



        ## If I record and report data from each trial
        if probability_only ==0:
            S1_traj.append(S1)
            S2_traj.append(S2)
            r1_traj.append(r1)
            r2_traj.append(r2)
            r1smooth_traj.append(r1smooth)
            r2smooth_traj.append(r2smooth)
            # win_120_list.append(win_120_temp)

    # tvec = expt_pars['dt']*np.arange(NT)
    # z = {'S1':S1_traj, 'S2':S2_traj, # NMDA gating variables
    #      'r1':r1_traj, 'r2':r2_traj, # Firing rates
    #      'r1smooth':r1smooth_traj, 'r2smooth':r2smooth_traj, # smoothed firing rates
    #      't':tvec}

    if probability_only==1:
        z = {'win_120_list':win_120_list}
    elif probability_only==0:
        tvec = expt_pars['dt']*np.arange(NT)
        z = {'S1':S1_traj, 'S2':S2_traj, # NMDA gating variables
         'r1':r1_traj, 'r2':r2_traj, # Firing rates
         'r1smooth':r1smooth_traj, 'r2smooth':r2smooth_traj, # smoothed firing rates
         't':tvec}



    return z

# Now let's build some functions that let us analyze the model using tools of dynamical systems theory: phase plane, nullclines fixed points, and flow fields.



def run_sim_reg(pars,expt_pars,expt, verbose=False, probability_only=1):
    '''
    Run simulation, for multiple trials.

    Parameters:
    pars : circuit model parameters
    expt_pars : other parameters
    expt: Experimental paradigm: 'WM' (for working memory ) or 'DM' (for decision making)

    Return: dictionary with activity traces
    '''

    # Make lists to store firing rate (r) and gating variable (s)
    S1_traj = []
    S2_traj = []
    r1_traj = []
    r2_traj = []
    r1smooth_traj = []
    r2smooth_traj = []
    winner_list = []
    win_120_list = np.zeros(3)

    for i in xrange(expt_pars['Ntrials']): #Loop through trials
        # print i

        if verbose and (i % 10 == 0):
            print "trial # ", i+1, 'of', expt_pars['Ntrials']

        # ##Set random seed
        # np.random.seed(i)

        #Initialize
        r1smooth = []
        r2smooth = []

        NT = int(expt_pars['Ttotal']/expt_pars['dt'])

        Ieta1 = np.zeros(NT+1)
        Ieta2 = np.zeros(NT+1)
        S1 = np.zeros(NT+1)
        S2 = np.zeros(NT+1)
        r1 = np.zeros(NT)
        r2 = np.zeros(NT)
        dt_input = 0.25

        flag_2_streams_constrained = 0                                                                                  # Flag for whether the within-session constraints of the two streams are fulfilled.
        ## Regression/Standard trials. Not used for Narrow-Broad trials.
        # input_SD_narrow_list = input_SD_narrow*np.random.randn(input_n_t)
        # input_SD_broad_list = input_SD_broad*np.random.randn(input_n_t)
        input_Z_narrow = np.random.uniform (-0.25, 0.25)
        input_Z_broad  = np.random.uniform (-0.25, 0.25)
        mean_narrow_stream_reg = 2.*0.5 + pars['fluc_Input1']*input_Z_narrow                                                # Standard trials: Narrow stream uniformly distributed in [0.47,0.53].
        mean_broad_stream_reg  = 2.*0.5 + pars['fluc_Input2']*input_Z_broad                                                   # Standard trials: Broad stream uniformly distributed in [0.44,0.56].





        while flag_2_streams_constrained==0:
            flag_2_streams_constrained = 1
            # generating mean/Z values are fixed and not re-computed if trial stimuli streams are rejected, just the particular stream stimuli over time.
            input_SD_narrow_list = pars['fluc_Input1']*np.random.randn(int(expt_pars['Tdur']/dt_input))
            input_SD_broad_list  = pars['fluc_Input2']*np.random.randn(int(expt_pars['Tdur']/dt_input))

            ######################### Standard/Regression Trials
            if any(mean_narrow_stream_reg+input_SD_narrow_list < 2.*0.01):                                                  # None of the narrow/correct bars are less than 0.01.
                flag_2_streams_constrained = 0
                # print 'a'
                # print mean_corect_stream+input_SD_narrow_list
                # print mean_corect_stream+input_SD_narrow_list < 2.*0.01
                # print any(mean_corect_stream+input_SD_narrow_list < 2.*0.01)
            elif any(mean_narrow_stream_reg+input_SD_narrow_list > 2.*0.99):                                                  # None of the narrow/correct bars are larger than 0.99.
                flag_2_streams_constrained = 0
                # print 'b'
            elif np.abs(np.std(input_SD_narrow_list)-pars['fluc_Input1']) > 2.*0.04:                                           # Sampled Narrow/correct SD is within a tolerance of 4 of the generating Narrow  SD.
                flag_2_streams_constrained = 0
                # print 'c'
            elif any(mean_broad_stream_reg+input_SD_broad_list < 2.*0.01):                                                 # None of the broad/incorrect bars are less than 0.01.
                flag_2_streams_constrained = 0
                # print 'd'
            elif any(mean_broad_stream_reg+input_SD_broad_list > 2.*0.99):                                                 # None of the broad/incorrect bars are larger than 0.99.
                flag_2_streams_constrained = 0
                # print 'e'
            elif np.abs(np.std(input_SD_broad_list)-pars['fluc_Input2']) > 2.*0.04:                                             # Sampled broad/incorrect SD is within a tolerance of 4 of the generating broad SD.
                flag_2_streams_constrained = 0
                # print 'f'



        # Initialize S1, S2
        S1[0] = expt_pars['S1_init']
        S2[0] = expt_pars['S2_init']

        for t in xrange(0,NT): #Loop through time for a trial

            #---- Stimulus------------------------------------------------------

            if expt == 'WM':
                Istim1 = \
                    ((expt_pars['Tstim']/expt_pars['dt'] < t) & \
                     (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                    ) * (pars['Jext']*pars['mu1']) # To population 1
                Istim2 = \
                    ((expt_pars['Tstim']/expt_pars['dt'] < t) & \
                     (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                    ) * (pars['Jext']*pars['mu2']) # To population 1

            elif expt == 'DM':
                ## mu0=mu0_slope
                # Istim1 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                #           (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                #          ) * (pars['Jext']*pars['mu0']*(1+pars['coh']/100. + input_SD_narrow_list[max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))])) # To population 1
                #          # ) * (pars['Jext']*pars['mu0']*(1+pars['coh']/100.)) # To population 1
                # Istim2 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                #           (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                #          ) * (pars['Jext']*pars['mu0']*(1-pars['coh']/100. + input_SD_broad_list[ max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))])) # To population 2
                #          # ) * (pars['Jext']*pars['mu0']*(1-pars['coh']/100.)) # To population 2
                ## mu0_slope=50Hz, for any mu0
                # ## With coh
                # Istim1 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                #           (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                #          ) * (pars['Jext']*(pars['mu0']+pars['mu0_slope']*pars['coh']/100. +pars['mu0_slope']*pars['fluc_Input1']*input_Z_narrow + pars['mu0_slope']*input_SD_narrow_list[max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))])) # To population 1
                #          # ) * (pars['Jext']*pars['mu0']*(1+pars['coh']/100.)) # To population 1
                # Istim2 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                #           (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                #          ) * (pars['Jext']*(pars['mu0']-pars['mu0_slope']*pars['coh']/100. +pars['mu0_slope']*pars['fluc_Input2']*input_Z_broad + pars['mu0_slope']*input_SD_broad_list[ max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))])) # To population 2
                #          # ) * (pars['Jext']*pars['mu0']*(1-pars['coh']/100.)) # To population 2
                Istim1 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                          (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                         ) * (pars['Jext']*(pars['mu0'] + pars['mu0_slope']* (mean_narrow_stream_reg - 2.*0.5 + input_SD_narrow_list[max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))]))) # To population 1
                         # ) * (pars['Jext']*pars['mu0']*(1+pars['coh']/100.)) # To population 1
                Istim2 = ((expt_pars['Tstim']/expt_pars['dt']<t) & \
                          (t<(expt_pars['Tstim']+expt_pars['Tdur'])/expt_pars['dt'])
                         ) * (pars['Jext']*(pars['mu0'] + pars['mu0_slope']* (mean_broad_stream_reg - 2.*0.5 + input_SD_broad_list[max(0, min(7, int((t*expt_pars['dt']-expt_pars['Tstim'])/dt_input)))]))) # To population 2
                         # ) * (pars['Jext']*pars['mu0']*(1-pars['coh']/100.)) # To population 2


            # Total synaptic input
            Isyn1 = pars['JE']*S1[t] + pars['JI']*S2[t] + Istim1 + Ieta1[t]
            Isyn2 = pars['JI']*S1[t] + pars['JE']*S2[t] + Istim2 + Ieta2[t]
            #Isyn1, Isyn2 = currents_WM(S1[t], S2[t], pars)

            # Transfer function to get firing rate

            r1[t]  = F(Isyn1, Isyn2, pars)
            r2[t]  = F(Isyn2, Isyn1, pars)

            #---- Dynamical equations -------------------------------------------

            # Mean NMDA-mediated synaptic dynamics updating
            S1[t+1] = S1[t] + expt_pars['dt']*(-S1[t]/pars['tauS'] + (1-S1[t])*pars['gamma']*r1[t]);
            S2[t+1] = S2[t] + expt_pars['dt']*(-S2[t]/pars['tauS'] + (1-S2[t])*pars['gamma']*r2[t]);

            # Ornstein-Uhlenbeck generation of noise in pop1 and 2
            Ieta1[t+1] = Ieta1[t] + \
                            (expt_pars['dt']/pars['tau0']) * (pars['I0']-Ieta1[t]) + \
                            np.sqrt(expt_pars['dt']/pars['tau0'])*pars['sigma']*np.random.randn()
            Ieta2[t+1] = Ieta2[t] + \
                            (expt_pars['dt']/pars['tau0']) * (pars['I0']-Ieta2[t]) + \
                            np.sqrt(expt_pars['dt']/pars['tau0'])*pars['sigma']*np.random.randn()

        smooth_wind = int(expt_pars['dt_smooth']/expt_pars['dt'])

        r1smooth = np.array([np.mean(r1[j:j+smooth_wind]) for j in xrange(NT)])
        r2smooth = np.array([np.mean(r2[j:j+smooth_wind]) for j in xrange(NT)])

        # Compute committed choice here instead of in the simulation page, if we do not want to output any trajs.
        # t_mean_end = np.minimum(expt_pars['Tstim'] + expt_pars['Tdur'] + 2., expt_pars['Ttotal'])
        # t_mean_dur = 0.1
        # t_list_to_mean = t_vec_list[((t_mean_end-t_mean_dur)/dt_vec): (t_mean_end/dt_vec)]
        # r_Pe1_end_temp_mean = np.mean(r1smooth[int((t_mean_end-t_mean_dur)/expt_pars['dt']): int(t_mean_end/expt_pars['dt'])])
        # r_Pe2_end_temp_mean = np.mean(r2smooth[int((t_mean_end-t_mean_dur)/expt_pars['dt']): int(t_mean_end/expt_pars['dt'])])
        # print r_Pe1_end_temp_mean - r_Pe2_end_temp_mean
        # if   r_Pe1_end_temp_mean > r_Pe2_end_temp_mean:
        #     win_120_temp = 1
        # elif r_Pe1_end_temp_mean < r_Pe2_end_temp_mean:
        #     win_120_temp = 2
        # elif r_Pe1_end_temp_mean ==r_Pe2_end_temp_mean:
        #     win_120_temp = 0
        t_DM = np.minimum(expt_pars['Tstim'] + expt_pars['Tdur'] + 0.*2. - 2.*expt_pars['dt_smooth'], expt_pars['Ttotal'])
        if   r1smooth[t_DM/expt_pars['dt']] >  r2smooth[t_DM/expt_pars['dt']]:
            win_120_temp = 1                                                                                          # If I record and report data from each trial
            win_120_list[0] +=1
        elif r1smooth[t_DM/expt_pars['dt']] <  r2smooth[t_DM/expt_pars['dt']]:
            win_120_temp = 2                                                                                          # If I record and report data from each trial
            win_120_list[1] +=1
        elif r1smooth[t_DM/expt_pars['dt']] == r2smooth[t_DM/expt_pars['dt']]:
            win_120_temp = 0                                                                                          # If I record and report data from each trial
            win_120_list[2] +=1
        else:
            print "ERROR! no winner is decided!"

        ## If I record and report data from each trial
        if probability_only ==0:
            S1_traj.append(S1)
            S2_traj.append(S2)
            r1_traj.append(r1)
            r2_traj.append(r2)
            r1smooth_traj.append(r1smooth)
            r2smooth_traj.append(r2smooth)
            winner_list.append(win_120_temp)

    # tvec = expt_pars['dt']*np.arange(NT)
    # z = {'S1':S1_traj, 'S2':S2_traj, # NMDA gating variables
    #      'r1':r1_traj, 'r2':r2_traj, # Firing rates
    #      'r1smooth':r1smooth_traj, 'r2smooth':r2smooth_traj, # smoothed firing rates
    #      't':tvec}

    if probability_only==1:
        z = {'win_120_list':win_120_list}
    elif probability_only==0:
        tvec = expt_pars['dt']*np.arange(NT)
        z = {'S1':S1_traj, 'S2':S2_traj, # NMDA gating variables
         'r1':r1_traj, 'r2':r2_traj, # Firing rates
         'r1smooth':r1smooth_traj, 'r2smooth':r2smooth_traj, # smoothed firing rates
         't':tvec, 'winner':winner_list}

    return z



def currents_WM(S1,S2,pars, fluc_Input1, fluc_Input2):
    '''
    Input currents for working memory task.
    '''

    I1 = (pars['JE']*S1 + pars['JI']*S2) + pars['I0'] + pars['mu1']*pars['Jext'] * (1.+fluc_Input1)
    I2 = (pars['JI']*S1 + pars['JE']*S2) + pars['I0'] + pars['mu2']*pars['Jext'] * (1.+fluc_Input2)
    return I1, I2

def currents_DM(S1,S2,pars, fluc_Input1, fluc_Input2):
    '''
    Input currents for decision making task.
    '''

    # mu0 = mu0_slope
    # I1 = (pars['JE']*S1 + pars['JI']*S2) + pars['I0'] + pars['mu0']*pars['Jext']*(1 + pars['coh']/100.) * (1.+fluc_Input1)
    # I2 = (pars['JI']*S1 + pars['JE']*S2) + pars['I0'] + pars['mu0']*pars['Jext']*(1 - pars['coh']/100.) * (1.+fluc_Input2)
    ## mu0_slope = 50, constant
    I1 = (pars['JE']*S1 + pars['JI']*S2) + pars['I0'] + pars['Jext']*(pars['mu0'] + pars['mu0_slope']*(pars['coh']+fluc_Input1)/100.)
    I2 = (pars['JI']*S1 + pars['JE']*S2) + pars['I0'] + pars['Jext']*(pars['mu0'] + pars['mu0_slope']*(-pars['coh']+fluc_Input2)/100.)
    return I1, I2

def Sderivs(S1,S2,I1,I2,pars):
    '''
    Time derivatives for S variables (dS/dt).
    '''
    dS1dt = -S1/pars['tauS'] + pars['gamma']*(1.0-S1)*F(I1, I2, pars)
    dS2dt = -S2/pars['tauS'] + pars['gamma']*(1.0-S2)*F(I2, I1, pars)
    return dS1dt, dS2dt

def plot_nullcline(ax,x,y,z,color='k',label=''):
    '''
    Nullclines.
    '''
    nc = ax.contour(x,y,z,levels=[0],colors=color, fontsize=fontsize_legend) # S1 nullcline
    # nc.collections[0].set_label(label)
    return nc

def plot_flow_field(ax,x,y,dxdt,dydt,n_skip=1,scale=None,facecolor='gray'):
    '''
    Vector flow fields.
    '''
    v = ax.quiver(x[::n_skip,::n_skip], y[::n_skip,::n_skip],
              dxdt[::n_skip,::n_skip], dydt[::n_skip,::n_skip],
              angles='xy', scale_units='xy', scale=scale,facecolor=facecolor)
    return v

# def plot_phase_plane(pars, fluc_Input1, fluc_Input2, expt, ax=None, color1='orange', color2='green'):
def plot_phase_plane(pars, fluc_Input1, fluc_Input2, expt, ax=None):
    '''
    Phase plane plot with nullclines and flow fields.
    '''

    if ax is None:
        ax = plt.gca()

    # Make 2D grid of (S1,S2) values
    S_vec = np.linspace(0.001,0.999,200) # things break down at S=0 or S=1
    S1,S2 = np.meshgrid(S_vec,S_vec)

    if expt == 'WM':
        I1, I2 = currents_WM(S1,S2,pars, fluc_Input1, fluc_Input2)
    elif expt == 'DM':
        I1, I2 = currents_DM(S1,S2,pars, fluc_Input1, fluc_Input2)
    else:
        print "Must define expt as 'WM' or 'DM'"
        return 0

    dS1dt, dS2dt = Sderivs(S1, S2, I1, I2, pars)

    # plot_nullcline(ax, S2, S1, dS1dt, color=color_NB[1], label='S1 nullcline') # S1 nullcline
    # plot_nullcline(ax, S2, S1, dS2dt, color=color_NB[0], label='S2 nullcline') # S2 nullcline
    # ax.contour(S2,S1,dS1dt, levels=[0],colors=(140./255, 81./255, 10./255), fontsize=fontsize_legend)
    # ax.contour(S2,S1,dS2dt, levels=[0],colors=(128./255, 177./255, 211./255), fontsize=fontsize_legend)
    ax.contour(S2,S1,dS1dt, levels=[0],colors=[color_NB[1]], fontsize=fontsize_legend, zorder=10)
    ax.contour(S2,S1,dS2dt, levels=[0],colors=[color_NB[0]], fontsize=fontsize_legend, zorder=10)
    plt.legend(loc=1)

    plot_flow_field(ax,S2,S1,dS2dt,dS1dt,n_skip=12,scale=40)

    ax.set_xlabel('$S_N$', fontsize=8.)
    ax.set_ylabel('$S_B$', fontsize=8.)
    ax.set_xlim(0,0.8)
    ax.set_ylim(0,0.8)
    ax.set_aspect('equal')





