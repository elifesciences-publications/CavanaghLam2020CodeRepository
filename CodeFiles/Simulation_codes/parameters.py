# === Simulation-related parameters ====
# clear()                   # Don't think I need this, Think Thiago used it as he is using the same node to loop over simulation runs...
#defaultclock.t = 0.0*ms
# defaultclock.t = 0.0*ms
defaultclock.dt = 0.02*ms # simulation step length [ms]
dt_default = 0.02*ms # simulation step length [ms]
simtime     = 5000.0*ms     # total simulation time [ms]
simulation_clock=Clock(dt=dt_default)   # clock for the time steps

 
# === Parameters determining model dynamics =====

# network structure
NE = int(1600*1)
NI = int(400*1)
f = 0.15
N1 = (int)(f*NE)
N2 = (int)(f*NE)
N0=NE-N1-N2
wp = 1.84
wm = 1.0-f*(wp-1.0)/(1.0-f)

# pyramidal cells 
Cm_e = 0.5*nF     # [nF] total capacitance 
gl_e = 25.0*nS    # [ns] total leak conductance 
El_e = -70.0*mV   # [mV] leak reversal potential 
Vt_e = -50.0*mV   # [mV] threshold potential
Vr_e = -55.0*mV   # [mV] reset potential 
tr_e = 2.0*ms   # [ms] refractory time 
 
# interneuron cells 
Cm_i = 0.2*nF   # [nF] total capacitance 
gl_i = 20.0*nS   # [ns] total leak conductance 
El_i = -70.0*mV    # [mV] leak reversal potential 
Vt_i = -50.0*mV    # [mV] threshold potential 
Vr_i = -55.0*mV    # [mV] reset potential 
tr_i = 1.0*ms     # [ms] refractory time

# AMPA receptor
E_ampa = 0.0*mV     # [mV] synaptic reversial potential 
t_ampa = 2.0*ms     # [ms] exponential decay time constant  
g_ext_e = 2.07*nS     # [nS] conductance from external to pyramidal cells # original 2.1
g_ext_i = 1.62*nS    # [nS] conductance from external to interneuron cells
g_ampa_i = 0.04*nS   # [nS] conductance from pyramidal to interneuron cells
g_ampa_e = 0.05*nS   # [nS] conductance from pyramidal to pyramidal cells
wext_e = g_ext_e/g_ampa_e # normalized external conductance to E cells
wext_i = g_ext_i/g_ampa_i # normalized external conductance to I cells
 
# GABA receptor
E_gaba = -70.0*mV      # [mV] synaptic reversial potential 
t_gaba = 5.0*ms   # [ms] exponential decay time constant
g_gaba_e = 1.3*nS  # [nS] conductance to pyramidal cells
g_gaba_i = 1.0*nS  # [nS] conductance to interneuron cells


# NMDA receptor
E_nmda = 0.0*mV     # [mV] synaptic reversial potential 
t_nmda = 100.0*ms   # [ms] decay time of NMDA currents 
t_x = 2.0*ms   # [ms] controls the rise time of NMDAR channels 
alpha = 0.5*kHz      # [kHz] controls the saturation properties of NMDAR channels

# Conductance strength between Pyramidal neurons groups, for control/ reduced EE/ reduced EI
g_nmda_e = 0.165* (1. - 0.*0.75*0.0175) *nS             # Control or reduced EE
g_nmda_i = 0.13 * (1. - 0.*0.75*0.035) *nS             # Control or reduced EI

a=0.062/mV  # control the voltage dependance of NMDAR channel
b=1/3.57  # control the voltage dependance of NMDAR channel ([Mg2+]=1mM )




########################################################################################################################
### External Input. Need to modify
fext     = 2400.0*Hz
ts_start = 1.0*second # stimulus onset
ts_stop  = 3.0*second # stimulus offset
mu0       = 1.*30.0*Hz   # stimulus strength
mu_slope  = 1.*50.0*Hz   # stimulus strength
#dt_stim       = 100.*ms               # Time step of updating stimulus (should be the highest common denominator between tc_start and tr_stop)


coherence = 0.  #Temp

dt_psth = 0.001*second
