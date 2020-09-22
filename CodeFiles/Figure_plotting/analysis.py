import numpy as np


def load_raster(filename):
    raster = np.genfromtxt(filename,delimiter='',invalid_raise=False)
    return raster

def load_raster_spikes(filename):
    raster = np.genfromtxt(filename,delimiter=',',invalid_raise=False)
    return raster

def spike_count_onebin(neuron_ind, spike_times,n_neurons,t_start,t_end):
    '''
    Retuns a vector of length n_neurons with the spike count (not rate) in a given time_bin
    '''
    sc = np.zeros(n_neurons)
    relevant_indices = np.nonzero((spike_times >= t_start) & (spike_times < t_end))[0]
    relevant_neuron_ind = neuron_ind[relevant_indices]
    n_spikes = len(relevant_indices)
    for k_spike in xrange(n_spikes):
        ind1 = relevant_neuron_ind[k_spike]
        if ind1> n_neurons-1:
            continue        # Ignore (hopefully) indiivudual cases where the code messes up and we get 1nd1 ~ 1027 etc...
        sc[ind1] += 1
    return sc

def burst_count_onebin(neuron_ind, spike_times,n_neurons,t_window, input_rate):
    '''
    Retuns a vector of length n_neurons with the burst count (not rate) in a given time_bin
    '''
    t_start, t_end = t_window
    bc = np.zeros(n_neurons)
    sc_1st = np.zeros(n_neurons)            #Count of spikes that are not in a burst (1st spike of either single spikes, or of bursts)
    relevant_indices = np.nonzero((spike_times >= t_start) & (spike_times < t_end))[0]
    relevant_neuron_ind = neuron_ind[relevant_indices]
    #Note: esp because of our definition here, it is important to omit the data of the initial portion
    spike_intervals          = np.diff(spike_times)
    spike_intervals_diff     = np.diff(spike_intervals)
    spike_intervals_diff     = np.insert(spike_intervals_diff, 0, -10)
    #Add a term to the beginning of spike_intervals_diff to ensure that the 2nd spike, if it's a part of burst, will be counted in bc. Note that we do not need to do this for spike_intervals.
    relevant_spike_intervals      = spike_intervals[relevant_indices -1]
    relevant_spike_intervals_diff = spike_intervals_diff[relevant_indices -1]
    n_spikes = len(relevant_indices)
    # interval_threshold      = 1/input_rate/2        # [s] Threshold interval such that only consecutive spikes within this duration will be counted as bursts.
    # interval_diff_threshold = 1/input_rate/2        # [s] Minimum diff in spike interval such that any interval_diff less means this spike belongs in the same burst as the previous one.
    #     #Note that both thresholds are chosen based on the 5Hz input rate data set. Would probably do worse for 10Hz, 20Hz etc...
    interval_threshold      = 0.008        # [s] Threshold interval such that only consecutive spikes within this duration will be counted as bursts.
    interval_diff_threshold = 0.003        # [s] Minimum diff in spike interval such that any interval_diff less means this spike belongs in the same burst as the previous one.
        #Note that both thresholds are chosen based on the 5Hz input rate data set. Would probably do worse for 10Hz, 20Hz etc...
    for k_spike in xrange(n_spikes):
        ind1 = relevant_neuron_ind[k_spike]
        if (relevant_spike_intervals[k_spike]<interval_threshold and abs(relevant_spike_intervals_diff[k_spike])>interval_diff_threshold):
            bc[ind1] += 1
        if (relevant_spike_intervals[k_spike]>interval_threshold ):
            sc_1st[ind1] += 1
    return bc/sc_1st

def spike_in_burst_count_onebin(neuron_ind, spike_times,n_neurons,t_window):
    '''
    Retuns a vector of length n_neurons with the burst count (not rate) in a given time_bin
    '''
    t_start, t_end = t_window
    sbc_max = 100            #Possible max spikes per burst to record
#    sbc = np.zeros((n_neurons, sbc_max-1))          #Count for spikes in burst
    sbc = np.zeros((sbc_max-1))          #Count for spikes in burst
    sbc_values = 1+np.arange(sbc_max-1)                      #Corresponding values for each slot in sbc. Note that the first one (=1) will not be used in sbc calculation.
    relevant_indices = np.nonzero((spike_times >= t_start) & (spike_times < t_end))[0]
    relevant_neuron_ind = neuron_ind[relevant_indices]
    #Note: esp because of our definition here, it is important to omit the data of the initial portion
    spike_intervals          = np.diff(spike_times)
    relevant_spike_intervals      = spike_intervals[relevant_indices -1]
    n_spikes = len(relevant_indices)
    interval_threshold      = 0.008        # [s] Threshold interval such that only consecutive spikes within this duration will be counted as bursts.
#    interval_diff_threshold = 0.06        # [s] Minimum diff in spike interval such that any interval_diff less means this spike belongs in the same burst as the previous one.
        #Note that both thresholds are chosen based on the 5Hz input rate data set. Would probably do worse for 10Hz, 20Hz etc...
    n_spike_burst_count_temp=1      #Counter for number of spikes in burst.
    for k_spike in xrange(n_spikes):
        ind1 = relevant_neuron_ind[k_spike]
        if (relevant_spike_intervals[k_spike]<interval_threshold):      #Assumed only 1 neuron/index.
            n_spike_burst_count_temp += 1
            n_spike_burst_count_temp = min(n_spike_burst_count_temp, sbc_max-1)           #Just to prevent it from going to infinity, when the alg is not working properly somehow. sbc_max should be large enough that when this happens we know sth's wrong.
        else:
            n_spike_burst_count_temp  = 1
        sbc[n_spike_burst_count_temp-1] +=1
    sbc = -np.diff(sbc)                          #Do diff to prevent overcounting/ only count spikes that correspond to max values.
    last_digits_to_skip = 3
    burst_total = sum([sbc[i] for i in range(1, len(sbc)-1-last_digits_to_skip)])           #Total number of bursts
    if burst_total==0:
        mean_spikes_in_burst=0
    else:
#        mean_spikes_in_burst = sum([sbc_values[i]*sbc[i] for i in range(1, len(sbc)-1)]) /burst_total
#        mean_spikes_in_burst = (sum(sbc_values*sbc)-sbc_values[0]*sbc[0]) /burst_total
        mean_spikes_in_burst = sum(sbc_values[1:(sbc_max-1-last_digits_to_skip)]*sbc[1:(sbc_max-1-last_digits_to_skip)]) /burst_total
        # print(sum(sbc_values[1:(sbc_max-2)]*sbc[1:(sbc_max-2)]))
        # print(burst_total)
    return mean_spikes_in_burst


def sliding_win_on_circ_data(data_mat,window_width,axis=0):
    smaller_half = np.floor((window_width)/2)
    bigger_half = np.ceil((window_width)/2)
    sum_up_mat = np.zeros(np.shape(data_mat))
    for k_circ in xrange(-int(smaller_half),int(bigger_half+np.mod(window_width,2))):
        sum_up_mat += np.roll(data_mat,-k_circ,axis)
    return sum_up_mat/window_width

def sliding_win_on_lin_data(data_mat,window_width,axis=0):
    smaller_half = np.floor((window_width)/2)
    bigger_half = np.ceil((window_width)/2)
    data_mat_result = np.zeros(len(data_mat))
    for k_lin in range(len(data_mat)):
        lower_bound = np.maximum(k_lin-smaller_half, 0)
        upper_bound = np.minimum(k_lin+bigger_half, len(data_mat))
        data_mat_result[k_lin] = np.mean(data_mat[lower_bound:upper_bound])
    return data_mat_result


#-----------------------------------------------------------------------------------------------------------------------
### Filter and PSTH
#def filter_gauss(t,tau=0.1):
def filter_gauss(t,tau=0.0265):
    z = 0. + 1./(np.sqrt(2*np.pi)*tau)*np.exp(-t**2./(2.*tau**2.))
    return z

def filter_exp(t,tau=0.02):

    z = np.zeros(np.shape(t))

    ind1 = np.nonzero(t<0.)
    z[ind1] = 0.
    ind2 = np.nonzero(t>=0.)
    z[ind2] = 0. + 1./tau*np.exp(-t[ind2]/tau)#y[ind2]/(1.-np.exp(-d*y[ind2]))

    # print np.sum(z[ind2])
    return z

def filter_box(t,tau=0.1,t_center=0):
    z = 0. + 1./tau*(np.abs(t-t_center) <= tau/2.)#(-tau/2. <= (t-t_center))*((t-t_center) < tau/2.)
    return z

def filter_alpha(t,tau=0.1,t_center=0):
    z = 0. + (t-t_center)/tau*np.exp(-t/tau)*(t>=0.)
    return z




def psth(spikes_times, t_vec=None, t_start = 0, t_end = None, dt_psth = 0.001, filter_fn = filter_gauss, params_filter={},duty_cycle=1., seed=None):

    if t_end == None:
        t_end = np.ceil(np.max([np.max(np.hstack((spikes_i,[t_start]))) for spikes_i in spikes]))

    if t_vec is None:
        tpts = int(np.round((t_end-t_start)/dt_psth))+1
        t_vec = np.linspace(t_start, t_end, tpts)
    else:
        tpts = len(t_vec)
        #print 'tpts', tpts

    filt_t_vec = np.arange(-(t_end-t_start),(t_end-t_start),dt_psth)

    filt_vec = filter_fn(filt_t_vec,**params_filter)

    filt_vec /= np.sum(filt_vec)

    fr = np.zeros(len(t_vec))

    if seed is None:
        seed=np.random.randint(duty_cycle)

    relevant_indices = np.nonzero((spikes_times >= t_start) & (spikes_times <= t_end) &
                                    (np.mod(spikes_times+seed,duty_cycle)<=1.0001))[0]

    relevant_spikes_times = spikes_times[relevant_indices]
    for k_spike in xrange(len(relevant_spikes_times)):
        t_spike = relevant_spikes_times[k_spike]
        spike_pt = np.argmin(np.abs(t_spike - t_vec))
        fr += np.roll(filt_vec,-tpts+spike_pt)[:tpts]                                                                   # Adding part of the filter corresponding to the impact after the spike
        fr += np.roll(filt_vec,-tpts+spike_pt)[-tpts:][::-1]                                                            # Adding part of the filter corresponding to the impact before the spike. 0 for casual filters (exponential, alpha etc)
    return fr/dt_psth                                                                                                   # Note that this fr hsa to be normalized by the number of neurons in the corresponding group (N1, N2)


    # n_trials = len(spikes)
    # for i in xrange(n_trials):
    #     spikes_i = np.array(spikes[i])
    #     relevant_indices = np.nonzero((spikes_i >= t_start) & (spikes_i <= t_end) &
    #                                     (np.mod(spikes_i+seed,duty_cycle)<=1.0001))[0]
    #
    #     relevant_spikes = spikes_i[relevant_indices]
    #     for k_spike in xrange(len(relevant_spikes)):
    #         t_spike = relevant_spikes[k_spike]
    #         spike_pt = np.argmin(np.abs(t_spike - t_vec))
    #         fr += np.roll(filt_vec,-tpts+spike_pt)[:tpts]                                                               # Adding part of the filter corresponding to the impact after the spike
    #         fr += np.roll(filt_vec,-tpts+spike_pt)[-tpts:][::-1]                                                        # Adding part of the filter corresponding to the impact before the spike. 0 for casual filters (exponential, alpha etc)
    # fr /= n_trials
    # return fr
    #
    # out = {'t':t_vec, 'r': fr}
    # return out

