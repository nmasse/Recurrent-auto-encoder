import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

print("--> Loading parameters...")

global par, analysis_par

"""
Independent parameters
"""

rnd_save_suffix = np.random.randint(10000)

par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'debug_model'           : False,
    'load_previous_model'   : False,
    'analyze_model'         : False,

    # Network configuration
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'var_delay'             : True,
    'catch_trials'          : False,     # Note that turning on var_delay implies catch_trials

    # Network shape
    'num_motion_tuned'      : 36,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 200,
    'n_output'              : 3,
    'n_latent'              : 5,
    'topographic_inputs'    : True,

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 1e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1,         # Usually 1

    # Variance values
    'clip_max_grad_val'     : 0.25,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.01,
    'noise_rnn_sd'          : 0.2,

    # Tuning function data
    'num_motion_dirs'       : 12,
    'tuning_height'         : 1,        # magnitutde scaling factor for von Mises
    'kappa'                 : 1,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 0.00000001,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_train_size'      : 128,
    'num_batches'           : 1,
    'num_iterations'        : 200000,
    'iters_between_outputs' : 500,

    # Task specs
    'trial_type'            : 'DMS', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 400,
    'fix_time'              : 500,
    'sample_time'           : 500,
    'delay_time'            : 500,
    'test_time'             : 500,
    'rule_onset_time'       : 1900,
    'rule_offset_time'      : 2100,
    'variable_delay_max'    : 500,
    'mask_duration'         : 50,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task

    # Save paths
    'save_fn'               : 'model_results.pkl',
    'ckpt_save_fn'          : 'model' + str(rnd_save_suffix) + '.ckpt',
    'ckpt_load_fn'          : 'model' + str(rnd_save_suffix) + '.ckpt',

    # Analysis
    'svm_normalize'         : True
}

"""
Parameters to be used before running analysis
"""
analysis_par = {
    'analyze_model'         : True,
    'load_previous_model'   : True,
    'num_iterations'        : 1,
    'num_batches'           : 1,
    'batch_train_size'      : 2048,
    'var_delay'             : False,
    'learning_rate'         : 0,
    'catch_trial_pct'       : 0,
}

"""
Parameters to be used after running analysis
"""
revert_analysis_par = {
    'analyze_model'         : False,
    'load_previous_model'   : False,
    'num_iterations'        : 1000,
    'num_batches'           : 8,
    'batch_train_size'      : 128,
    'var_delay'             : True,
    'learning_rate'         : 5e-3,
    'catch_trial_pct'       : 0,
    'delay_time'            : 1000
}


"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for key, val in updates.items():
        par[key] = val
        print(key, val)

    update_trial_params()
    update_dependencies()

def update_trial_params():

    """
    Update all the trial parameters given trial_type
    """

    if par['trial_type'] == 'DMS':
        par['num_rules'] = 1
        par['num_rule_tuned'] = 0

    if par['trial_type'] == 'DMRS45':
        par['rotation_match'] = 45

    elif par['trial_type'] == 'DMRS90':
        par['rotation_match'] = 90

    elif  par['trial_type'] == 'DMRS180':
        par['rotation_match'] = 180

    elif par['trial_type'] == 'dualDMS':
        par['catch_trial_pct'] = 0
        par['num_receptive_fields'] = 2
        par['num_rules'] = 2
        par['probe_trial_pct'] = 0
        par['probe_time'] = 10
        par['num_rule_tuned'] = 12
        par['spike_cost'] = 0.005
        #par['num_iterations'] = 1500


    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        par['catch_trial_pct'] = 0
        par['match_test_prob'] = 0.5
        par['max_num_tests'] = 3
        par['delay_time'] = 3000
        par['ABBA_delay'] = int(par['delay_time']/par['max_num_tests']/2)
        par['repeat_pct'] = 0
        if par['trial_type'] == 'ABBA':
            par['repeat_pct'] = 0.5

    elif par['trial_type'] == 'DMS+DMRS' or par['trial_type'] == 'DMS+DMRS_early_cue':

        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        if par['trial_type'] == 'DMS+DMRS':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 500
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 700
        else:
            par['rotation_match'] = [0, 45]
            par['rule_onset_time'] = par['dead_time']
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time']

    elif par['trial_type'] == 'DMS' or par['trial_type'] == 'DMC':
        pass

    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

    # Possible rules based on rule type values
    #par['possible_rules'] = [par['num_receptive_fields'], par['num_categorizations']]

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = par['dt']/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']


    # General event profile info
    #par['name_of_stimulus'], par['date_stimulus_created'], par['author_of_stimulus_profile'] = get_profile(par['profile_path'])
    # List of events that occur for the network
    #par['events'] = get_events(par['profile_path'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    if par['trial_type'] == 'dualDMS':
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+2*par['test_time']
    else:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_train_size']), dtype=np.float32)

    par['input_to_hidden_dims'] = [par['n_hidden'], par['n_input']]
    par['hidden_to_hidden_dims'] = [par['n_hidden'], par['n_hidden']]


    # Initialize input weights
    if par['topographic_inputs']:
        par['w_in0'] = organize_inputs()
    else:
        par['w_in0'] = initialize(par['input_to_hidden_dims'], par['connection_prob'])

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        par['w_rnn0'] = initialize(par['hidden_to_hidden_dims'], par['connection_prob'])

        for i in range(par['n_hidden']):
            par['w_rnn0'][i,i] = 0
        par['w_rnn_mask'] = np.ones((par['hidden_to_hidden_dims']), dtype=np.float32) - np.eye(par['n_hidden'])
    else:
        par['w_rnn0'] = np.eye(par['n_hidden'], dtype=np.float32)
        par['w_rnn_mask'] = np.ones((par['hidden_to_hidden_dims']), dtype=np.float32)

    par['b_rnn0'] = np.zeros((par['n_hidden'], 1), dtype=np.float32)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    par['w_rnn0'] = par['w_rnn0']/(spectral_radius(par['w_rnn0']))

    # Initialize output weights and biases
    par['w_out0'] =initialize([par['n_output'], par['n_hidden']], par['connection_prob'])
    par['b_out0'] = np.zeros((par['n_output'], 1), dtype=np.float32)
    par['w_out_mask'] = np.ones((par['n_output'], par['n_hidden']), dtype=np.float32)

    if par['EI']:
        par['ind_inh'] = np.where(par['EI_list'] == -1)[0]
        par['w_out0'][:, par['ind_inh']] = 0
        par['w_out_mask'][:, par['ind_inh']] = 0

    # encoder model
    # following VARIATIONAL RECURRENT AUTO-ENCODERS
    # Fabius et al., 2015
    Z = 0.01


    par['w_mu0'] = Z*initialize([par['n_latent'], par['n_hidden']], par['connection_prob'])
    par['w_sigma0'] = Z*initialize([par['n_latent'], par['n_hidden']], par['connection_prob'])
    par['b_mu0'] = np.zeros((par['n_latent'], 1), dtype=np.float32)
    par['b_sigma0'] = np.zeros((par['n_latent'], 1), dtype=np.float32)

    # decoder model
    """
    par['w_z0'] = Z*initialize([par['n_hidden'], par['n_latent']], par['connection_prob'])
    par['w_x0'] = Z*initialize([par['n_hidden'], par['n_input']], par['connection_prob'])
    par['w_dec0'] = Z*initialize([par['n_hidden'], par['n_hidden']], par['connection_prob'])
    par['w_out0'] = Z*initialize([par['n_input'], par['n_hidden']], par['connection_prob'])
    par['b_z0'] = np.zeros((par['n_hidden'], 1), dtype=np.float32)
    par['b_dec0'] = np.zeros((par['n_hidden'], 1), dtype=np.float32)
    """

def organize_inputs():

    par['w_in0'] = np.zeros((par['n_hidden'], par['n_input']), dtype = np.float32)

    ring_exc = np.exp(1j*2*np.pi*np.arange(par['num_exc_units'])/par['num_exc_units'])
    ring_inh = np.exp(1j*2*np.pi*np.arange(par['num_inh_units'])/par['num_inh_units'])
    ring_input = np.exp(1j*2*np.pi*np.arange(par['n_input'])/par['n_input'])

    for i in range(par['n_input']):
        for j in range(par['num_exc_units']):
            if np.random.rand() < ring_exc[j]*np.conj(ring_input[i]):
                par['w_in0'][j,i] = np.random.gamma(shape=0.25, scale=1.0)
        for j in range(par['num_inh_units']):
            if np.random.rand() < ring_inh[j]*np.conj(ring_input[i]):
                par['w_in0'][par['num_exc_units']+j,i] = np.random.gamma(shape=0.25, scale=1.0)

    #plt.imshow(par['w_in0'], aspect = 'auto', interpolation = 'none')
    #plt.show()


    return par['w_in0']



def initialize(dims, connection_prob):
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


def spectral_radius(A):

    return np.max(abs(np.linalg.eigvals(A)))

update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")
