"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import time
import pickle
import os
import matplotlib.pyplot as plt
from parameters import *

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Using EI Network:\t', par['EI'])

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=0)
        self.target_data = tf.unstack(target_data, axis=0)
        self.time_mask = tf.unstack(mask, axis=0)

        # Load the initial hidden state activity to be used at the start of each trial
        self.hidden_init = tf.constant(par['h_init'])

        self.declare_variables()

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def declare_variables(self):
        """ Initialize all required variables """

        # All the possible prefixes based on network setup
        lstm_var_prefixes   = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', 'bf', 'bi', 'bo', 'bc']
        bio_var_prefixes    = ['W_in', 'b_rnn', 'W_rnn']
        rl_var_prefixes     = ['W_pol_out', 'b_pol_out', 'W_val_out', 'b_val_out']
        base_var_prefies    = ['W_out', 'b_out']
        latent_var_prefixes = ['W_mu', 'W_sigma', 'W_r_out', 'b_mu', 'b_sigma', 'b_r_out']


        # Add relevant prefixes to variable declaration
        prefix_list = base_var_prefies
        prefix_list += latent_var_prefixes
        if par['architecture'] == 'LSTM':
            prefix_list += lstm_var_prefixes
        elif par['architecture'] == 'BIO':
            prefix_list += bio_var_prefixes

        if par['training_method'] == 'RL':
            prefix_list += rl_var_prefixes
        elif par['training_method'] == 'SL':
            pass

        # Use prefix list to declare required variables and place them in a dict
        self.var_dict = {}
        with tf.variable_scope('network'):
            for p in prefix_list:
                self.var_dict[p] = tf.get_variable(p, initializer=par[p + '_init'])
                if p in lstm_var_prefixes:
                    # create another copy for the recurrent decoder
                    if 'W' in p:
                        # W matrices have different sizes
                        self.var_dict[p + '_r'] = tf.get_variable(p + '_r', initializer=par[p + '_r_init'])
                    else:
                        self.var_dict[p + '_r'] = tf.get_variable(p + '_r', initializer=par[p + '_init'])


        if par['architecture'] == 'BIO':
            # Modify recurrent weights if using EI neurons (in a BIO architecture)
            self.W_rnn_eff = (tf.constant(par['EI_matrix']) @ tf.nn.relu(self.var_dict['W_rnn'])) \
                if par['EI'] else self.var_dict['W_rnn']


    def run_model(self):

        """
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """
        h = tf.zeros_like(par['h_init'])
        c = tf.zeros_like(par['h_init'])
        syn_x = tf.constant(par['syn_x_init'])
        syn_u = tf.constant(par['syn_u_init'])
        self.h = []
        self.output = []
        self.x_hat = []

        # Loop through the neural inputs, indexed in time
        for rnn_input, target, time_mask in zip(self.input_data, self.target_data, self.time_mask):

            # Compute the state of the hidden layer
            h, c, syn_x, syn_u = self.recurrent_cell(h, c, syn_x, syn_u, rnn_input)

            # Record hidden state
            self.h.append(h)
            # Compute outputs for loss
            self.output.append(h @ self.var_dict['W_out'] + self.var_dict['b_out'])

        self.latent_loss = 0.
        self.recotruction_loss = 0.
        input_data = tf.stack(self.input_data, axis=0)
        for t in [39,44,49,54,59,64,69,74,79,84,89,94,99]:
            self.x_hat = []

            self.latent_mu = tf.matmul(self.h[t], self.var_dict['W_mu']) + self.var_dict['b_mu']
            self.latent_sigma = tf.matmul(self.h[t], self.var_dict['W_sigma']) + self.var_dict['b_sigma']
            self.latent_loss += -0.5*tf.reduce_sum(1 + self.latent_sigma - tf.square(self.latent_mu) - tf.exp(self.latent_sigma))

            self.sample_latent = self.latent_mu + tf.exp(self.latent_sigma)* \
                tf.random_normal([par['batch_size'], par['n_latent']], 0, 1 , dtype=tf.float32)

            h = tf.zeros_like(par['h_init'])
            c = tf.zeros_like(par['h_init'])
            for _ in range(t+1):
                h, c, syn_x, syn_u = self.recurrent_cell(h, c, syn_x, syn_u, self.sample_latent, reverse = True)
                self.x_hat.append(h @ self.var_dict['W_r_out'] + self.var_dict['b_r_out'])


            x_hat = tf.stack(self.x_hat, axis=0)
            self.recotruction_loss += tf.reduce_sum(tf.square(x_hat[-1::-1,:,:] - input_data[:t+1,:,:]))/1000000


    def recurrent_cell(self, h, c, syn_x, syn_u, rnn_input, reverse = False):
        """ Using the appropriate recurrent cell
            architecture, compute the hidden state """

        if par['architecture'] == 'BIO':

            # Apply synaptic short-term facilitation and depression, if required
            if par['synapse_config'] == 'std_stf':
                syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
                syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
                syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                h_post = syn_u*syn_x*h
            else:
                h_post = h

            # Compute hidden state
            h = self.gating*tf.nn.relu((1-par['alpha_neuron'])*h \
              + par['alpha_neuron']*(rnn_input @ self.var_dict['W_in'] + h_post @ self.W_rnn_eff + self.var_dict['b_rnn']) \
              + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))
            c = tf.constant(-1.)

        elif par['architecture'] == 'LSTM':

            suffix = '_r' if reverse else ''

            # Compute LSTM state
            # f : forgetting gate, i : input gate,
            # c : cell state, o : output gate
            f   = tf.sigmoid(rnn_input @ self.var_dict['Wf'+suffix] + h @ self.var_dict['Uf'+suffix] + self.var_dict['bf'+suffix])
            i   = tf.sigmoid(rnn_input @ self.var_dict['Wi'+suffix] + h @ self.var_dict['Ui'+suffix] + self.var_dict['bi'+suffix])
            cn  = tf.tanh(rnn_input @ self.var_dict['Wc'+suffix] + h @ self.var_dict['Uc'+suffix] + self.var_dict['bc'+suffix])
            c   = f * c + i * cn
            o   = tf.sigmoid(rnn_input @ self.var_dict['Wo'+suffix] + h @ self.var_dict['Uo'+suffix] + self.var_dict['bo'+suffix])

            # Compute hidden state
            h = o * tf.tanh(c)
            syn_x = tf.constant(-1.)
            syn_u = tf.constant(-1.)

        return h, c, syn_x, syn_u



    def optimize(self):

        """
        Calculate the loss functions and optimize the weights

        perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]
        """
        """
        cross_entropy
        """

        self.perf_loss = tf.reduce_mean([mask*tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_hat, labels = desired_output, dim=1) \
                for (y_hat, desired_output, mask) in zip(self.output, self.target_data, self.time_mask)])




        self.loss = self.perf_loss + self.recotruction_loss + 0*self.latent_loss


        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)

        self.train_op = opt.apply_gradients(grads_and_vars)



def main(gpu_id = None):


    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


    tf.reset_default_graph()
    stim = stimulus.MultiStimulus()

    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size']])
    x = tf.placeholder(tf.float32, shape=[par['num_time_steps'] , par['batch_size'], par['n_input'],])  # input data
    y = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_output']]) # target data

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session() as sess:

        #with tf.device("/gpu:0"):
        model = Model(x, y, mask)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], \
            'recotruction_loss': [], 'trial': [], 'time': []}

        task = 17

        for i in range(par['n_train_batches']):

            # generate batch of N (batch_size X num_batches) trials
            name, stim_in, target_data, train_mask, _ = stim.generate_trial(task)

            _, loss, perf_loss, recotruction_loss, x_hat, output, h, latent = \
                sess.run([model.train_op, model.loss, model.perf_loss, model.recotruction_loss, \
                model.x_hat, model.output, model.h, model.sample_latent], \
                {x: stim_in, y: target_data, mask: train_mask})


            accuracy = get_perf(target_data, output, train_mask)

            iteration_time = time.time() - t_start

            #model_performance = append_model_performance(model_performance, accuracy, loss, perf_loss, \
            #    recotruction_loss, (i+1)*N, iteration_time)

            """
            Save the network model and output model performance to screen
            """
            if i%100==0:
                print_results(i, iteration_time, perf_loss, recotruction_loss, h, accuracy)
                weights = sess.run([model.var_dict])
                pickle.dump(weights[0], open('./savedir/saved_weights.pkl','wb'))
                print('Weights saved')

                if i%1000==0:
                    x_hat = np.stack(x_hat, axis = 0)
                    f = plt.figure(figsize = (8,4))
                    for k in range(2):
                        ax = f.add_subplot(2, 2, 1+k*2)
                        ax.imshow(stim_in[:,k,:], aspect = 'auto')
                        ax = f.add_subplot(2, 2, 2+k*2)
                        ax.imshow(x_hat[-1::-1,k,:], aspect = 'auto')
                    plt.show()


        """
        Analyze the network model and save the results
        """
        if par['analyze_model']:
            weights = eval_weights()
            analysis.analyze_model(trial_info, y_hat, x_hat, latent, state_hist, model_performance, weights)


def append_model_performance(model_performance, accuracy, loss, perf_loss, recotruction_loss, spike_loss, trial_num, iteration_time):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['recotruction_loss'].append(recotruction_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['trial'].append(trial_num)
    model_performance['time'].append(iteration_time)

    return model_performance

def eval_weights():

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    with tf.variable_scope('latent', reuse=True):
        W_mu = tf.get_variable('W_mu')
        W_sigma = tf.get_variable('W_sigma')
        b_mu = tf.get_variable('b_mu')
        b_sigma = tf.get_variable('b_sigma')

    weights = {
        'w_in'      : W_in.eval(),
        'w_rnn'     : W_rnn.eval(),
        'w_out'     : W_out.eval(),
        'w_mu'      : W_mu.eval(),
        'w_sigma'   : W_sigma.eval(),
        'b_mu'      : b_mu.eval(),
        'b_sigma'   : b_sigma.eval(),
        'b_rnn'     : b_rnn.eval(),
        'b_out'     : b_out.eval()
    }

    return weights

def print_results(iter_num, iteration_time, perf_loss, recotruction_loss, state_hist, accuracy):

    print('Iteration {:5d}'.format(iter_num) + ' | Time {:0.2f} s'.format(iteration_time) +
      ' | Perf loss {:0.4f}'.format(perf_loss)  +
      ' | Recon. loss {:0.4f}'.format(recotruction_loss) + ' | Mean activity {:0.4f}'.format(np.mean(state_hist)) +
      ' | Accuracy {:0.4f}'.format(accuracy))


def get_perf(target, output, mask):

    """ Calculate task accuracy by comparing the actual network output
    to the desired output only examine time points when test stimulus is
    on in another words, when target[:,:,-1] is not 0 """

    output = np.stack(output, axis=0)
    mk = mask*np.reshape(target[:,:,-1] == 0, (par['num_time_steps'], par['batch_size']))

    target = np.argmax(target, axis = 2)
    output = np.argmax(output, axis = 2)

    return np.sum(np.float32(target == output)*np.squeeze(mk))/np.sum(mk)
