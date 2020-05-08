# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
global treshold_sinr
treshold_sinr = 10.0**(30.0/10.0)


def get_random_rayleigh_variable(rayleigh_var, N, K=None):
    if K is None:
        return np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn(N, N) +
                                                1j * rayleigh_var * np.random.randn(N, N))
    else:
        return np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn(K, N) +
                                                1j * rayleigh_var * np.random.randn(K, N))
def get_markov_rayleigh_variable(state,correlation,rayleigh_var, N, K=None):
    if K is None:
        return correlation*state +np.sqrt(1-np.square(correlation)) * np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn(N, N) +
                                                1j * rayleigh_var * np.random.randn(N, N))
    else:
        return correlation*state +np.sqrt(1-np.square(correlation)) * np.sqrt(2.0/np.pi) * (rayleigh_var * np.random.randn(K, N) +
                                                1j * rayleigh_var * np.random.randn(K, N))

def get_abs_initial_normal_channel(N):
    return np.random.randn(N, N)
def get_abs_brownian_normal_channel(state,correlation,N):
    return state + np.sqrt(1-np.square(correlation))*np.random.randn(N, N)
# FP algorithm that is in Naiming Shen's paper
def FP_algorithm_weighted(N, H, Pmax, noise_var,weights):
    start_FP_time = time.time()
    f_new = 0
    gamma = np.zeros(N)
    y = np.zeros(N)
    p_init = Pmax * np.ones(N)
    # Initial power is just all transmitters transmit with full power
    p = np.array(p_init)
    # Take pow 2 of abs_H, no need to take it again and again
    H_2 = H ** 2

    for i in range(N):
        tmp_1 = H_2[i, i] * p[i]
        tmp_2 = np.matmul(H_2[i, :], p) + noise_var
        # Initialize gamma
        gamma[i] = tmp_1 / (tmp_2 - tmp_1)
    for iter in range(100):
        f_old = f_new
        for i in range(N):
            tmp_1 = H_2[i, i] * p[i]
            tmp_2 = np.matmul(H_2[i, :], p) + noise_var
            # Update y
            y[i] = np.sqrt(weights[i] * (1 + gamma[i]) * tmp_1) / (tmp_2)
            # Update gamma
            gamma[i] = tmp_1 / (tmp_2 - tmp_1)


        f_new = 0
        for i in range(N):
            # Update p
            p[i] = min (Pmax, (y[i] ** 2) * weights[i] * (1 + gamma[i]) * H_2[i,i] / np.square(np.matmul(np.square(y), H_2[:,i])))
        for i in range(N):
            # Get new result
            f_new = f_new + 2 * y[i] * np.sqrt(weights[i] * (1+gamma[i]) * H_2[i,i] * p[i]) - (y[i] ** 2) * (np.matmul(H_2[i, :], p)
                                                                                                            + noise_var)
        #Look for convergence
        if f_new - f_old <= 0.001:
            break
#     # DEBUG
    end_time = time.time() - start_FP_time
    end_statistics = [end_time, iter]
    # Return optimum result after convergence
    return p, end_statistics

# Functions for WMMSE algorithm
def WMMSE_algorithm_weighted(N, H, Pmax, var_noise, weights):
    start_WMMSE_time = time.time()
    vnew = 0
    # random initialization gives much lower performance.
    b = np.sqrt(Pmax) * np.ones(N)#* np.random.rand(N) 
    f = np.zeros(N)
    w = np.zeros(N)
    for i in range(N):
        f[i] = H[i, i] * b[i] / (np.matmul(np.square(H[i, :]), np.square(b)) + var_noise)
        w[i] = 1.0 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + np.log2(w[i])

#    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(N):
            btmp = weights[i]* w[i] * f[i] * H[i, i] / sum(weights * w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(N):
            f[i] = H[i, i] * b[i] / (np.matmul(np.square(H[i, :]) ,np.square(b)) + var_noise)
            w[i] = 1.0 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + np.log2(w[i])

        if vnew - vold <= 0.01:
            break
    p_opt = np.square(b)
    end_time = time.time() - start_WMMSE_time
    end_statistics = [end_time, iter]
    return p_opt, end_statistics


def sumrate_weighted_clipped(H,p,N,noise_var,weights):
    #H = H.reshape(N,N)
    # Take pow 2 of abs_H, no need to take it again and again
    H_2 = (H**2)
    sum_rate = 0
    for loop in range (N):
        tmp_1 = H_2[loop, loop] * p[loop]
        tmp_2 = np.matmul(H_2[loop, :], p) + noise_var - tmp_1
        if(tmp_1/tmp_2 > treshold_sinr):
            sum_rate += weights[loop]*np.log2(1.0+treshold_sinr)
        else:
            sum_rate += weights[loop]*np.log2(1.0+tmp_1/tmp_2)
    return sum_rate


def sumrate_list_clipped(H,p,N,noise_var):
    # Take pow 2 of abs_H, no need to take it again and again
    H_2 = H ** 2
    #H = H.reshape(N,N)
    sum_rate = []
    for loop in range (N):
        tmp_1 = H_2[loop, loop] * p[loop]
        tmp_2 = np.matmul(H_2[loop, :], p) + noise_var - tmp_1
        if(tmp_1/tmp_2 > treshold_sinr):
            sum_rate.append(np.log2(1.0+treshold_sinr))
        else:
            sum_rate.append(np.log2(1.0+tmp_1/tmp_2))
#        sum_rate.append(10*np.log10(tmp_1/tmp_2))
    return sum_rate
def sinr_list(H,p,N,noise_var):
    # Take pow 2 of abs_H, no need to take it again and again
    H_2 = H ** 2
    #H = H.reshape(N,N)
    sum_rate = []
    for loop in range (N):
        tmp_1 = H_2[loop, loop] * p[loop]
        tmp_2 = np.matmul(H_2[loop, :], p) + noise_var - tmp_1
#        sum_rate.append(np.log2(1+tmp_1/tmp_2))
        sum_rate.append(10*np.log10(tmp_1/tmp_2))
    return sum_rate


def reward_helper(H,p,N,noise_var,Pmax,neighbors_in):
    # Take pow 2 of abs_H, no need to take it again and again
    H_2 = H ** 2
    #H = H.reshape(N,N)
    # Diagonals are self sum_rate
    # Off diagonal is the rate without the competitor transmitted.
    sum_rate = np.zeros((N,N))
    for loop in range (N):
        tmp_1 = H_2[loop, loop] * p[loop]
        tmp_2 = np.matmul(H_2[loop, :], p) + noise_var - tmp_1
        sum_rate[loop,loop]=(np.log2(1+np.minimum(treshold_sinr,tmp_1/tmp_2)))
        if(len(neighbors_in[loop])>0):
            sum_rate[loop,neighbors_in[loop]] = -((np.log2(1+np.minimum(treshold_sinr,tmp_1/([tmp_2]*(len(neighbors_in[loop]))-H_2[loop, neighbors_in[loop]]* p[neighbors_in[loop]]))))-(np.log2(1+np.minimum(treshold_sinr,tmp_1/tmp_2))))
    return sum_rate

def neural_net(x, weights, biases): # Create model
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])   # x = w*x+b
    layer_1 = tf.nn.relu(layer_1)                                 # x = max(0, x)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Kumber of neurons at the layer3 is basically number of possible actions.
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
#    out_layer = tf.nn.relu(out_layer)
#    out_layer = tf.nn.tanh(out_layer)
    return out_layer

def actor_net(x, weights, biases): # Create model
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])   # x = w*x+b
    layer_1 = tf.nn.relu(layer_1)                                 # x = max(0, x)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Kumber of neurons at the layer3 is basically number of possible actions.
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    out_layer = tf.nn.sigmoid(out_layer)
    # out_layer = tf.nn.relu6(out_layer) / 6.
    return out_layer

def critic_net(x_s,x_a, weights, biases): # Create model
    x = tf.concat([x_s,x_a],axis=1)
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])   # x = w*x+b
    layer_1 = tf.nn.relu(layer_1)                                 # x = max(0, x)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    # out_layer = tf.nn.tanh(out_layer)
    return out_layer

def initial_weights (num_input, n_hidden_1, n_hidden_2, n_hidden_3, num_output):
    weights = {
        'h1': tf.Variable(tf.random_uniform([num_input, n_hidden_1], -1./np.sqrt(n_hidden_1),1./np.sqrt(n_hidden_1))),
        'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], -1./ np.sqrt(n_hidden_2), 1./ np.sqrt(n_hidden_2))),
        'h3': tf.Variable(tf.random_uniform([n_hidden_2, n_hidden_3], -1/ np.sqrt(n_hidden_3), 1./ np.sqrt(n_hidden_3))),
        'out': tf.Variable(tf.random_uniform([n_hidden_3, num_output], -0.003, 0.003)),
    }
    return weights

# def initial_weights (num_input, n_hidden_1, n_hidden_2, n_hidden_3, num_output):
#     weights = {
#         'h1': tf.Variable(tf.truncated_normal([num_input, n_hidden_1]) / np.sqrt(num_input)),
#         'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]) / np.sqrt(n_hidden_1)),
#         'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3]) / np.sqrt(n_hidden_2)),
#         'out': tf.Variable(tf.truncated_normal([n_hidden_3, num_output]) / np.sqrt(n_hidden_3)),
#     }
#     return weights



def update_weights (source_weights, destination_weights):
    destination_weights['h1']=tf.identity(source_weights['h1'])
    destination_weights['h2']=tf.identity(source_weights['h2'])
    destination_weights['h3']=tf.identity(source_weights['h3'])
    destination_weights['out']=tf.identity(source_weights['out'])
    #return 0

def initial_biases (n_hidden_1, n_hidden_2, n_hidden_3, num_output):
    biases = {
        'b1': tf.Variable(tf.random_uniform([n_hidden_1], -1./np.sqrt(n_hidden_1),1./np.sqrt(n_hidden_1))),
        'b2': tf.Variable(tf.random_uniform([n_hidden_2], -1./ np.sqrt(n_hidden_2), 1./ np.sqrt(n_hidden_2))),
        'b3': tf.Variable(tf.random_uniform([n_hidden_3], -1/ np.sqrt(n_hidden_3), 1./ np.sqrt(n_hidden_3))),
        'out': tf.Variable(tf.random_uniform([num_output], -0.003, 0.003)),
    }
    return biases
# def initial_biases (n_hidden_1, n_hidden_2, n_hidden_3, num_output):
#     biases = {
#         'b1': tf.Variable(tf.ones([n_hidden_1]) * 0.001),
#         'b2': tf.Variable(tf.ones([n_hidden_2]) * 0.001),
#         'b3': tf.Variable(tf.ones([n_hidden_3]) * 0.001),
#         'out': tf.Variable(tf.ones([num_output]) * 0.001),
#     }
#     return biases

def update_biases (source_biases, destination_biases):
    destination_biases['b1']=tf.identity(source_biases['b1'])
    destination_biases['b2']=tf.identity(source_biases['b2'])
    destination_biases['b3']=tf.identity(source_biases['b3'])
    destination_biases['out']=tf.identity(source_biases['out'])
    #return None

def get_gains_hexagon_neighbors_shadowinginc (N,K,R,min_dist,total_samples,shadowing_dev=10,dcor=10,
                                              equal_number_for_BS=True,draw=False,
                                              T=20e-3,train_episodes = None,mobility_params = None):
    TX_loc, RX_loc, TX_xhex, TX_yhex,neighbors, mirrors, N = get_random_locationshexagon_neighbors(N,K,R,
                                                                                                   min_dist,equal_number_for_BS = equal_number_for_BS,
                                                                                                   total_samples = total_samples,T=T,
                                                                                                   train_episodes = train_episodes,mobility_params = mobility_params)
    distance_vector = get_distance(N,TX_loc, RX_loc, mirrors = mirrors, total_samples = total_samples)
    
    gains = []
    # Get 2D distance pathloss, original pathloss tried in the previous versions
    # Get channel gains
    g_dB2 = - (128.1 + 37.6* np.log10(0.001*distance_vector))
    
    # init_shadowing
    tmp_g_dB = np.zeros((N,N,total_samples))
    shadowing = np.zeros((K,N,total_samples))
    shadowing[:,:,0] = np.random.randn(K,N)
    RX_displacement = mirrors['RX_displacement']
    cell_mapping = mirrors['cell_mapping']
    for sample in range(1,total_samples):
        for n in range(K):
            correlation = np.exp(- np.sqrt(RX_displacement[0,:,sample]**2+RX_displacement[1,:,sample]**2)/dcor)
            shadowing[n,:,sample] = np.multiply(correlation,shadowing[n,:,sample-1]) + np.multiply(np.sqrt(1.0-np.square(correlation)),np.random.randn(1,N))
    for sample in range(total_samples):
        for k in range(N):
            tmp_g_dB[k,:,sample] = g_dB2[k,:,sample] + shadowing_dev * shadowing[cell_mapping[:,sample],k,sample]
    gains.append(np.power(10.0,tmp_g_dB/10.0))
        
    if (draw == True):
#        plt.plot(TX_loc.T, 'b^')
#        plt.plot(RX_loc.T, 'ro')
        for i in range(K):
            plt.plot(TX_loc[0,i],TX_loc[1,i],'g^')
            plt.plot(TX_xhex [:,i],TX_yhex [:,i],'k-')
        for i in range(N):
            plt.plot(RX_loc[0,i,0],RX_loc[1,i,0],'ro')
        plt.show()
    return gains,TX_loc,RX_loc,TX_xhex, TX_yhex, neighbors, mirrors


# Ray tracing
def inside_hexagon(x,y,TX_xhex,TX_yhex):
    n = len(TX_xhex)-1
    inside = False
    p1x,p1y = TX_xhex[0],TX_yhex[0]
    for i in range(n+1):
        p2x,p2y = TX_xhex[i % n],TX_yhex[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


def get_random_locationshexagon_neighbors(N,K,R,min_dist,equal_number_for_BS = True,
                                          total_samples = None,bvar=1.0, T=20e-3,train_episodes = None,mobility_params=None):
#    assert N % UE_perBS == 0, 'N needs to be divisible by UE_perBS!'
#    # Kumber of BSs
#    K = int(N / UE_perBS)
    mirrors = []
    cell_mapping = []
    # Brownian motion case...
    neighbors = [] # neighboring cells
    
    # IMAC Case: we have the mirror BS at the same location.
    mirrors = []
    max_dist = R
    x_hexagon=R*np.array([0, -np.sqrt(3)/2, -np.sqrt(3)/2, 0, np.sqrt(3)/2, np.sqrt(3)/2, 0])
    y_hexagon=R*np.array([-1, -0.5, 0.5, 1, 0.5, -0.5, -1])

    TX_loc = np.zeros((2,K))
    TX_xhex = np.zeros((7,K))
    TX_yhex = np.zeros((7,K))
    
    RX_loc = np.zeros((2,N,total_samples))
    cell_mapping = np.zeros((N,total_samples)).astype(int)
    RX_displacement = np.zeros((4,N,total_samples)) # displacement and angle
    
############### DROP KLL KKTERS    
    generated_hexagons = 0
    i = 0
    #if (N>0):
    for k in range(1):
        TX_loc [0, generated_hexagons*1 + k] = 0.0
        TX_loc [1, generated_hexagons*1 + k] = 0.0
        TX_xhex [:,generated_hexagons*1 + k] = x_hexagon
        TX_yhex [:,generated_hexagons*1 + k] = y_hexagon
    generated_hexagons += 1

    while(generated_hexagons < K):

        for j in range(6):
            tmp_xloc = TX_loc [0, i]+np.sqrt(3)*R*np.cos(j*np.pi/(3))
            tmp_yloc = TX_loc [1, i]+np.sqrt(3)*R*np.sin(j*np.pi/(3))
            tmp_xhex = tmp_xloc+x_hexagon
            tmp_yhex = tmp_yloc+y_hexagon
            was_before = False
            for inner_loop in range(generated_hexagons):
                if (abs(tmp_xloc-TX_loc [0, inner_loop*1])<R*1e-2 and abs(tmp_yloc-TX_loc [1, inner_loop*1])<R*1e-2):
                    was_before = True
                    break
            if (not was_before):
                for k in range(1):
                    TX_loc [0, generated_hexagons*1 + k] = tmp_xloc
                    TX_loc [1, generated_hexagons*1 + k] = tmp_yloc
                    TX_xhex [:,generated_hexagons*1 + k] = tmp_xhex
                    TX_yhex [:,generated_hexagons*1 + k] = tmp_yhex      
                generated_hexagons += 1
            if(generated_hexagons>= K):
                break
        i += 1
    # Then find the neighbors
    for i in range(K):
        tmp_neighbors = []
        for j in range(6):
            tmp_xloc = TX_loc [0, i]+np.sqrt(3)*R*np.cos(j*np.pi/(3))
            tmp_yloc = TX_loc [1, i]+np.sqrt(3)*R*np.sin(j*np.pi/(3))
            tmp_xhex = tmp_xloc+x_hexagon
            tmp_yhex = tmp_yloc+y_hexagon
            for inner_loop in range(K):
                if (inner_loop != i and abs(tmp_xloc-TX_loc [0, inner_loop])<R*1e-2 and abs(tmp_yloc-TX_loc [1, inner_loop])<R*1e-2):
                    tmp_neighbors.append(inner_loop)
        for j in range(1):
            neighbors.append(tmp_neighbors)
############### DROP USERS
    a_max = mobility_params['a_max']
    v_max = mobility_params['v_max']
    alpha_angle = mobility_params['alpha_angle']
    T_mobility = mobility_params['T_mobility']
    for i in range(N):
        # Randomly assign initial cell placement
        if equal_number_for_BS:
            assert N % K == 0
            UE_perBS = int(N/K)
            cell_mapping[i,0] = int(i/UE_perBS)
        else:
            cell_mapping[i,0] = np.random.randint(K)
        this_cell = cell_mapping[i,0]
        # Place UE within that cell.
        constraint_minx_UE=min(TX_xhex[:,this_cell])
        constraint_maxx_UE=max(TX_xhex[:,this_cell])
        constraint_miny_UE=min(TX_yhex[:,this_cell])
        constraint_maxy_UE=max(TX_yhex[:,this_cell])
        inside_checker = True
        while (inside_checker):
            RX_displacement[2,i,0] = np.random.uniform(0,v_max) # Initial speed.
            RX_displacement[3,i,0] = np.random.uniform(-np.pi,np.pi) # Initial angle.
            RX_loc[0,i,0]=np.random.uniform(constraint_minx_UE,constraint_maxx_UE)
            RX_loc[1,i,0]=np.random.uniform(constraint_miny_UE,constraint_maxy_UE)
            tmp_distance2center = np.sqrt(np.square(RX_loc[0,i,0]-TX_loc [0, this_cell])+np.square(RX_loc[1,i,0]-TX_loc [1, this_cell]))
            if(inside_hexagon(RX_loc[0,i,0],RX_loc[1,i,0],TX_xhex[:,this_cell],TX_yhex[:,this_cell])
                and tmp_distance2center>min_dist and tmp_distance2center<max_dist):
                inside_checker = False
############### MOVE USERS
    step_size = T_mobility
    
    sleep_step_size = 1.0 + float(train_episodes['T_sleep'])/float(train_episodes['T_train'])            
    RX_loc_all = np.zeros((2,N,int(total_samples*sleep_step_size)))
    RX_loc_all[:,:,0] = RX_loc[:,:,0]
    cell_mapping_all = np.zeros((N,int(total_samples*sleep_step_size))).astype(int)
    cell_mapping_all[:,0] = cell_mapping[:,0]
    cell_request_change = (-1 * np.ones(N)).astype(int)
    cell_request_counter = train_episodes['T_register'] * np.ones(N)
    RX_displacement_all = np.zeros((4,N,int(total_samples*sleep_step_size))) # displacement and angle
    RX_displacement_all[:,:,0] = RX_displacement[:,:,0] # Initial speed and angle.
    is_mode_sleep = False
    is_mode_train = True
    sample_train = 0
    for sample in range(1,int(total_samples*sleep_step_size)):
        # Get the mode if necessary
        if sample != 1 and (sample) % train_episodes['T_train'] == 0:
            is_mode_sleep = True
            is_mode_train = False
        if (sample) % (train_episodes['T_sleep']+train_episodes['T_train']) == 0:
            is_mode_sleep = False
            is_mode_train = True
        if is_mode_train:
            sample_train += 1
        for i in range(N):
            if sample % step_size == 0:
                delta_v =np.random.uniform(-a_max,a_max)
                delta_angle =np.random.uniform(-alpha_angle,alpha_angle)
                RX_displacement_all[2,i,sample] = min(max(RX_displacement_all[2,i,sample-1] + delta_v, 0.0),v_max) # v
                RX_displacement_all[3,i,sample] = RX_displacement_all[3,i,sample-1] + delta_angle # angle
            else:
                RX_displacement_all[2,i,sample] = RX_displacement_all[2,i,sample-1]
                RX_displacement_all[3,i,sample] = RX_displacement_all[3,i,sample-1]
            if cell_request_change[i] == -1:
                prev_cell = cell_mapping_all[i,sample-1]
                prev_cell_map = prev_cell
            elif cell_request_counter[i] > 0:
                cell_request_counter[i] = cell_request_counter[i] - 1
                prev_cell = cell_request_change[i]
                prev_cell_map = cell_mapping_all[i,sample-1]
            else:
                prev_cell = cell_request_change[i]
                prev_cell_map = prev_cell
                cell_request_change[i] = -1
                cell_request_counter[i] = train_episodes['T_register']
            
            constraint_minx_UE=min(TX_xhex[:,prev_cell])
            constraint_maxx_UE=max(TX_xhex[:,prev_cell])
            constraint_miny_UE=min(TX_yhex[:,prev_cell])
            constraint_maxy_UE=max(TX_yhex[:,prev_cell])
            inside_checker = True
            while (inside_checker):
                RX_displacement_all[0,i,sample] = T * RX_displacement_all[2,i,sample] * np.cos(RX_displacement_all[3,i,sample]) # displacement x
                RX_displacement_all[1,i,sample] = T * RX_displacement_all[2,i,sample] * np.sin(RX_displacement_all[3,i,sample]) # displacement y
                RX_loc_all[0,i,sample]=RX_loc_all[0,i,sample-1]+RX_displacement_all[0,i,sample]
                RX_loc_all[1,i,sample]=RX_loc_all[1,i,sample-1]+RX_displacement_all[1,i,sample]
                tmp_distance2center = np.sqrt(np.square(RX_loc_all[0,i,sample]-TX_loc [0, prev_cell])+np.square(RX_loc_all[1,i,sample]-TX_loc[1, prev_cell]))                        
                if is_mode_train:                            
                    RX_displacement[:,i,sample_train] = RX_displacement_all[:,i,sample]
                    RX_loc[0,i,sample_train]=RX_loc_all[0,i,sample-1]
                    RX_loc[1,i,sample_train]=RX_loc_all[1,i,sample-1]
                if(inside_hexagon(RX_loc_all[0,i,sample],RX_loc_all[1,i,sample],TX_xhex[:,prev_cell],TX_yhex[:,prev_cell])
                    and tmp_distance2center>min_dist and tmp_distance2center<max_dist):
                    inside_checker = False
                    cell_mapping_all[i,sample] = prev_cell_map # The UE is still inside the prev cell
                    if is_mode_train:
                        cell_mapping[i,sample_train] = prev_cell_map
                elif (is_mode_train and train_episodes['cell_passing_training']) or (is_mode_sleep and train_episodes['cell_passing_sleeping']): # Kow check the immediate neighbors of the cell to see whether UE is passing to another cell or it is out of bounds.
                    for neigh in neighbors[prev_cell]:
                        tmp_distance2center = np.sqrt(np.square(RX_loc_all[0,i,sample]-TX_loc [0, neigh])+np.square(RX_loc_all[1,i,sample]-TX_loc [1, neigh]))
                        if(inside_hexagon(RX_loc_all[0,i,sample],RX_loc_all[1,i,sample],TX_xhex[:,neigh],TX_yhex[:,neigh])
                            and tmp_distance2center>min_dist and tmp_distance2center<max_dist):
                            inside_checker = False
                            cell_mapping_all[i,sample] = prev_cell_map # The UE is still inside the prev cell
                            if is_mode_train:
                                cell_mapping[i,sample_train] = prev_cell_map
                            if neigh == prev_cell_map: # Cell get back to original cell, dismiss register.
                                cell_request_change[i] = -1
                                cell_request_counter[i] = train_episodes['T_register']
                            else:
                                cell_request_change[i] = neigh
                                cell_request_counter[i] = train_episodes['T_register']                                
                            break
                    if inside_checker:
                        # If none of the edges worked boucne back with a random angle.
                        RX_displacement_all[3,i,sample] = np.random.uniform(-np.pi,np.pi)
                else: # cell passing is not allowed boucne back with a random angle.
                    RX_displacement_all[3,i,sample] = np.random.uniform(-np.pi,np.pi)                      
                #If user is out of bounds, redo the motion.
    # Don't want to modify the input structure hence the mirrors used for other cases will be used as a dictionary.
    mirrors = {}
    mirrors['cell_mapping'] = cell_mapping
    mirrors['RX_displacement'] = RX_displacement
    if train_episodes is not None:
        mirrors['RX_loc_all'] = RX_loc_all
        mirrors['cell_mapping_all'] = cell_mapping_all
        mirrors['RX_displacement_all'] = RX_displacement_all
    return TX_loc, RX_loc, TX_xhex, TX_yhex, neighbors, mirrors, N
def positive_or_negative():
    if np.random.uniform(0,1) < 0.5:
        return 1
    else:
        return -1

def get_distance(N,TX_loc, RX_loc, mirrors = None, total_samples = 1):
    distance_vector = np.zeros((N,N,total_samples))
    cell_mapping = mirrors['cell_mapping']
    tmp_TX_loc = np.zeros((2,N))
    for sample in range(total_samples):
        
        tmp_TX_loc = TX_loc[:,cell_mapping[:,sample]]
        for i in range (N):
            distance_vector[:,i,sample]=np.sqrt(np.square(tmp_TX_loc[0,i]-RX_loc[0,:,sample])+
                           np.square(tmp_TX_loc[1,i]-RX_loc[1,:,sample]))   
            
    return distance_vector
