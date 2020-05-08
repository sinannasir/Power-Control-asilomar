# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""
import numpy as np
#import matplotlib.pyplot as plt
import project_backend as pb
import tensorflow as tf
import collections
import copy

class DDPG:
    def __init__(self, options,options_policy,N,Pmax,noise_var):
        tf.reset_default_graph()
        self.total_samples = options['simulation']['total_samples']
        self.train_episodes = options['train_episodes']
        R_defined = options['simulation']['R_defined']
        self.R = (2.0/np.sqrt(3))*R_defined
        self.N = N
        self.Pmax = Pmax
        self.noise_var = noise_var
        self.tmp_exp_type_1 = []
        self.tmp_exp_type_2 = []
        self.prev_suminterferences = np.zeros(N)
        for i in range(self.N):
            self.tmp_exp_type_1.append(collections.deque([],4))
            self.tmp_exp_type_2.append(collections.deque([],3))
        
        self.num_output = self.num_actions = 1 # Kumber of actions
        self.discount_factor = options_policy['discount_factor']
        
        self.N_neighbors = options_policy['N_neighbors']
        self.num_input = 6 + 7 * self.N_neighbors
        learning_rate_0 = options_policy['learning_rate_0_critic']
        learning_rate_decay = options_policy['learning_rate_decay_critic']
        learning_rate_min = options_policy['learning_rate_min_critic']
        self.learning_rate_all_critic = [learning_rate_0]
        for i in range(1,self.total_samples):
            if i % self.train_episodes['T_train'] == 0:
                self.learning_rate_all_critic.append(learning_rate_0)
            else:
                self.learning_rate_all_critic.append(max(learning_rate_min,learning_rate_decay*self.learning_rate_all_critic[-1]))
        learning_rate_0 = options_policy['learning_rate_0_actor']
        learning_rate_decay = options_policy['learning_rate_decay_actor']
        learning_rate_min = options_policy['learning_rate_min_actor']
        self.learning_rate_all_actor = [learning_rate_0]
        for i in range(1,self.total_samples):
            if i % self.train_episodes['T_train'] == 0:
                self.learning_rate_all_actor.append(learning_rate_0)
            else:
                self.learning_rate_all_actor.append(max(learning_rate_min,learning_rate_decay*self.learning_rate_all_actor[-1]))
    
        self.batch_size = options_policy['batch_size']
        memory_per_agent = options_policy['memory_per_agent']
        # epsilon greedy algorithm
        max_epsilon = options_policy['max_epsilon']
        epsilon_decay = options_policy['epsilon_decay']
        min_epsilon = options_policy['min_epsilon']
        # quasi-static target network update
        self.target_update_count = options_policy['target_update_count']
        self.time_slot_to_pass_weights = options_policy['time_slot_to_pass_weights'] # 50 slots needed to pass the weights
        n_hidden_1 = options_policy['n_hiddens'][0]
        n_hidden_2 = options_policy['n_hiddens'][1]
        n_hidden_3 = options_policy['n_hiddens'][2]
        scale_R_inner = options_policy['scale_R_inner']
        scale_R_interf = options_policy['scale_R_interf']
        scale_g_dB_R = scale_R_inner*self.R
        rb = 200.0
        if(scale_g_dB_R < rb):
            scale_g_dB = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_R))
        else:
            scale_g_dB = - (128.1 + 37.6* np.log10(scale_g_dB_R/rb) + 37.6* np.log10(0.001*rb)) 
        self.scale_gain = np.power(10.0,scale_g_dB/10.0)
        self.input_placer = np.log10(self.noise_var/self.scale_gain)
        scale_g_dB_inter_R = scale_R_interf * self.R
        if(scale_g_dB_R < rb):
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_inter_R))
        else:
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(scale_g_dB_inter_R/rb) + 37.6* np.log10(0.001*rb))
        self.scale_gain_interf = np.power(10.0,scale_g_dB_interf/10.0)
        
        # Experience-replay memory size
        self.memory_len = memory_per_agent*N
        # learning rate
    
        # epsilon greedy algorithm       
        self.epsilon_all=[max_epsilon]
        for i in range(1,self.total_samples):
            if i % self.train_episodes['T_train'] == 0:
#                if int(i/self.train_episodes['T_train']) == (self.total_samples/self.train_episodes['T_train']-1):
#                    self.epsilon_all.append(0.0) # Test scenario
#                else:
                self.epsilon_all.append(max_epsilon)
            else:
                self.epsilon_all.append(max(min_epsilon,epsilon_decay*self.epsilon_all[-1]))
        
        # Experience replay memory
        self.memory = {}
        self.memory['s'] = collections.deque([],self.memory_len+self.N)
        self.memory['s_prime'] = collections.deque([],self.memory_len+self.N)
        self.memory['rewards'] = collections.deque([],self.memory_len+self.N)
        self.memory['actions'] = collections.deque([],self.memory_len+self.N)
        
        self.previous_state = np.zeros((self.N,self.num_input))
        self.previous_action = np.ones(self.N) * self.num_actions
       
        # required for session to know whether dictionary is train or test
        self.is_train = tf.placeholder("bool")
        
        ##
        # Critic Ketwork
        self.x_s_critic = tf.placeholder("float", [None, self.num_input])
        self.x_a_critic = tf.placeholder("float", [None, self.num_actions])
        self.y_critic = tf.placeholder("float", [None, 1])
        self.x_s_critic_target = tf.placeholder("float", [None, self.num_input])
        self.x_a_critic_target = tf.placeholder("float", [None, self.num_actions])
        self.y_critic_target = tf.placeholder("float", [None, 1])
        with tf.name_scope("C_weights"):
            self.weights_critic = pb.initial_weights (self.num_input+self.num_actions, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output)
        with tf.name_scope("Ctarget_weights"): 
            self.weights_target_critic = pb.initial_weights (self.num_input+self.num_actions, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output)
        with tf.name_scope("C_biases"):
            self.biases_critic = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output)
        with tf.name_scope("Ctarget_biases"): 
            self.biases_target_critic = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output)
            
        # initialize the neural network for each agent
        self.critic= pb.critic_net(self.x_s_critic,self.x_a_critic, self.weights_critic, self.biases_critic)
        self.critic_target = pb.critic_net(self.x_s_critic_target,self.x_a_critic_target, self.weights_target_critic,
                                            self.biases_target_critic)
        
        self.action_grads_v = tf.gradients(self.critic, self.x_a_critic)
        self.action_grads = [self.action_grads_v[0]]#/(tf.to_float(tf.shape(self.action_grads_v[0])[0]))]#*self.batch_size)]
        
        # l2_regularizer_loss = 0.001*tf.reduce_sum(tf.pow(self.weights_critic['h2'],2))
        self.critic_loss = tf.nn.l2_loss(self.y_critic_target - self.critic) # + l2_regularizer_loss
        
        self.c_loss = []
        self.c_loss_track = []
        # self.critic_loss = tf.reduce_mean(tf.pow(self.y_critic_target- self.critic,2)) #+ l2_regularizer_loss
        self.critic_learning_rate = (tf.placeholder('float'))
        # self.critic_optimizer = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.critic_loss)
        self.critic_optimizer = tf.train.RMSPropOptimizer(self.critic_learning_rate, decay=0.9,
                                              epsilon=1e-10).minimize(self.critic_loss)
        

        # Actor Ketwork
        self.x_actor = tf.placeholder("float", [None, self.num_input])
        self.y_actor = tf.placeholder("float", [None, 1])
        self.x_actor_agent = tf.placeholder("float", [None, self.num_input])
        with tf.name_scope("A_weights"):
            self.weights_actor = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output)
        with tf.name_scope("Aagent_weights"): 
            self.weights_target_actor = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output)
        with tf.name_scope("Abroadcast_weights"): 
            self.weights_tmp_actor = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output)
        with tf.name_scope("A_biases"):
            self.biases_actor = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output)
        with tf.name_scope("Aagent_biases"): 
            self.biases_target_actor = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output)
        with tf.name_scope("Abroadcast_biases"): 
            self.biases_tmp_actor = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output)
            
        
            
        # initialize the neural network for each agent
        self.actor= pb.actor_net(self.x_actor, self.weights_actor, self.biases_actor)
        self.actor_agent = pb.actor_net(self.x_actor_agent, self.weights_target_actor,
                                            self.biases_target_actor)
        
        self.critic_gradient = tf.placeholder(tf.float32, [None, self.num_output])
        
        self.actor_params = self.get_params('A_')
        self.policy_gradients = tf.gradients(self.actor, self.actor_params, -self.critic_gradient)       
        self.actor_learning_rate = (tf.placeholder('float'))
        # Adam
        # self.actor_optimizer = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(zip(self.policy_gradients,self.actor_params))
        # RMSprop algorithm used
        self.actor_optimizer = tf.train.RMSPropOptimizer(self.actor_learning_rate, decay=0.9,
                                              epsilon=1e-10).apply_gradients(zip(self.policy_gradients,self.actor_params))

        self.init = tf.global_variables_initializer()
        # quasi-static target update simulation counter = 0
        self.saver = tf.train.Saver()
        
        self.std = tf.placeholder("float")
        self.noise = tf.random_uniform(shape = (1, 1), minval=-self.std, maxval=self.std)
    
    def get_params(self, para_name):
        sets=[]
        for var in tf.trainable_variables():
            if not var.name.find(para_name):
                sets.append(var)
        return sets

    def initialize_critic_updates(self,sess): # Keed to rund this before calling quasi static.
        self.saver = tf.train.Saver(tf.global_variables())
        self.update_class1_critic = []
        for (w,tmp_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='C_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ctarget_weights')):
            self.update_class1_critic.append(tf.assign(tmp_w,w))
            sess.run(self.update_class1_critic[-1])
        for (b,tmp_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='C_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Ctarget_biases')):
            self.update_class1_critic.append(tf.assign(tmp_b,b))
            sess.run(self.update_class1_critic[-1])
        print('first critic update')

    def initialize_actor_updates(self,sess): # Keed to rund this before calling quasi static.
        self.saver = tf.train.Saver(tf.global_variables())
        self.update_class1 = []
        for (w,tmp_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='A_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Abroadcast_weights')):
            self.update_class1.append(tf.assign(tmp_w,w))
            sess.run(self.update_class1[-1])
        for (b,tmp_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='A_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Abroadcast_biases')):
            self.update_class1.append(tf.assign(tmp_b,b))
            sess.run(self.update_class1[-1])
        self.update_class2 = []
        for (tmp_w,t_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Abroadcast_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Aagent_weights')):
            self.update_class2.append(tf.assign(t_w,tmp_w))
            sess.run(self.update_class2[-1])
        for (tmp_b,t_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Abroadcast_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Aagent_biases')):
            self.update_class2.append(tf.assign(t_b,tmp_b))
            sess.run(self.update_class2[-1])
        self.simulation_target_update_counter = self.target_update_count
        self.process_weight_update = False
        self.simulation_target_pass_counter = self.time_slot_to_pass_weights
        print('first update')
    
    def check_memory_restart(self,sess,sim):   
        if(sim %self.train_episodes['T_train'] == 0 and sim != 0): # Restart experience replay.
            self.memory = {}
            self.memory['s'] = collections.deque([],self.memory_len+self.N)
            self.memory['s_prime'] = collections.deque([],self.memory_len+self.N)
            self.memory['rewards'] = collections.deque([],self.memory_len+self.N)
            self.memory['actions'] = collections.deque([],self.memory_len+self.N)
            
            self.previous_state = np.zeros((self.N,self.num_input))
            self.previous_action = np.ones(self.N) * self.num_actions
    
    def update_handler(self,sess,sim):
        # First check whether target network has to be changed.
        self.simulation_target_update_counter -= 1
        # Update critic all the time after training
        for update_instance in self.update_class1_critic:
            sess.run(update_instance)
        # Actor broadcast
        if (self.simulation_target_update_counter == 0):
            for update_instance in self.update_class1:
                sess.run(update_instance)
            self.simulation_target_update_counter = self.target_update_count
            self.process_weight_update = True

        if self.process_weight_update:
            self.simulation_target_pass_counter -= 1
        
        if (self.simulation_target_pass_counter <= 0):
            for update_instance in self.update_class2:
                sess.run(update_instance)
            self.process_weight_update = False
            self.simulation_target_pass_counter = self.time_slot_to_pass_weights
            
    def act(self,sess,current_local_state,sim,actor_idx):
        # for stability return something random for first 100 time slots.
        if sim<500 and np.random.rand() < 0.25:
            return 0.
        # epsilon greedy algorithm
        if np.random.rand() < self.epsilon_all[sim]:# or sum(self.previous_action>0.95)==self.N or sum(self.previous_action<0.01)==self.N:
            strategy = np.random.rand()
            return strategy
        
        strategy = sess.run(self.actor_agent, feed_dict={self.x_actor_agent: current_local_state.reshape(1,self.num_input), self.is_train: False})[0][0]
        return strategy
    
    def act_noepsilon(self,sess,current_local_state,sim):
        # Current QNN outputs for all available actions
        return sess.run(self.actor_agent, feed_dict={self.x_actor_agent: current_local_state.reshape(1,self.num_input), self.is_train: False})[0][0]
    
    def remember(self,agent,current_local_state,current_reward):
        self.memory['s'].append(copy.copy(self.previous_state[agent,:]).reshape(self.num_input))
        self.memory['s_prime'].append(copy.copy(current_local_state))
        self.memory['actions'].append(copy.copy(self.previous_action[agent]))
        self.memory['rewards'].append(copy.copy(current_reward))
    
    def train(self,sess,sim):
        # skip training for 100 time slots.
        # if sim < 100: return
        if len(self.memory['s']) >= self.batch_size+self.N:
            # Minus N ensures that experience samples from previous timeslots been used
            idx = np.random.randint(len(self.memory['rewards'])-self.N,size=self.batch_size)
            s_prime_shaped = np.array(self.memory['s_prime'])[idx, :].reshape(self.batch_size,self.num_input)
            action_t_1_batch = sess.run(self.actor_agent, feed_dict={self.x_actor_agent: s_prime_shaped})
            #Q'(s_i+1,a_i+1)
            q_t_1 = sess.run(self.critic_target, feed_dict={self.x_s_critic_target: s_prime_shaped,
                                                            self.x_a_critic_target: action_t_1_batch,self.is_train: False})
            
            y_batch = np.array(self.memory['rewards'])[idx].reshape(self.batch_size,1) + self.discount_factor * q_t_1
            
            s_shaped = np.array(self.memory['s'])[idx, :].reshape(self.batch_size,self.num_input)
            (tmp,tmp_critloss) = sess.run([self.critic_optimizer, self.critic_loss], feed_dict={self.critic_learning_rate:self.learning_rate_all_critic[sim],
                                self.x_s_critic: s_shaped,
                                self.x_a_critic: np.array(self.memory['actions'])[idx].reshape(self.batch_size,self.num_actions),
                                self.y_critic_target: y_batch.reshape(self.batch_size,1), self.is_train: True})
            self.c_loss_track.append(tmp_critloss)
            if sim%100==0:
                self.c_loss.append(np.mean(self.c_loss_track))
                self.c_loss_track = []
            # if sim%5==0:
            action_for_delQ = sess.run(self.actor, feed_dict={self.x_actor:s_shaped})    
            del_Q_a = sess.run(self.action_grads, feed_dict={self.x_s_critic: s_shaped,
                                                                 self.x_a_critic: action_for_delQ,self.is_train: False})[0]
            tmp = sess.run([self.actor_optimizer], feed_dict={self.actor_learning_rate:self.learning_rate_all_actor[sim],
                                                              self.x_actor: s_shaped, 
                                                self.critic_gradient: del_Q_a, self.is_train: True})
            

    
    def equalize(self,sess):
        for update_instance in self.update_class1:
            sess.run(update_instance)
        for update_instance in self.update_class2:
            sess.run(update_instance)
            
    def save(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        save_path = self.saver.save(sess, model_destination)
        print("Model saved in path: %s" % save_path)
        
    def load(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(sess, model_destination)
        print('Model loaded from: %s' %(model_destination))
        
    def local_state(self,sim,agent,p_strategy_all,H_all_2,neighbors,neighbors_in,sum_rate_list_distributed_policy,sims_pos_p):
        current_experiences = np.zeros(self.num_input)
        if(p_strategy_all[-1][agent]==0):
            current_experiences[0] = 0.0
        else:
            current_experiences[0] = (p_strategy_all[-1][agent])/self.Pmax
        current_experiences[1] = np.log10(H_all_2[sim][agent,:][agent]/self.scale_gain)
        
        current_experiences[2] = np.log10(H_all_2[sim-1][agent,:][agent]/self.scale_gain)
        current_experiences[3] = 0.5 * sum_rate_list_distributed_policy[-1].diagonal()[agent] # maximum value of sum-rate is around 10, so we wanna slightly reduce for better performance.
        if(len(np.where(np.delete(p_strategy_all[-2],agent)==0)[0])!=self.N-1):
            current_experiences[4] = np.log10((self.noise_var+np.matmul(np.delete(H_all_2[sim-2][agent,:],agent),
                                           np.delete(p_strategy_all[-2],agent)))/(self.scale_gain))
        else:
            current_experiences[4] = self.input_placer
        if(len(np.where(np.delete(p_strategy_all[-1],agent)==0)[0])!=self.N-1):
            current_experiences[5] = np.log10((self.noise_var+np.matmul(np.delete(H_all_2[sim-1][agent,:],agent),
                                           np.delete(p_strategy_all[-1],agent)))/(self.scale_gain))                                
        else:
            current_experiences[5] = self.input_placer      
        if(len(self.tmp_exp_type_1[agent]) == 0):
            if(len(neighbors_in[-2][agent]) != 0):
                self.tmp_exp_type_1[agent].append(np.log10(np.multiply(H_all_2[sim-2][agent,neighbors_in[-2][agent]],p_strategy_all[-2][neighbors_in[-2][agent]])/(self.scale_gain_interf)))
                
                tmp_exp_type_1_index = np.argsort(self.tmp_exp_type_1[agent][-1])[::-1]
                self.tmp_exp_type_1[agent][-1] = self.tmp_exp_type_1[agent][-1][tmp_exp_type_1_index]
                self.tmp_exp_type_1[agent].append(0.5 * sum_rate_list_distributed_policy[-2].diagonal()[neighbors_in[-2][agent]][tmp_exp_type_1_index])
            else:
                self.tmp_exp_type_1[agent].append(np.array([]))
                self.tmp_exp_type_1[agent].append(np.array([]))
            # Append negative numbers if needed
            if (len(self.tmp_exp_type_1[agent][-2]) < self.N_neighbors):
                self.tmp_exp_type_1[agent][-2] = np.append(self.tmp_exp_type_1[agent][-2],(self.N_neighbors - len(self.tmp_exp_type_1[agent][-2]))*[self.input_placer])
                self.tmp_exp_type_1[agent][-1] = np.append(self.tmp_exp_type_1[agent][-1],(self.N_neighbors - len(self.tmp_exp_type_1[agent][-1]))*[self.input_placer])
        if(len(neighbors_in[-1][agent]) != 0):
            self.tmp_exp_type_1[agent].append(np.log10(np.multiply(H_all_2[sim-1][agent,neighbors_in[-1][agent]],p_strategy_all[-1][neighbors_in[-1][agent]])/(self.scale_gain_interf)))
            tmp_exp_type_1_index = np.argsort(self.tmp_exp_type_1[agent][-1])[::-1]
            self.tmp_exp_type_1[agent][-1] = self.tmp_exp_type_1[agent][-1][tmp_exp_type_1_index]
            self.tmp_exp_type_1[agent].append(0.5 * sum_rate_list_distributed_policy[-1].diagonal()[neighbors_in[-1][agent]][tmp_exp_type_1_index])
        else:
            self.tmp_exp_type_1[agent].append(np.array([]))
            self.tmp_exp_type_1[agent].append(np.array([]))                   
        # Append negative numbers if needed
        if (len(self.tmp_exp_type_1[agent][-2]) < self.N_neighbors):
            self.tmp_exp_type_1[agent][-2] = np.append(self.tmp_exp_type_1[agent][-2],(self.N_neighbors - len(self.tmp_exp_type_1[agent][-2]))*[self.input_placer])
            self.tmp_exp_type_1[agent][-1] = np.append(self.tmp_exp_type_1[agent][-1],(self.N_neighbors - len(self.tmp_exp_type_1[agent][-1]))*[-1])
        current_experiences[(6 + 0 * self.N_neighbors):(6 + 1 * self.N_neighbors)] = self.tmp_exp_type_1[agent][-1][:self.N_neighbors]
        current_experiences[(6 + 1 * self.N_neighbors):(6 + 2 * self.N_neighbors)] = self.tmp_exp_type_1[agent][-2][:self.N_neighbors]
        current_experiences[(6 + 2 * self.N_neighbors):(6 + 3 * self.N_neighbors)] = self.tmp_exp_type_1[agent][-3][:self.N_neighbors]
        current_experiences[(6 + 3 * self.N_neighbors):(6 + 4 * self.N_neighbors)] = self.tmp_exp_type_1[agent][-4][:self.N_neighbors]
        
        current_experiences[(6 + 4 * self.N_neighbors):(6 + 5 * self.N_neighbors)] = current_experiences[(6 + 4 * self.N_neighbors):(6 + 5 * self.N_neighbors)] + self.input_placer
        current_experiences[(6 + 5 * self.N_neighbors):(6 + 6 * self.N_neighbors)] = current_experiences[(6 + 5 * self.N_neighbors):(6 + 6 * self.N_neighbors)] + self.input_placer
        current_experiences[(6 + 6 * self.N_neighbors):(6 + 7 * self.N_neighbors)] = current_experiences[(6 + 6 * self.N_neighbors):(6 + 7 * self.N_neighbors)] + self.input_placer
        if(len(neighbors[-1][agent])>0 and p_strategy_all[-1][agent] != 0):
            self.tmp_exp_type_2[agent].append(np.log10(H_all_2[sim-1][np.array(neighbors[-1][agent]),agent]/self.prev_suminterferences[neighbors[-1][agent]]))
            tmp_exp_type_2_index = np.argsort(self.tmp_exp_type_2[agent][-1])[::-1]
            self.tmp_exp_type_2[agent][-1] = self.tmp_exp_type_2[agent][-1][tmp_exp_type_2_index]
                                    
    
            self.tmp_exp_type_2[agent].append(np.log10((H_all_2[sim-1].diagonal()[np.array(neighbors[-1][agent])])/self.scale_gain))
            self.tmp_exp_type_2[agent][-1] = self.tmp_exp_type_2[agent][-1][tmp_exp_type_2_index]
            self.tmp_exp_type_2[agent].append(0.5 * sum_rate_list_distributed_policy[-1].diagonal()[neighbors[-1][agent]][tmp_exp_type_2_index])
            
            if (len(self.tmp_exp_type_2[agent][-2]) < self.N_neighbors):
                self.tmp_exp_type_2[agent][-1] = np.append(self.tmp_exp_type_2[agent][-1],(self.N_neighbors - len(self.tmp_exp_type_2[agent][-1]))*[self.input_placer])
                self.tmp_exp_type_2[agent][-2] = np.append(self.tmp_exp_type_2[agent][-2],(self.N_neighbors - len(self.tmp_exp_type_2[agent][-2]))*[self.input_placer])
                self.tmp_exp_type_2[agent][-3] = np.append(self.tmp_exp_type_2[agent][-3],(self.N_neighbors - len(self.tmp_exp_type_2[agent][-3]))*[self.input_placer])
            current_experiences[(6 + 4 * self.N_neighbors):(6 + 5 * self.N_neighbors)] = self.tmp_exp_type_2[agent][-3][:self.N_neighbors]
            current_experiences[(6 + 5 * self.N_neighbors):(6 + 6 * self.N_neighbors)] = self.tmp_exp_type_2[agent][-2][:self.N_neighbors]
            current_experiences[(6 + 6 * self.N_neighbors):(6 + 7 * self.N_neighbors)] = self.tmp_exp_type_2[agent][-1][:self.N_neighbors]
        elif(sims_pos_p[agent]>0):
            sim_pos_p = sims_pos_p[agent]
            self.tmp_exp_type_2[agent].append(np.log10(H_all_2[sim_pos_p-1][np.array(neighbors[-1][agent]),agent]/self.prev_suminterferences[neighbors[-1][agent]]))
            tmp_exp_type_2_index = np.argsort(self.tmp_exp_type_2[agent][-1])[::-1]
            self.tmp_exp_type_2[agent][-1] = self.tmp_exp_type_2[agent][-1][tmp_exp_type_2_index]
            self.tmp_exp_type_2[agent].append(np.log10((H_all_2[sim-1].diagonal()[np.array(neighbors[-1][agent])])/self.scale_gain))
            self.tmp_exp_type_2[agent][-1] = self.tmp_exp_type_2[agent][-1][tmp_exp_type_2_index]
            self.tmp_exp_type_2[agent].append(0.5 * sum_rate_list_distributed_policy[-1].diagonal()[neighbors[-1][agent]][tmp_exp_type_2_index])
            if (len(self.tmp_exp_type_2[agent][-2]) < self.N_neighbors):
                self.tmp_exp_type_2[agent][-1] = np.append(self.tmp_exp_type_2[agent][-1],(self.N_neighbors - len(self.tmp_exp_type_2[agent][-1]))*[self.input_placer])
                self.tmp_exp_type_2[agent][-2] = np.append(self.tmp_exp_type_2[agent][-2],(self.N_neighbors - len(self.tmp_exp_type_2[agent][-2]))*[self.input_placer])
                self.tmp_exp_type_2[agent][-3] = np.append(self.tmp_exp_type_2[agent][-3],(self.N_neighbors - len(self.tmp_exp_type_2[agent][-3]))*[self.input_placer])                      
            current_experiences[(6 + 4 * self.N_neighbors):(6 + 5 * self.N_neighbors)] = self.tmp_exp_type_2[agent][-3][:self.N_neighbors]
            current_experiences[(6 + 5 * self.N_neighbors):(6 + 6 * self.N_neighbors)] = self.tmp_exp_type_2[agent][-2][:self.N_neighbors]
            current_experiences[(6 + 6 * self.N_neighbors):(6 + 7 * self.N_neighbors)] = self.tmp_exp_type_2[agent][-1][:self.N_neighbors]  
        return current_experiences
    
