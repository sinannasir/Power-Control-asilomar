# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""
import os
import numpy as np
#import matplotlib.pyplot as plt
import project_backend as pb
import time
import collections
import json
import DDPG

import argparse


def main(args):
    
    json_file = args.json_file
    json_file_policy = args.json_file_policy
    num_sim = args.num_sim
    
    with open ('./config/deployment/'+json_file+'.json','r') as f:
        options = json.load(f)
    with open ('./config/policy/'+json_file_policy+'.json','r') as f:
        options_policy = json.load(f)
        
    if not options_policy['cuda']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    
    ## Kumber of samples
    total_samples = options['simulation']['total_samples']
        
    N = options['simulation']['N']
    
    
    if num_sim == -1:
        num_simulations = options['simulation']['num_simulations']
        simulation = options['simulation']['simulation_index_start']
    else:
        num_simulations = 1
        simulation = num_sim
    
    # simulation parameters
    train_episodes = options['train_episodes']
    mobility_params = options['mobility_params']
    mobility_params['alpha_angle'] = options['mobility_params']['alpha_angle_rad'] * np.pi #radian/sec
    #Some defaults
    Pmax_dB = 38.0-30
    Pmax = np.power(10.0,Pmax_dB/10)
    n0_dB = -114.0-30
    noise_var = np.power(10.0,n0_dB/10)
    # Hyper aprameters
    N_neighbors = options_policy['N_neighbors']
    neightresh = noise_var*options_policy['neightresh']
    
    for overal_sims in range(simulation,simulation+num_simulations):
        tf.reset_default_graph()
        tf.set_random_seed(100+overal_sims)
        np.random.seed(100+overal_sims)
    
        file_path = './simulations/channel/%s_network%d'%(json_file,overal_sims)
        data = np.load(file_path+'.npz',allow_pickle=True)
        
        H_all = data['arr_1']
        H_all_2 = []
        for i in range(total_samples):
            H_all_2.append(H_all[i]**2)
        
        weights = []
        for loop in range(total_samples):
            weights.append(np.array(np.ones(N)))
        
        time_calculating_strategy_takes = []
            
        # Virtual neighbor placer
        neighbors_in = collections.deque([],2)
        neighbors = collections.deque([],2)
    
        sims_pos_p = np.zeros(N).astype(int) - 1
    
        policy = DDPG.DDPG(options,options_policy,N,Pmax,noise_var)
       
       
        # Start the simulation 2
        # Sum rate for the simulation 1
        sum_rate_distributed_policy = []
        sum_rate_list_distributed_policy = collections.deque([],2)
        # Initial allocation is just random
        p_central = Pmax * np.random.rand(N)
        p_strategy = np.array(p_central) # strategy is a completely different object
        p_strategy_current = np.array(p_strategy)
       
        time_calculating_strategy_takes = []
        time_optimization_at_each_slot_takes = []
       
        p_strategy_all=[]
    
        with tf.Session() as sess:
            sess.run(policy.init)
            policy.initialize_critic_updates(sess) 
            policy.initialize_actor_updates(sess) 
            # Start iterating voer time slots
            for sim in range (total_samples):
                policy.check_memory_restart(sess,sim)       
                policy.update_handler(sess,sim)
                # save an instance per training episode for testing purposes.
                if(sim %train_episodes['T_train'] == 0):
                    model_destination = ('./simulations/sumrate/policy/%s_%s_network%d_episode%d.ckpt'%(
                            json_file,json_file_policy,overal_sims,int(float(sim)/train_episodes['T_train']))).replace('[','').replace(']','')
                    policy.save(sess,model_destination)
        
                # If at least one time slot passed to get experience
                if (sim %train_episodes['T_train'] > 1):                    
                    # Each agent picks its strategy.
                    for agent in range (N):
                        current_local_state = policy.local_state(sim,agent,p_strategy_all,H_all_2,neighbors,neighbors_in,sum_rate_list_distributed_policy,sims_pos_p) 
                        a_time = time.time()  
                        strategy = policy.act(sess,current_local_state,sim,agent)
                        time_calculating_strategy_takes.append(time.time()-a_time)
                        
                        if (sim %train_episodes['T_train'] > 2): # Koew, There is prev state to form experience.
                            sorted_neighbors_criteria = np.log10(H_all_2[sim-1][np.array(neighbors[-1][agent]),agent]/policy.prev_suminterferences[neighbors[-1][agent]])
                            sorted_neighbors = neighbors[-1][agent][np.argsort(sorted_neighbors_criteria)[::-1]]
                            if len(sorted_neighbors)>N_neighbors:
                                sorted_neighbors = sorted_neighbors[:N_neighbors]
                            sorted_neighbors = np.append(sorted_neighbors,agent)
                            current_reward = np.sum(np.multiply(weights[sim-1],sum_rate_list_distributed_policy[-1][:,agent])[sorted_neighbors])
                            policy.remember(agent,current_local_state,current_reward)
                            
                        # Only train it once per timeslot agent == 0 ensures that
                        if agent == (N-1): # If there is enough data to create a mini batch
                            a_time = time.time()
                            
                            # TRAIK for a minibatch
                            policy.train(sess,sim)
                            
                            time_optimization_at_each_slot_takes.append(time.time()-a_time)
                                
                        # Pick the action
                        p_strategy[agent] = policy.Pmax * strategy #** 10
    
                        # Add current state to the short term memory to observe it during the next state
                        policy.previous_state[agent,:] = current_local_state
                        policy.previous_action[agent] = strategy
    
                if(sim %train_episodes['T_train'] < 2):
                    p_strategy = np.random.rand(N)
                p_strategy_current = np.array(p_strategy)
                policy.prev_suminterferences = np.matmul(H_all_2[sim],p_strategy) - (H_all_2[sim].diagonal()*p_strategy) + noise_var
                sims_pos_p[np.where(p_strategy_current>0)] = sim
    
                tmp_neighbors_in = []
                tmp_neighbors = []
                for nei_i in range(N):
                    neigh_tmp_variab = np.where((H_all[sim][nei_i,:]**2)*p_strategy_current>neightresh)
                    neigh_tmp_variab = np.delete(neigh_tmp_variab,np.where(neigh_tmp_variab[0]==nei_i))
                    tmp_neighbors_in.append(neigh_tmp_variab)
    
                for nei_i in range(N):
                    tmp_neighlist = []
                    for nei_j in range(N):
                        if(len(np.where(tmp_neighbors_in[nei_j]==nei_i)[0]) != 0):
                            tmp_neighlist.append(nei_j)
                    if (len(tmp_neighlist) == 0 and len(neighbors) >0):
                        tmp_neighbors.append(np.array(neighbors[-1][nei_i]))
                    else:
                        tmp_neighbors.append(np.array(tmp_neighlist))
                neighbors.append(tmp_neighbors)
                neighbors_in.append(tmp_neighbors_in)
                # all sumrates in a list
                sum_rate_list_distributed_policy.append(pb.reward_helper(H_all[sim],p_strategy,N,noise_var,Pmax,neighbors_in[-1]))
    
                sum_rate_distributed_policy.append(pb.sumrate_weighted_clipped(H_all[sim],p_strategy,N,noise_var,weights[sim]))
                p_strategy_all.append(np.array(p_strategy))
                if(sim%2500 == 0):
                    print('Time %d sim %d'%(sim,overal_sims))
           
            policy.equalize(sess)
            print('Train is over sim %d'%(overal_sims))
    
            model_destination = ('./simulations/sumrate/policy/%s_%s_network%d_episode%d.ckpt'%(
                    json_file,json_file_policy,overal_sims,int(float(total_samples)/train_episodes['T_train']))).replace('[','').replace(']','')
            policy.save(sess,model_destination)
               
        # End Train Phase
        np_save_path = './simulations/sumrate/train/%s_%s_network%d.ckpt'%(json_file,json_file_policy,overal_sims)
        print(np_save_path)
        np.savez(np_save_path,options,options_policy,sum_rate_distributed_policy,p_strategy_all,
                 time_optimization_at_each_slot_takes,time_calculating_strategy_takes)
    
if __name__ == "__main__":     
    parser = argparse.ArgumentParser(description='give test scenarios.')
    parser.add_argument('--json-file', type=str, default='train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5',
                       help='json file for the deployment')
    parser.add_argument('--json-file-policy', type=str, default='ddpg200_100_50',
                       help='json file for the hyperparameters')
    parser.add_argument('--num-sim', type=int, default=-1,
                       help='If set to -1, it uses num_simulations of the json file. If set to positive, it runs one simulation with the given id.')
    
    args = parser.parse_args()
    main(args)
    