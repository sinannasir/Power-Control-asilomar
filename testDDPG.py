# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""

import numpy as np
import project_backend as pb
import time
import collections
import json
import DDPG
import copy
import os
import argparse

def main(args):
    
    json_file = args.json_file
    json_files_train = args.json_files_train
    json_file_policy_train = args.json_file_policy_train
    
    with open ('./config/deployment/'+json_file+'.json','r') as f:
        options = json.load(f)
    with open ('./config/policy/'+json_file_policy_train+'.json','r') as f:
        options_policy = json.load(f)
    if not options_policy['cuda']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    
    for json_file_train in json_files_train:
        with open ('./config/deployment/'+json_file_train+'.json','r') as f:
            options_train = json.load(f)
        included_train_episodes = []
        tot_train_episodes = int(options_train['simulation']['total_samples']/options_train['train_episodes']['T_train'])
        N = options['simulation']['N']
        if N <=20:
            for i in range(tot_train_episodes+1):
                if i<=15 or i%5==0:
                    included_train_episodes.append(i)
        else:
            included_train_episodes.append(tot_train_episodes)
        
        train_tot_simulations = options_train['simulation']['num_simulations']
        tot_test_episodes = int(options['simulation']['total_samples']/options['train_episodes']['T_train'])
        inner_train_networks = [[]]*tot_test_episodes
        for i in range(tot_test_episodes):
            if options['simulation']['test_include'] == 'all':
                inner_train_networks[i] = 0
            else:
                inner_train_networks[i] = list(np.random.randint(0,train_tot_simulations,options['simulation']['test_include']))
        ## Number of samples
        total_samples = options['simulation']['total_samples']
        
        
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
        neightresh = noise_var*options_policy['neightresh']        
        i_train = -1
        
        for ep in included_train_episodes:
            #
            np.random.seed(500 + N + ep)
            # i_train = np.random.randint(train_tot_simulations)
            i_train+=1
            i_train = i_train % train_tot_simulations
            
            file_path = './simulations/channel/%s_network%d'%(json_file,0)
            data = np.load(file_path+'.npz')
            
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
           
            time_calculating_strategy_takes = []
            time_optimization_at_each_slot_takes = []
            sum_rate_distributed_policy_episode = []
            p_strategy_all_apisode = []
            
            sum_rate_distributed_policy = []
            sum_rate_list_distributed_policy = collections.deque([],2)
            # Initial allocation is just random
            p_central = Pmax * np.random.rand(N)
            p_strategy = np.array(p_central) # strategy is a completely different object
            p_strategy_current = np.array(p_strategy)
                      
            p_strategy_all=[]
        
            with tf.Session() as sess:
                sess.run(policy.init)
                policy.initialize_critic_updates(sess) 
                policy.initialize_actor_updates(sess) 
                # Start iterating voer time slots
                for sim in range (total_samples):
                    # save an instance per training episode for testing purposes.
                    if(sim %train_episodes['T_train'] == 0):
                        train_network_idx = i_train
                        model_destination = ('./simulations/sumrate/policy/%s_%s_network%d_episode%d.ckpt'%(
                                json_file_train,json_file_policy_train,train_network_idx,ep)).replace('[','').replace(']','')
                        policy.load(sess,model_destination)
                        i_train+=1
                        i_train = i_train % train_tot_simulations
            
                    # If at least one time slot passed to get experience
                    if (sim %train_episodes['T_train'] > 1):                    
                        # Each agent picks its strategy.
                        for agent in range (N):
                            current_local_state = policy.local_state(sim,agent,p_strategy_all,H_all_2,neighbors,neighbors_in,sum_rate_list_distributed_policy,sims_pos_p) 
                            a_time = time.time()  
                            strategy = policy.act_noepsilon(sess,current_local_state,sim)
                            time_calculating_strategy_takes.append(time.time()-a_time)
                                    
                            # Pick the action
                            p_strategy[agent] = policy.Pmax * strategy
        
                            # Add current state to the short term memory to observe it during the next state
                            policy.previous_state[agent,:] = current_local_state
                            policy.previous_action[agent] = strategy
        
                    if(sim %train_episodes['T_train'] < 2):
                        p_strategy = Pmax * np.ones(N)
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
                        print('Test Time %d episode %d'%(sim, ep))
            sum_rate_distributed_policy_episode.append(copy.copy(sum_rate_distributed_policy))
            p_strategy_all_apisode.append(copy.copy(p_strategy_all))
                   
            # End Train Phase
            np_save_path = './simulations/sumrate/test/%s_%s_%s_episode%d.ckpt'%(json_file,json_file_train,json_file_policy_train,ep)
            print('Saved to %s'%(np_save_path))
            np.savez(np_save_path,options,options_policy,sum_rate_distributed_policy_episode,p_strategy_all_apisode,
                     time_optimization_at_each_slot_takes,time_calculating_strategy_takes,included_train_episodes,inner_train_networks)
    
if __name__ == "__main__":     
    parser = argparse.ArgumentParser(description='give test scenarios.')
    parser.add_argument('--json-file', type=str, default='test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5',
                       help='json file for the deployment')
    parser.add_argument('--json-files-train', nargs='+', default=["train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5",
                        "train_K10_N20_shadow10_episode10-5000_travel0_fd10"],
                       help='json files train.')
    parser.add_argument('--json-file-policy-train', type=str, default='ddpg200_100_50',
                       help='json file for the hyperparameters')
    
    args = parser.parse_args()
    print(args.json_files_train)
    main(args)