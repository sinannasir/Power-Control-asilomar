# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""

import numpy as np
import project_backend as pb
import json
import argparse



def main(args):

    json_file = args.json_file
    num_sim = args.num_sim
    
    with open ('./config/deployment/'+json_file+'.json','r') as f:
        options = json.load(f)
    
    ## Kumber of samples
    total_samples = options['simulation']['total_samples']
        
    N = options['simulation']['N']
    
    # Kow assume each time slot is 1ms and 
    isTrain = options['simulation']['isTrain']
    if isTrain and num_sim == -1:
        num_simulations = options['simulation']['num_simulations']
        simulation = options['simulation']['simulation_index_start']
    elif isTrain:
        num_simulations = 1
        simulation = num_sim
    else:
        simulation = 0
        num_simulations = 1
    # simulation parameters
    mobility_params = options['mobility_params']
    mobility_params['alpha_angle'] = options['mobility_params']['alpha_angle_rad'] * np.pi #radian/sec
    #Some defaults
    Pmax_dB = 38.0-30
    Pmax = np.power(10.0,Pmax_dB/10)
    n0_dB = -114.0-30
    noise_var = np.power(10.0,n0_dB/10)
    # Hyper aprameters
    
    for overal_sims in range(simulation,simulation+num_simulations):
        if isTrain:
            np.random.seed(50+overal_sims)
        else:
            np.random.seed(1050 + overal_sims + N)
        file_path = './simulations/channel/%s_network%d'%(json_file,overal_sims)
        data = np.load(file_path+'.npz',allow_pickle=True)
        
        H_all = data['arr_1']
        
        weights = []
        for loop in range(total_samples):
            weights.append(np.array(np.ones(N)))    
        # Init Optimizer results
        p_FP_nodelay= []
        time_FP_nodelay = []
        p_WMMSE_nodelay= []
        time_WMMSE_nodelay = []
        
        print('Ideal Case Run sim %d'%(overal_sims))
        print('Run FP sim %d'%(overal_sims))
        (p_FP_nodelay,time_FP_nodelay) = zip(*[pb.FP_algorithm_weighted(N, H, Pmax, noise_var,weight) for (H,weight) in zip(H_all,weights)])
             
        print('Run WMMSE sim %d'%(overal_sims))
        (p_WMMSE_nodelay,time_WMMSE_nodelay) = zip(*[pb.WMMSE_algorithm_weighted(N, H, Pmax, noise_var,weight) for (H,weight) in zip(H_all,weights)])
       
    #    # General simulations
        sum_rate_nodelay = [pb.sumrate_weighted_clipped(H,p,N,noise_var,weight) for (H,p,weight) in zip(H_all,p_FP_nodelay,weights)]
        sum_rate_WMMSE = [pb.sumrate_weighted_clipped(H,p,N,noise_var,weight) for (H,p,weight) in zip(H_all,p_WMMSE_nodelay,weights)]   
        
        # Kow, simulate the process where we use the original FP algorithm
        # Assumption is we ignore the delay at the backhaul network, i.e. there is no delay between the UE and the central controller.
       
        ##################### OTHER BENCHMARKS #####################
        # In this simulation I assume that the central allocator directly uses the most recent channel condition available.
        # Sum rate for the simulation 1
        sum_rate_delayed_central = []
        sum_rate_random = []
        sum_rate_max = []
        # Initial allocation is just random
        p_central = Pmax * np.random.rand(N)
       
        for sim in range (total_samples):
            if (sim > 0):
                p_central = p_FP_nodelay[sim-1]
            sum_rate_delayed_central.append(pb.sumrate_weighted_clipped(H_all[sim],p_central,N,noise_var,weights[sim]))
            sum_rate_random.append(pb.sumrate_weighted_clipped(H_all[sim],Pmax * np.random.rand(N),N,noise_var,weights[sim]))
            sum_rate_max.append(pb.sumrate_weighted_clipped(H_all[sim],Pmax * np.ones(N),N,noise_var,weights[sim]))
    
        np_save_path = './simulations/sumrate/benchmarks/%s_network%d'%(json_file,overal_sims)
        np.savez(np_save_path,p_FP_nodelay,time_FP_nodelay,sum_rate_nodelay,
                 p_WMMSE_nodelay,time_WMMSE_nodelay,sum_rate_WMMSE,
                 sum_rate_delayed_central,sum_rate_random,sum_rate_max)
        print('Saved to %s'%(np_save_path))



if __name__ == "__main__": 

    json_file = "train_K10_N20_shadow10_episode2-5000_travel50000_vmax2_5" 
    json_file = "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" 
    json_file = "train_K10_N20_shadow10_episode10-5000_travel0_fd10" 
    
    
    json_file = "test_N10_K20_shadow10_episode5-2500_travel0_vmax2_5_"
    json_file = "test_N20_K40_shadow10_episode5-2500_travel0_vmax2_5" 
    json_file = "test_N20_K60_shadow10_episode5-2500_travel0_vmax2_5_" 
    json_file = "test_N20_K80_shadow10_episode5-2500_travel0_vmax2_5" 
    json_file = "test_N20_K100_shadow10_episode5-2500_travel0_vmax2_5" 
    json_file = "train_N10_K20_shadow10_episode5-5000_travel20000_vmax2_5"
    json_file = "train_N5_K20_shadow10_episode1-5000_travel0_vmax2_5"  
    
    parser = argparse.ArgumentParser(description='give test scenarios.')
    parser.add_argument('--json-file', type=str, default='train_K10_N20_shadow10_episode5-5000_travel50000_vmax2_5',
                       help='json file for the deployment')
    parser.add_argument('--num-sim', type=int, default=-1,
                       help='If set to -1, it uses num_simulations of the json file. If set to positive, it runs one simulation with the given id.')
    
    args = parser.parse_args()
    main(args)