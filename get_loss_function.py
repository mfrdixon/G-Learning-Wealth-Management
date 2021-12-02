
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import scipy 
from G_learning_portfolio_opt import G_learning_portfolio_opt

def get_loss(trajs,
             num_steps, 
             benchmark_portf, 
             gamma, 
             num_risky_assets, 
             riskfree_rate, 
             expected_risky_returns, 
             Sigma_r, 
             x_vals_init, 
             max_iter_RL, 
             reward_params,
             beta, 
             num_trajs, 
             grad=False, 
             eps=1e-7):
  
  error_tol= 1.e-12
  max_iter_RL = 200
  num_assets= num_risky_assets + 1
  data_xvals = torch.zeros(num_trajs,  num_steps, num_assets, dtype=torch.float64, requires_grad=False)
  data_uvals = torch.zeros(num_trajs,  num_steps, num_assets, dtype=torch.float64, requires_grad=False)
        
  for n in range(num_trajs):
        for t in range(num_steps):
            data_xvals[n,t,:] = torch.tensor(trajs[n][t][0],dtype=torch.float64)
            data_uvals[n,t,:] = torch.tensor(trajs[n][t][1],dtype=torch.float64)
                
                
  # allocate memory for tensors that wil be used to compute the forward pass
  realized_rewards = torch.zeros(num_trajs, num_steps, dtype=torch.float64, requires_grad=False)
  realized_cum_rewards = torch.zeros(num_trajs, dtype=torch.float64, requires_grad=False)

  realized_G_fun = torch.zeros(num_trajs, num_steps, dtype=torch.float64, requires_grad=False)
  realized_F_fun  = torch.zeros(num_trajs,  num_steps, dtype=torch.float64, requires_grad=False)

  realized_G_fun_cum = torch.zeros(num_trajs, dtype=torch.float64, requires_grad=False)  
  realized_F_fun_cum = torch.zeros(num_trajs, dtype=torch.float64, requires_grad=False)  

  reward_params_dict={}
  loss_dict={}
  loss_dict[-1]=np.array([0]*len(reward_params), dtype='float64') # perturb up
  loss_dict[1]=np.array([0]*len(reward_params), dtype='float64') # perturb down
  loss_grad = np.array([0]*len(reward_params), dtype='float64') 

  if grad: # compute gradient
    for j in range(len(reward_params)):
      for k in [-1,1]:
            reward_params_dict[k]=reward_params
            reward_params_dict[k][j]= reward_params_dict[k][j] + k*eps
              
            # 1. create a G-learner
            G_learner = G_learning_portfolio_opt(num_steps,
                                             reward_params_dict[k],    
                                             beta,
                                             benchmark_portf,
                                             gamma,
                                             num_risky_assets,
                                             riskfree_rate,
                                             expected_risky_returns,
                                             Sigma_r,
                                             x_vals_init,
                                             use_for_WM=True)
        
            G_learner.reset_prior_policy()
        
            # run the G-learning recursion to get parameters of G- and F-functions
            G_learner.G_learning(error_tol, max_iter_RL)
        
            # compute the rewards and realized values of G- and F-functions from 
            # all trajectories
            for n in range(num_trajs):
              for t in range(num_steps):
                
                realized_rewards[n,t] = G_learner.compute_reward_on_traj(t,
                                data_xvals[n,t,:], data_uvals[n,t,:])


                realized_G_fun[n,t] = G_learner.compute_G_fun_on_traj(t,
                                data_xvals[n,t,:], data_uvals[n,t,:])


                realized_F_fun[n,t] = G_learner.compute_F_fun_on_traj(t,
                                data_xvals[n,t,:])
                
              realized_cum_rewards[n] = realized_rewards[n,:].sum()
              realized_G_fun_cum[n] = realized_G_fun[n,:].sum()
              realized_F_fun_cum[n] = realized_F_fun[n,:].sum()
          


            
            loss_dict[k][j] = - beta *(realized_G_fun_cum.sum() - realized_F_fun_cum.sum())
      loss_grad[j]=(loss_dict[1][j]-loss_dict[-1][j])/(2.0*eps)
  
  G_learner = G_learning_portfolio_opt(num_steps,
                                      reward_params,
                                      beta,
                                      benchmark_portf,
                                      gamma,
                                      num_risky_assets,
                                      riskfree_rate,
                                      expected_risky_returns,
                                      Sigma_r,
                                      x_vals_init,
                                      use_for_WM=True)
        
  G_learner.reset_prior_policy()
        
  G_learner.G_learning(error_tol, max_iter_RL)
        
  # compute the rewards and realized values of G- and F-functions from 
  # all trajectories
  for n in range(num_trajs):
      for t in range(num_steps):
                
                realized_rewards[n,t] = G_learner.compute_reward_on_traj(t,
                                data_xvals[n,t,:], data_uvals[n,t,:])


                realized_G_fun[n,t] = G_learner.compute_G_fun_on_traj(t,
                                data_xvals[n,t,:], data_uvals[n,t,:])


                realized_F_fun[n,t] = G_learner.compute_F_fun_on_traj(t,
                                data_xvals[n,t,:])
                
      realized_cum_rewards[n] = realized_rewards[n,:].sum()
      realized_G_fun_cum[n] = realized_G_fun[n,:].sum()
      realized_F_fun_cum[n] = realized_F_fun[n,:].sum()
          
    

  loss = - beta *(realized_G_fun_cum.sum() - realized_F_fun_cum.sum())   
  if grad:
    return loss, loss_grad
  else:
    return loss  