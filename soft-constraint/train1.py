#coding:utf-8

# This is a clone and changes from https://github.com/googleinterns/controllabledl /pendulum-system
# of which license is following.
'''Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from __future__ import division
from __future__ import print_function

import sys
import os
import random
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.beta import Beta

from utils_dp1 import DoublePendulum, calc_double_E, verification

from model import RuleEncoder, DataEncoder, Net, NaiveModel, SharedNet, DataonlyNet


model_info = {'ruleonly': {},
              'dataonly': {},
              'dataonly-nobatch': {},
              'dataonly-nobatch-constraint1.0': {'constraint': 1.0},
              'dataonly-nobatch-constraint0.1': {'constraint': 0.1},
              'dataonly-nobatch-constraint0.01': {'constraint': 0.01},
              'ours-shared-test1': {'beta': [0.1], 'scale': 0, 'shared': True},
              'dnn-crr': {'beta': [0.1], 'scale': 1, 'shared': True},
              'dnn-crr-autoscale': {'beta': [0.1], 'scale': 0, 'shared': True}, 
              'dnn-crr-autoscale2': {'beta': [0.1], 'scale': 0, 'shared': False}  # add
             }

### 
### What is 'scale' ?
### task（角度や角速度）と rule（エネルギー）では物理量が違うので、単位が違えば、桁が大小になる。
### Loss関数として　加算するので、対等？値にスケール調整する必要がある。
### When sclae=0 and Not constraint, auto_scaleで、さしあたり、初めのロスの比をそのままスケールとして使っているみたい。

def main():
  parser = ArgumentParser()
  # train/test hyper parameters
  parser.add_argument('--L1', type=float, default=1.0, help='Pendulum rod length 1')
  parser.add_argument('--L2', type=float, default=1.0, help='Pendulum rod length 2')
  parser.add_argument('--M1', type=float, default=1.0, help='Pendulum mass 1')
  parser.add_argument('--M2', type=float, default=5.0, help='Pendulum mass 2')
  parser.add_argument('--F1', type=float, default=0.001, help='Friction coefficient 1')
  parser.add_argument('--F2', type=float, default=0.001, help='Friction coefficient 2')
  
  ### Initial conditions: theta1　1番目の振り子の角度90度から-90度の範囲, dtheta1/dt(=omega1?)速度, theta2　2番目の振り子の角度と速度, dtheta2/dt.
  parser.add_argument('--theta1', type=float, default=0.4725, help='Initial theta 1. If None, it will be sampled from [-np.pi/4, np.pi/4]')
  parser.add_argument('--omega1', type=float, default=0.0, help='Initial omega 1. If None, it will be 0.0')
  parser.add_argument('--theta2', type=float, default=0.3449, help='Initial theta 2. If None, it will be sampled from [-np.pi/4, np.pi/4]')
  parser.add_argument('--omega2', type=float, default=0.0, help='Initial omega 2. If None, it will be 0.0')
  
  
  parser.add_argument('--delta_t', type=float, default=0.005, help='Delta t used to generate a sequence')
  parser.add_argument('--tmax', type=int, default=3000, help='Max timesteps')
  parser.add_argument('--sampling_step', type=int, default=20, help='Sampling timesteps')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--batch_size', type=int, default=64, help='default: 64')
  parser.add_argument('--model_type', type=str, default='dnn-crr-autoscale', help='dataonly, ours, ruleonly. default:dataonly')  ### chg
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--input_dim_encoder', type=int, default=16)
  parser.add_argument('--output_dim_encoder', type=int, default=64)
  parser.add_argument('--hidden_dim_encoder', type=int, default=64)
  parser.add_argument('--hidden_dim_db', type=int, default=64)
  parser.add_argument('--n_layers', type=int, default=2)
  parser.add_argument('--epochnum', type=int, default=1000, help='default: 1000')
  parser.add_argument('--early_stopping_thld', type=int, default=10, help='default: 10')
  parser.add_argument('--valid_freq', type=int, default=5, help='default: 5')
  parser.add_argument('--noreload', action='store_true', help='previous model is not reloaded if specified')  ### add
  parser.add_argument('--noreload2', action='store_true', help='previous dp t,y is not reloaded if specified')  ### add
  
  args = parser.parse_args()
  print(args)


  ### chg
  # device = args.device
  cuda = torch.cuda.is_available()
  device = torch.device("cuda" if cuda else "cpu")
  
  seed = args.seed

  # Generate DP sequence
  print("Generate Double-pendulum sequence...")
  L1, L2, M1, M2, F1, F2 = args.L1, args.L2, args.M1, args.M2, args.F1, args.F2
  init_theta1 = round(args.theta1, 4) if args.theta1 is not None else round(np.random.uniform(-np.pi/4, np.pi/4), 4)
  init_omega1 = round(args.omega1, 4) if args.omega1 is not None else 0.0
  init_theta2 = round(args.theta2, 4) if args.theta2 is not None else round(np.random.uniform(-np.pi/4, np.pi/4), 4)
  init_omega2 = round(args.omega2, 4) if args.omega2 is not None else 0.0

  tmax, dt = args.tmax, args.delta_t
  dp = DoublePendulum(L1, L2, M1, M2, init_theta1, init_omega1, init_theta2, init_omega2, F1, F2)
  
  
  ### add
  dp_filename='dp_dataset'
  dp_filename2=dp_filename+'.npz'
  if not args.noreload2 and os.path.exists(dp_filename2):
    dataset0=np.load(dp_filename2)
    t=dataset0['arr_0']
    y=dataset0['arr_1']
    print ('re-load t y from ' + dp_filename2)
  else:
    t, y = dp.generate(tmax=tmax, dt=dt)
    np.savez(dp_filename,t, y)


  dp_params = {'M1': dp.M1, 'M2': dp.M2, 'L1': dp.L1, 'L2': dp.L2, 'g': dp.g, 'F1': dp.F1, 'F2': dp.F2}
  print('sequence length: {} ({} sec)'.format(len(y), tmax))
  print('dt: {} (sec)\n'.format(dt))

  subsampling = True
  if subsampling:
    # Fine dt for generation and subsample for learning
    sampling_step = args.sampling_step    # sample a row for every the step.
    sampling_dt = dt*sampling_step
    sampling_ind = np.arange(0, t.shape[0] - 1, sampling_step)
    sampling_t = t[sampling_ind]

    input_output_y = np.concatenate((y[:-1], y[1:]), axis=1)    # [[input, output]]
    X = input_output_y[sampling_ind]
  
  else:
    sampling_dt = dt
    sampling_t = t

    input_output_y = np.concatenate((y[:-1], y[1:]), axis=1)    # [[input, output]]
    X = input_output_y

  X_np = np.array(X)
  print('subsampled sequence length: {} ({} sec)'.format(len(X), tmax))
  print('sampling dt: {} (sec)'.format(sampling_dt))

  # Data preprocessing
  X = torch.tensor(X_np, dtype=torch.float32, device=device)
  num_samples = X.shape[0]
  input_dim = X.shape[1]//2    # (theta1, omega1, theta2, omega2)
  print ('input_dim', input_dim) ### add

  # 60:10:30 split
  train_X, train_y = X[:int(num_samples*0.6), :input_dim], X[:int(num_samples*0.6), input_dim:]
  valid_X, valid_y = X[int(num_samples*0.6):int(num_samples*0.7), :input_dim], X[int(num_samples*0.6):int(num_samples*0.7), input_dim:]
  test_X, test_y = X[int(num_samples*0.7):, :input_dim], X[int(num_samples*0.7):, input_dim:]

  total_train_sample = len(train_X)
  total_valid_sample = len(valid_X)
  total_test_sample = len(test_X)

  batch_size = args.batch_size
  train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=valid_X.shape[0])
  test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=test_X.shape[0])

  print("data size: {}/{}/{}".format(len(train_X), len(valid_X), len(test_X)))

  # Start
  model_type = args.model_type
  if model_type not in model_info:
    lr = 0.001
    shared = False
    constraint = 0.0
    scale = 1.0
    beta_param = [1.0]
    alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
    model_params = {}

  else:
    model_params = model_info[model_type]
    lr = model_params['lr'] if 'lr' in model_params else 0.001
    shared = model_params['shared'] if 'shared' in model_params else False
    constraint = model_params['constraint'] if 'constraint' in model_params else 0.0
    scale = model_params['scale'] if 'scale' in model_params else 1.0
    beta_param = model_params['beta'] if 'beta' in model_params else [1.0]
    if len(beta_param) == 1:
      alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
    elif len(beta_param) == 2:
      alpha_distribution = Beta(float(beta_param[0]), float(beta_param[1]))

  print('model_type: {}\tscale:{}\tBeta distribution: Beta({})\tlr: {}, constraint: {}, seed: {}'
        .format(model_type, scale, beta_param, lr, constraint, seed))

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  SKIP = True    # Delta value (x(t+1)-x(t)) prediction if True else absolute value (x(t+1)) prediction
  if model_type.startswith('dataonly'):
    merge = 'cat'
  elif model_type.startswith('ruleonly'):
    merge = 'cat'
  elif model_type.startswith('ours') or model_type.startswith('dnn-crr'):
    merge = 'cat'

#   input_dim = 4
  input_dim_encoder = args.input_dim_encoder    # default=16
  output_dim_encoder = args.output_dim_encoder  # default=64
  hidden_dim_encoder = args.hidden_dim_encoder  # default=64
  hidden_dim_db = args.hidden_dim_db            # default=64
  output_dim = input_dim
  n_layers = args.n_layers

  ### What is "shared" ?  SharedNet   task_encoder, rule_encoderの前処理として共通利用されているSharedNetが加わったもの
  if shared:
    rule_encoder = RuleEncoder(input_dim_encoder, output_dim_encoder, hidden_dim=hidden_dim_encoder)
    data_encoder = DataEncoder(input_dim_encoder, output_dim_encoder, hidden_dim=hidden_dim_encoder)
    #                  ↓　あくまで外からの入出の次元は、input_dim = 4
    model = SharedNet(input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge, skip=SKIP).to(device)    # delta value prediction
  else:  ### normal net...
    rule_encoder = RuleEncoder(input_dim, output_dim_encoder, hidden_dim=hidden_dim_encoder)
    data_encoder = DataEncoder(input_dim, output_dim_encoder, hidden_dim=hidden_dim_encoder)
    if model_type.startswith('dataonly'):
      model = DataonlyNet(input_dim, output_dim, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, skip=SKIP).to(device)
    else:
      model = Net(input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge, skip=SKIP).to(device)    # delta value prediction

  total_params = sum(p.numel() for p in model.parameters())
  print("total parameters: {}".format(total_params))

  loss_rule_func = lambda x,y: torch.mean(F.relu(x-y))    # if x>y, penalize it.
  loss_task_func = nn.L1Loss()    # return scalar (reduction=mean)
  l1_func = nn.L1Loss()
  best_val_loss = float('inf')
  optimizer = optim.Adam(model.parameters(), lr=lr)

  epochnum = args.epochnum
  early_stopping_thld = args.early_stopping_thld
  counter_early_stopping = 1
  valid_freq = args.valid_freq
  saved_filename = 'dp-{}_{:.4f}_{:.1f}_{:.4f}_{:.1f}-seed{}.skip.demo.pt' \
                          .format(model_type, init_theta1, init_omega1, init_theta2, init_omega2, seed)

  saved_filename =  os.path.join('saved_models', saved_filename)
  ### add
  save_dir='saved_models'
  if not os.path.exists(save_dir):
      os.mkdir(save_dir)
  
  print('saved_filename: {}\n'.format(saved_filename))
  
  ### add
  reload_file = saved_filename
  if not args.noreload and os.path.exists(reload_file):
      state = torch.load(reload_file)
      print("Reloading model at epoch {}"
            ", with test error {}".format(
                state['epoch'],
                state['loss']))
      model.load_state_dict(state['model_state_dict'])
      optimizer.load_state_dict(state['optimizer_state_dict'])


  # Training
  for epoch in range(1, epochnum+1):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
      batch_train_x = batch_data[0] + 0.01*torch.randn(batch_data[0].shape).to(device)    # Adding noise
      batch_train_y = batch_data[1]

      optimizer.zero_grad()

      if model_type.startswith('dataonly'):
        alpha = 0.0
      elif model_type.startswith('ruleonly'):
        alpha = 1.0
      elif model_type.startswith('ours') or model_type.startswith('dnn-crr'):
        alpha = alpha_distribution.sample().item()

      output = model(batch_train_x, alpha=alpha)

      _, _, curr_E = calc_double_E(batch_train_x, **dp_params)    # E(X_t)    Energy of X_t (Current energy)
      _, _, next_E = calc_double_E(batch_train_y, **dp_params)    # E(X_{t+1})    Energy of X_{t+1} (Next energy from ground truth)
      _, _, pred_E = calc_double_E(output, **dp_params)    # E(\hat{X}_t+1)    Energy of \hat{X}_{t+1} (Next energy from prediction)

      loss_task = loss_task_func(output, batch_train_y)    # state prediction
      loss_rule = loss_rule_func(pred_E, curr_E)    # energy damping by friction: E_{t+1}<=E_t
      loss_mae = l1_func(output, batch_train_y).item()

      if constraint:
        loss = loss_task + constraint*loss_rule    # Constrained baseline　ロスに束縛条件の項を加えたもの。constraintはその重み係数
      else:
        if scale == 0:
          scale = loss_rule.item() / loss_task.item()
          print('scale is updated: {}'.format(scale))  # scaleはruleロス とtaskロスの比。はじめ０のときの1回しか行わない？
        loss = alpha * loss_rule + scale * (1-alpha) * loss_task

      loss.backward()
      optimizer.step()

    # Evaluate on validation set
    if epoch % valid_freq == 0:
      model.eval()
      with torch.no_grad():
        val_loss_task = 0
        val_loss_rule = 0
        val_ratio = 0
        for val_x, val_y in valid_loader:
          val_x += 0.01*torch.randn(val_x.shape).to(device)  # 入力に小さいランダム・ノイズを加算して、微分可能性にしている？
          output = model(val_x, alpha=0.0)  # alpha 0のときだけ計算しているよ！
          _, _, curr_E = calc_double_E(val_x, **dp_params)
          _, _, pred_E = calc_double_E(output, **dp_params)

          val_loss_task += (loss_task_func(output, val_y).item() * val_x.shape[0] / total_valid_sample)
          val_loss_rule += (loss_rule_func(pred_E, curr_E) * val_x.shape[0] / total_valid_sample)
          val_ratio += (verification(curr_E, pred_E, threshold=0.0).item() * val_x.shape[0] / total_valid_sample)
          ### var_ratio  verificationは（エネルギー減衰の）ルールを守っているものの比率。
          ### taskそのものは　次の（2重振り子の２つの振り子の）角度、角速度の予測

        if val_loss_task < best_val_loss:
          counter_early_stopping = 1
          best_val_loss = val_loss_task
          print('[Valid] Epoch: {} Loss(Task): {:.6f} Loss(Rule): {:.6f}  Ratio(Rule): {:.3f} (alpha: 0.0)\t best model is updated %%%%'
                      .format(epoch, best_val_loss, val_loss_rule, val_ratio))
          torch.save({
                    'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss
          }, saved_filename)
        else:
          print('[Valid] Epoch: {} Loss(Task): {:.6f} Loss(Rule): {:.6f} Ratio(Rule): {:.3f} (alpha: 0.0) ({}/{})'
                      .format(epoch, val_loss_task, val_loss_rule, val_ratio, counter_early_stopping, early_stopping_thld))
          if counter_early_stopping >= early_stopping_thld:
            break
          else:
            counter_early_stopping += 1

  # Test
  if shared:
    rule_encoder = RuleEncoder(input_dim_encoder, output_dim_encoder, hidden_dim=hidden_dim_encoder)
    data_encoder = DataEncoder(input_dim_encoder, output_dim_encoder, hidden_dim=hidden_dim_encoder)
    model_eval = SharedNet(input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge, skip=SKIP).to(device)    # delta value prediction
  else:
    rule_encoder = RuleEncoder(input_dim, output_dim_encoder, hidden_dim=hidden_dim_encoder)
    data_encoder = DataEncoder(input_dim, output_dim_encoder, hidden_dim=hidden_dim_encoder)
    if model_type.startswith('dataonly'):
      model_eval = DataonlyNet(input_dim, output_dim, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, skip=SKIP).to(device)
    else:
      model_eval = Net(input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge, skip=SKIP).to(device)    # delta value prediction

  checkpoint = torch.load(saved_filename)
  model_eval.load_state_dict(checkpoint['model_state_dict'])
  print("best model loss: {:.6f}\t at epoch: {}".format(checkpoint['loss'], checkpoint['epoch']))

  model_eval.eval()
  with torch.no_grad():
    test_loss_task = 0
    for test_x, test_y in test_loader:
      output = model_eval(test_x, alpha=0.0)
      test_loss_task += (loss_task_func(output, test_y).item() * test_x.shape[0] / total_test_sample)  # sum up batch loss

  print('\nTest set: Average loss: {:.8f}\n'.format(test_loss_task))

  # Best model
  alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
  for alpha in alphas:
    model_eval.eval()
    with torch.no_grad():
      test_loss_task, test_ratio = 0, 0
      for test_x, test_y in test_loader:

        if model_type.startswith('dataonly'):
          output = model_eval(test_x, alpha=0.0)
        elif model_type.startswith('ruleonly'):
          output = model_eval(test_x, alpha=1.0)
        elif model_type.startswith('ours') or model_type.startswith('dnn-crr'):
          output = model_eval(test_x, alpha=alpha)

        test_loss_task += (loss_task_func(output, test_y).item() * test_x.shape[0] / total_test_sample)  # sum up batch loss

        _, _, curr_E = calc_double_E(test_x, **dp_params)
        _, _, next_E = calc_double_E(test_y, **dp_params)
        _, _, pred_E = calc_double_E(output, **dp_params)

        test_ratio += (verification(curr_E, pred_E, threshold=0.0).item() * test_x.shape[0] / total_test_sample)

      print('Test set: Average loss: {:.8f} (alpha:{})'.format(test_loss_task, alpha))
      print("ratio of verified predictions: {:.6f} (alpha:{})".format(test_ratio, alpha))


if __name__ == '__main__':
  main()
