# Software License Agreement (BSD License)
#
# Copyright (c) 2023, OMRON SINIC X
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of OMRON SINIC X nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Cristian C. Beltran-Hernandez, Nicolas Erbetti

import torch
torch.autograd.set_detect_anomaly(False, check_nan=True)
torch.set_printoptions(threshold=10000)

import os
from datetime import datetime
import pytz
import pickle

from tensorboardX import SummaryWriter

from disect.cutting import load_settings, save_settings, optuna_trainer, adam_trainer, create_sim

settings = load_settings("examples/config/cooking/ansys_cucumber.json")

# settings.sim_duration = 1.3
settings.sim_dt = 4e-5
settings.initial_y = 0.059/2. + settings.veggie_height + 0.001  # center of knife + actual desired height
settings.velocity_y = -0.020
now = datetime.now(pytz.timezone('Asia/Tokyo'))
experiment_name = f"{now.strftime('%Y%m%d-%H%M%S')}_optuna_{settings.veggie}_param_inference_dt{settings.sim_dt}"
logger = SummaryWriter(logdir=f"log/{experiment_name}")
device = 'cuda'
optuna_results = None
best_params = None

optuna_opt = True

# optuna_results = '/root/o2ac-ur/disect/log/best_results/20230908-131328_optuna_cucumber_param_inference_dt3e-05/best_optuna_optimized_tensors.pkl'
# best_params = pickle.load(open(optuna_results, 'rb'))
# print("best params", best_params)

save_settings(settings, f"log/{experiment_name}/settings.json")

os.makedirs(f"log/{experiment_name}/plots")
os.makedirs(f"log/{experiment_name}/params")
###########################
### Optuna Optimization ###
###########################
if optuna_opt:
    sim, parameters = create_sim(settings, experiment_name, requires_grad=False,
                                best_params=best_params, device=device, verbose=False, shared_params=True)
    best_params = optuna_trainer(sim, parameters, logger, n_trials=500)

    # Display the information regarding the best trial
    print("Best trial:", best_params)

    optuna_results = f"log/{experiment_name}/best_optuna_optimized_tensors.pkl"
    pickle.dump(best_params, open(optuna_results, "wb"))

#########################
### Adam Optimization ###
#########################

learning_rate = 0.5
settings.sim_dt = 2e-5  # smaller dt is less likely to crash (Nans)

sim, parameters = create_sim(settings, experiment_name, requires_grad=True, best_params=best_params, device=device, verbose=True)

adam_trainer(sim, logger, learning_rate, iterations=100, previous_best=optuna_results)
