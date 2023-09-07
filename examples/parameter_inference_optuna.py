# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This example demonstrates how to leverage the differentiability of DiSECt to
infer the simulation parameters of our simulator to reduce the gap between
the simulated knife force profile and a ground-truth force profile.

The optimization progress is logged in tensorboard, which can be viewed by
running the following command:
    tensorboard --logdir=log

Note: this example currently allocates about 6 GB of GPU memory, which we aim to
reduce in further updates of our simulator.
"""

# fmt: off
import torch
torch.autograd.set_detect_anomaly(True)
import os
from datetime import datetime
import pytz
import pickle

from tensorboardX import SummaryWriter

from disect.cutting import load_settings, save_settings, optuna_trainer, adam_trainer, create_sim
# fmt: on

settings = load_settings("examples/config/cooking/ansys_cucumber.json")

# settings.sim_duration = 0.4
settings.sim_dt = 2e-5
settings.initial_y = 0.059/2. + settings.veggie_height + 0.001  # center of knife + actual desired height
settings.velocity_y = -0.020
now = datetime.now(pytz.timezone('Asia/Tokyo'))
experiment_name = f"{now.strftime('%Y%m%d-%H%M')}_optuna_{settings.veggie}_param_inference_dt{settings.sim_dt}"
logger = SummaryWriter(logdir=f"log/{experiment_name}")
device = 'cuda'
optuna_results = None
best_params = None

optuna_results = '/root/o2ac-ur/disect/log/20230907-1049_optuna_cucumber_param_inference_dt2e-05/best_optuna_optimized_tensors.pkl'
best_params = pickle.load(open(optuna_results, 'rb'))
print("best params", best_params)

save_settings(settings, f"log/{experiment_name}/settings.json")

os.makedirs(f"log/{experiment_name}/plots")
os.makedirs(f"log/{experiment_name}/params")
###########################
### Optuna Optimization ###
###########################

# sim, parameters = create_sim(settings, experiment_name, requires_grad=False, best_params=best_params, device=device, verbose=False)
# best_params = optuna_trainer(sim, parameters, logger, n_trials=300)

# # Display the information regarding the best trial
# print("Best trial:", best_params)

# optuna_results = f"log/{experiment_name}/best_optuna_optimized_tensors.pkl"
# pickle.dump(best_params, open(optuna_results, "wb"))

#########################
### Adam Optimization ###
#########################

learning_rate = 1.0
settings.sim_dt = 1e-5 # smaller dt is less likely to crash (Nans)


sim, parameters = create_sim(settings, experiment_name, requires_grad=True, best_params=best_params, device=device, verbose=True)

adam_trainer(sim, logger, learning_rate, iterations=100, previous_best=optuna_results)
