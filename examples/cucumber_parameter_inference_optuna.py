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
import tqdm
import sys
import os
import optuna
from datetime import datetime
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from disect.cutting import CuttingSim
from disect.cutting import load_settings, ConstantLinearVelocityMotion, Parameter
# fmt: on

parameters = {
    "initial_y": Parameter("initial_y", 0.055, 0.05, 0.06),
    "cut_spring_ke": Parameter("cut_spring_ke", 200, 100, 8000),
    "cut_spring_softness": Parameter("cut_spring_softness", 200, 10, 5000),
    "sdf_ke": Parameter("sdf_ke", 500, 200., 8000, individual=True),
    "sdf_kd": Parameter("sdf_kd", 1., 0.1, 100.),
    "sdf_kf": Parameter("sdf_kf", 0.01, 0.001, 8000.0),
    "sdf_mu": Parameter("sdf_mu", 0.5, 0.45, 1.0),
}

settings = load_settings("examples/config/cooking/ansys_cucumber.json")
settings.sim_duration = 1.2
settings.sim_dt = 1e-5
settings.initial_y = 0.03 + 0.025  # center of knife + actual desired height
settings.velocity_y = -0.020
device = "cuda"
learning_rate = 0.01

now = datetime.now()
experiment_name = f"optuna_param_inference_dt{settings.sim_dt}_{now.strftime('%Y%m%d-%H%M')}"
logger = SummaryWriter(logdir=f"log/{experiment_name}")

requires_grad = False

sim = CuttingSim(settings, experiment_name=experiment_name, adapter=device, requires_grad=requires_grad,
                 parameters=parameters, verbose=False)
sim.motion = ConstantLinearVelocityMotion(
    initial_pos=torch.tensor([0.0, settings.initial_y, 0.0], device=device),
    linear_velocity=torch.tensor([0.0, settings.velocity_y, 0.0], device=device))

sim.cut()

sim.load_groundtruth('osx_dataset/calibrated/cucumber_3_05.npy', groundtruth_dt=0.002)

###########################
### Optuna Optimization ###
###########################

optuna_trials = 200

# # Optuna training
def objective(trial):

    # Initialization
    print(f'\n### {experiment_name}  --  trial.number {trial.number}')

    # Creating the optuna dict
    suggestion = {}
    for name, param in parameters.items():
        suggestion.update({name: trial.suggest_float(name, param.low, param.high)})       

    optimized_tensors = {}
    for key, value in suggestion.items():
        optimized_tensors.update({key: torch.tensor([value]*sim.number_of_cut_spring)})
    optimized_tensors.update({'initial_y': torch.tensor(suggestion['initial_y'])})

    sim.init_parameters(optimized_tensors)

    # Computing the optuna_loss function
    hist_knife_force = sim.simulate()
    optuna_loss = torch.square(hist_knife_force -
                        sim.groundtruth_torch[:len(hist_knife_force)]).mean()

    # Logging of the new values in the tensorboard summary writer
    for name, param in sim.parameters.items():
        logger.add_scalar(
            f"{name}/value", param.actual_tensor_value.mean().item(), trial.number)
        print(f'\t{name} = {param.actual_tensor_value.mean().item()}')

    logger.add_scalar("optuna_loss", optuna_loss.item(), trial.number)
    fig = sim.plot_simulation_results()
    fig.savefig(f"log/{experiment_name}/optuna_{trial.number}.png")
    logger.add_figure("simulation", fig, trial.number)

    # Returning the value to be minimized
    return optuna_loss

# Launching the study
sampler=optuna.samplers.TPESampler()
# sampler=optuna.samplers.CmaEsSampler()
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=optuna_trials, show_progress_bar=True)

# Display the information regarding the best trial
print("Best trial:", study.best_params)

optimized_tensors = {}
for key, value in study.best_params.items():
    if key in parameters.keys():
        optimized_tensors.update({key: torch.tensor([value]*sim.number_of_cut_spring, device=device, dtype=torch.float32, requires_grad=True)})
    else:
        settings[key] = value

# Remove params not optimized with Adam
del parameters['initial_y']

torch.save(optimized_tensors, f"log/{experiment_name}/best_optuna_optimized_tensors.pt")

#########################
### Adam Optimization ###
#########################
del sim
print("============ Adam Optimization ===============")
requires_grad = True
settings.sim_duration = 1.0

sim = CuttingSim(settings, experiment_name=experiment_name, adapter=device, requires_grad=requires_grad,
                 parameters=parameters, verbose=False)
sim.motion = ConstantLinearVelocityMotion(
    initial_pos=torch.tensor([0.0, settings.initial_y, 0.0], device=device),
    linear_velocity=torch.tensor([0.0, settings.velocity_y, 0.0], device=device))

sim.cut()

sim.load_groundtruth('osx_dataset/calibrated/cucumber_3_05.npy', groundtruth_dt=0.002)

opt_params = sim.init_parameters(optimized_tensors)

opt = torch.optim.Adam(opt_params, lr=learning_rate)

for iteration in tqdm.trange(100):
    sim.motion = ConstantLinearVelocityMotion(
        initial_pos=torch.tensor(
            [0.0, settings.initial_y, 0.0], device=device),
        linear_velocity=torch.tensor([0.0, settings.velocity_y, 0.0], device=device))

    print(f'\n### {experiment_name}  --  Iteration {iteration}')

    hist_knife_force = sim.simulate()

    sim.save_optimized_parameters(f"log/{experiment_name}/adam_optimized_tensors_{iteration}.pt")

    loss = torch.square(hist_knife_force -
                        sim.groundtruth_torch[:len(hist_knife_force)]).mean()
    print("Loss:", loss.item())

    for name, param in sim.parameters.items():
        logger.add_scalar(
            f"{name}/value", param.actual_tensor_value.mean().item(), iteration)

    logger.add_scalar("loss", loss.item(), iteration)

    fig = sim.plot_simulation_results()
    fig.savefig(f"log/{experiment_name}/adam_{iteration}.png")
    logger.add_figure("simulation", fig, iteration)
    opt.zero_grad()
    loss.backward(retain_graph=False)
    for name, param in sim.parameters.items():
        if param.tensor.grad is None:
            print(
                f'\t{name} = {param.actual_tensor_value.mean().item()} \t\tgrad N/A!')
            print(f"Iteration {iteration}: {name} has no gradient!")
        else:
            print(
                f'\t{name} = {param.actual_tensor_value.mean().item()} \t\tgrad = {param.tensor.grad.mean().item()}')
            logger.add_scalar(
                f"{name}/grad", param.tensor.grad.mean().item(), iteration)

    opt.step()
