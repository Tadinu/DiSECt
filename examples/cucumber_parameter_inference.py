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
from datetime import datetime
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from disect.cutting import CuttingSim
from disect.cutting import load_settings, ConstantLinearVelocityMotion, Parameter
# fmt: on

parameters = {
    "cut_spring_ke": Parameter("cut_spring_ke", 200, 100, 4000),
    "cut_spring_softness": Parameter("cut_spring_softness", 200, 10, 5000),
    "sdf_ke": Parameter("sdf_ke", 500, 200., 4000, individual=True),
    "sdf_kd": Parameter("sdf_kd", 1., 0.1, 100.),
    "sdf_kf": Parameter("sdf_kf", 0.01, 0.001, 8000.0),
    "sdf_mu": Parameter("sdf_mu", 0.5, 0.45, 1.0),
}

settings = load_settings("examples/config/cooking/ansys_cucumber.json")
# settings = load_settings("examples/config/ansys_sphere_apple.json")
settings.sim_duration = 1.0
settings.sim_dt = 1e-5
settings.initial_y = 0.03 + 0.025 # center of knife + actual desired height
settings.velocity_y = -0.020
device = "cuda"
learning_rate = 0.01

now = datetime.now()
experiment_name = f"param_inference_dt{settings.sim_dt}_{now.strftime('%Y%m%d-%H%M')}"
logger = SummaryWriter(logdir=f"log/{experiment_name}")

requires_grad = True

sim = CuttingSim(settings, experiment_name=experiment_name, adapter=device, requires_grad=requires_grad,
                 parameters=parameters)
sim.motion = ConstantLinearVelocityMotion(
    initial_pos=torch.tensor([0.0, settings.initial_y, 0.0], device=device),
    linear_velocity=torch.tensor([0.0, settings.velocity_y, 0.0], device=device))

sim.cut()

# sim.visualize()
optimized_parameters = torch.load(f'log/optuna_param_inference_dt1e-05_20230831-0513/best_optuna_optimized_tensors.pt')

opt_params = sim.init_parameters(optimized_parameters)

# sim.load_groundtruth('dataset/forces/sphere_fine_resultant_force_xyz.csv')
sim.load_groundtruth('osx_dataset/calibrated/cucumber_3_05.npy', groundtruth_dt=0.002)

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
    fig.savefig(f"log/{experiment_name}/{experiment_name}_{iteration}.png")
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
