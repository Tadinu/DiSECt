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

from disect.cutting.cutting_sim import CuttingSim
from disect.cutting.motion import ConstantLinearVelocityMotion
from disect.cutting.settings import Parameter
import optuna
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import pickle
import timeit
import tqdm

def create_sim(settings, experiment_name, verbose=False, requires_grad=False, best_params={}, device='cuda', shared_params=False, allow_nans=False):
    parameters = {
        "initial_y": Parameter("initial_y", settings.initial_y, settings.initial_y - 0.005, settings.initial_y + 0.01),
        "cut_spring_ke": Parameter("cut_spring_ke", 200, 100, 8000),
        "cut_spring_softness": Parameter("cut_spring_softness", 200, 10, 5000),
        # "cut_spring_kd": Parameter("cut_spring_kd", 0.1, 0.01, 0.15),
        "sdf_ke": Parameter("sdf_ke", 500, 200., 8000, individual=True),
        "sdf_kd": Parameter("sdf_kd", 1., 0.1, 100., individual=True),
        "sdf_kf": Parameter("sdf_kf", 0.01, 0.001, 8000.0, individual=True),
        "sdf_mu": Parameter("sdf_mu", 0.5, 0.45, 1.0, individual=True),
        # "damping": Parameter("damping", 1., 0.1, 10., individual=True),
        # "density": Parameter("density", 700, 500., 1000., individual=True),
    }
    print("Parameters:", parameters)

    if shared_params:
        for p in parameters.values():
            p.individual = False

    if best_params:
        for key, value in best_params.items():
            constrain = parameters[key].range * 0.10
            parameters[key].low = max(value - constrain, parameters[key].low)
            parameters[key].high = min(value + constrain, parameters[key].high)
            parameters[key].set_value(value)

    sim = CuttingSim(settings, experiment_name=experiment_name, adapter=device, requires_grad=requires_grad,
                    parameters=parameters, verbose=verbose, allow_nans=allow_nans)
    sim.motion = ConstantLinearVelocityMotion(
        initial_pos=torch.tensor([0.0, settings.initial_y, 0.0], device=device),
        linear_velocity=torch.tensor([0.0, settings.velocity_y, 0.0], device=device))

    sim.cut()

    assert settings.groundtruth, "No groundtruth file path defined"
    sim.load_groundtruth(settings.groundtruth, groundtruth_dt=settings.get('groundtruth_dt', None))

    return sim, parameters

def optuna_trainer(sim, parameters, logger, n_trials=100):
    global best_loss
    best_loss = 1e8

    def objective(trial):
        global best_loss
        # Creating the optuna dict
        suggestion = {}
        for name, param in parameters.items():
            suggestion.update({name: trial.suggest_float(name, param.low, param.high)})

        sim.load_optimized_parameters(optimized_params=suggestion)
        sim.init_parameters()

        # Computing the optuna_loss function
        try:
            hist_knife_force = sim.simulate()
            # L2 loss
            optuna_loss = torch.square(hist_knife_force - sim.groundtruth_torch[:len(hist_knife_force)]).mean()
            # L1 loss
            # optuna_loss = torch.abs(hist_knife_force - sim.groundtruth_torch[:len(hist_knife_force)]).mean()


            # Logging of the new values in the tensorboard summary writer
            for name, param in sim.parameters.items():
                logger.add_scalar(f"{name}/value", param.actual_tensor_value.mean().item(), trial.number)

            logger.add_scalar("optuna_loss", optuna_loss.item(), trial.number)
            fig = sim.plot_simulation_results()
            logger.add_figure("simulation", fig, trial.number)
            
            # Save improments
            if optuna_loss < best_loss:
                optuna_results = f"log/{sim.experiment_name}/params/optuna_params_{trial.number:03}.pkl"
                pickle.dump(suggestion, open(optuna_results, "wb"))
                fig.savefig(f"log/{sim.experiment_name}/plots/optuna_{trial.number}.png")
                best_loss = optuna_loss

        except AssertionError:
            optuna_loss = 1e5 + (1 - (sim.sim_time/sim.sim_duration)) * 1e5
            print(f"NaN values found at trial {trial.number} num of timesteps {sim.sim_time}")

        # Returning the value to be minimized
        return optuna_loss

    # Launching the study
    sampler = optuna.samplers.TPESampler()
    # sampler=optuna.samplers.CmaEsSampler()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def adam_trainer(sim, logger, learning_rate, iterations=100, previous_best=None):

    if previous_best:
        sim.load_optimized_parameters(previous_best)
    opt_params = sim.init_parameters()

    opt = torch.optim.Adam(opt_params, lr=learning_rate)
    scheduler = StepLR(opt, step_size=int(iterations/4.), gamma=0.1)

    for iteration in tqdm.trange(iterations):
        sim.motion = ConstantLinearVelocityMotion(initial_pos=torch.tensor([0.0, sim.settings.initial_y, 0.0], device=sim.adapter),
                                                  linear_velocity=torch.tensor([0.0, sim.settings.velocity_y, 0.0], device=sim.adapter))

        print(f'\n### {sim.experiment_name}  --  Iteration {iteration}')

        hist_knife_force = sim.simulate()

        sim.save_optimized_parameters(f"log/{sim.experiment_name}/params/adam_optimized_tensors_{iteration}.pt")

        loss = torch.square(hist_knife_force - sim.groundtruth_torch[:len(hist_knife_force)]).mean()
        print("Loss:", loss.item())

        for name, param in sim.parameters.items():
            logger.add_scalar(f"{name}/value", param.actual_tensor_value.mean().item(), iteration)

        logger.add_scalar("loss", loss.item(), iteration)

        fig = sim.plot_simulation_results()
        fig.savefig(f"log/{sim.experiment_name}/plots/adam_{iteration}.png")
        logger.add_figure("simulation", fig, iteration)
        opt.zero_grad()
        loss.backward(retain_graph=False)
        for name, param in sim.parameters.items():
            if param.tensor.grad is None:
                print(f'\t{name} = {param.actual_tensor_value.mean().item()} \t\tgrad N/A!')
                print(f"Iteration {iteration}: {name} has no gradient!")
                return
            else:
                print(f'\t{name} = {param.actual_tensor_value.mean().item()} \t\tgrad = {param.tensor.grad.mean().item()}')
                logger.add_scalar(f"{name}/grad", param.tensor.grad.mean().item(), iteration)

        opt.step()
        scheduler.step()
