# Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This example demonstrates the basic cutting functionality of our simulator, and
will open an interactive 3D visualizer to show the simulation in real time.
"""

# fmt: off
from matplotlib import pyplot as plt
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from disect.cutting import load_settings, CuttingSim, Parameter

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
# settings = load_settings("examples/config/cooking/training_real_cucumber.json")

settings.sim_duration = 1.2
settings.sim_dt = 2e-5
settings.sim_substeps = 500
settings.initial_y = 0.03 + 0.025  # center of knife + actual desired height
settings.velocity_y = -0.020
device = "cuda"
requires_grad = False
experiment_name = "cutting_prism"


sim = CuttingSim(settings, experiment_name=experiment_name,
                 parameters=parameters,
                 adapter=device, requires_grad=requires_grad)

sim.load_groundtruth('osx_dataset/calibrated/cucumber_3_05.npy', groundtruth_dt=0.002)
# sim.motion.plot(settings.sim_duration)
# import matplotlib.pyplot as plt
# plt.show()

sim.cut()

optimized_parameters = torch.load(f'log/optuna_param_inference_dt1e-05_20230831-0550/best_optuna_optimized_tensors.pt')
if 'initial_y' in optimized_parameters.keys():
    optimized_parameters.update({'initial_y' : optimized_parameters['initial_y'][0]})
sim.init_parameters(optimized_parameters)

# sim.simulate()
# sim.plot_simulation_results()
# plt.show()
# sim.visualize_cut(cut_separation=0.01)

sim.visualize(plot_knife_force_history=False, auto_start=False)
