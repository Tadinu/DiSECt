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
import sys, os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from disect.cutting import load_settings, create_sim

calibration_files = '/root/o2ac-ur/disect/log/best_results/potato'
calibration_files = '/root/o2ac-ur/disect/log/best_results/tomato'
calibration_files = '/root/o2ac-ur/disect/log/best_results/cucumber'


settings = load_settings(f"{calibration_files}/settings.json")
settings.sim_dt = 2e-5
settings.velocity_y = -0.05
experiment_name = "visualization"

device = "cuda"

sim, parameters = create_sim(settings, experiment_name, requires_grad=False, device=device, verbose=True, allow_nans=True)

# Load optimized/pretrained parameters
pretrained_params = f'{calibration_files}/params/adam_optimized_tensors_10.pt'
# pretrained_params = f'{calibration_files}/best_optuna_optimized_tensors.pkl'
# pretrained_params = f'{calibration_files}/best_adam_optimized_tensors.pt'
sim.load_optimized_parameters(pretrained_params, verbose=True, update_initial_y=True)

sim.init_parameters()

# sim.simulate()
# sim.plot_simulation_results()
# plt.show()

# np_hist_knife_force = sim.hist_knife_force.detach().cpu().numpy()
# sim_force = np.array([np.linspace(0., sim.sim_time, len(np_hist_knife_force)), np_hist_knife_force])
# np.save(f"{calibration_files}/sim_force.npy", sim_force)

# sim.visualize_cut(cut_separation=0.01)

sim.visualize(plot_knife_force_history=False, auto_start=False, render_frequency=10)
# sim.ros_visualizer()
