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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from disect.cutting import load_settings, create_sim

calibration_files = '/root/o2ac-ur/disect/log/best_results/20230908-131328_optuna_cucumber_param_inference_dt3e-05'
# calibration_files = '/root/o2ac-ur/disect/log/best_results/20230909-093744_optuna_potato_param_inference_dt3e-05'
# calibration_files = '/root/o2ac-ur/disect/log/best_results/20230908-161350_optuna_tomato_param_inference_dt3e-05'

settings = load_settings(f"{calibration_files}/settings.json")
settings.sim_dt = 2e-5
experiment_name = "visualization"

device = "cuda"

sim, parameters = create_sim(settings, experiment_name, requires_grad=False, device=device, verbose=True)

# Load optimized/pretrained parameters
pretrained_params = f'{calibration_files}/params/adam_optimized_tensors_17.pt'
sim.load_optimized_parameters(pretrained_params, verbose=True, update_initial_y=True)

sim.init_parameters()

sim.simulate()
sim.plot_simulation_results()
plt.show()
# sim.visualize_cut(cut_separation=0.01)

# sim.visualize(plot_knife_force_history=False, auto_start=False, render_frequency=10)
# sim.ros_visualizer()
