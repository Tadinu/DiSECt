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
    "cut_spring_ke": Parameter("cut_spring_ke", 200, 100, 4000),
    "cut_spring_softness": Parameter("cut_spring_softness", 200, 10, 5000),
    "sdf_ke": Parameter("sdf_ke", 1000, 200., 4000, individual=True),
    "sdf_kd": Parameter("sdf_kd", 1., 0.1, 100.),
    "sdf_kf": Parameter("sdf_kf", 0.01, 0.001, 8000.0),
    "sdf_mu": Parameter("sdf_mu", 0.5, 0.45, 1.0),
}

# settings = load_settings("examples/config/ansys_prism.json")
# settings = load_settings("examples/config/ansys_cylinder_jello.json")
# settings = load_settings("examples/config/ansys_sphere_apple.json")
settings = load_settings("examples/config/cooking/ansys_cucumber.json")
settings.sim_duration = 1.0
settings.sim_dt = 1e-5  # 5e-5
settings.initial_y = 0.055
settings.velocity_y = -0.02
experiment_name = "cutting_prism"
device = "cuda"
requires_grad = False


optimized_parameters = torch.load(f'log/param_inference_dt1e-05_20230831-0240/optimized_tensors_0.pt')
sim = CuttingSim(settings, experiment_name=experiment_name,
                 parameters=parameters,
                 adapter=device, requires_grad=requires_grad)

sim.load_groundtruth('osx_dataset/calibrated/cucumber_3_05.npy', groundtruth_dt=0.002)
# sim.motion.plot(settings.sim_duration)
# import matplotlib.pyplot as plt
# plt.show()

sim.cut()

sim.init_parameters(optimized_parameters)

sim.simulate()
sim.plot_simulation_results()
plt.show()
# sim.visualize_cut(cut_separation=0.01)

# sim.visualize(plot_knife_force_history=False, auto_start=False)
