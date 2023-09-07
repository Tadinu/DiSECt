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

from disect.cutting import load_settings, CuttingSim, Parameter, padding_tensors

# fmt: on
parameters = {
    # "initial_y": Parameter("initial_y", 0.055, 0.05, 0.06),
    "cut_spring_ke": Parameter("cut_spring_ke", 200, 100, 8000),
    "cut_spring_softness": Parameter("cut_spring_softness", 200, 10, 5000),
    "sdf_ke": Parameter("sdf_ke", 500, 200., 8000, individual=True),
    "sdf_kd": Parameter("sdf_kd", 1., 0.1, 100.),
    "sdf_kf": Parameter("sdf_kf", 0.01, 0.001, 8000.0),
    "sdf_mu": Parameter("sdf_mu", 0.5, 0.45, 1.0),
}

settings = load_settings("examples/config/cooking/ansys_cucumber.json")
veggie_height = 0.028
# settings = load_settings("examples/config/cooking/ansys_potato.json")
# veggie_height = 0.041
# settings = load_settings("examples/config/cooking/ansys_tomato.json")
# veggie_height = 0.049

settings.sim_duration = 1.3
settings.sim_dt = 4e-5
settings.sim_substeps = 500
settings.initial_y = 59.e-3/2. + veggie_height  # center of knife + actual desired height
settings.velocity_y = -0.020
device = "cuda"
requires_grad = False
experiment_name = "cutting_prism"


sim = CuttingSim(settings, experiment_name=experiment_name,
                 parameters=parameters,
                 adapter=device, requires_grad=requires_grad)

# sim.motion.plot(settings.sim_duration)
# import matplotlib.pyplot as plt
# plt.show()

sim.cut()

# Load ground truth for plotting
sim.load_groundtruth('osx_dataset/calibrated/cucumber_3_05.npy', groundtruth_dt=0.002)
# sim.load_groundtruth('osx_dataset/calibrated/tomato_01.npy', groundtruth_dt=0.002)

# Load optimized/pretrained parameters
# pretrained_params = 'log/optuna_cucumber_param_inference_dt2e-05_20230904-1342/best_optuna_optimized_tensors.pkl'
pretrained_params = '/root/o2ac-ur/DiSECt2/log/optuna_cucumber_param_inference_dt2e-05_20230905-1824/best_optuna_optimized_tensors.pkl'

sim.load_optimized_parameters(pretrained_params, verbose=True)

sim.init_parameters()

sim.simulate()
sim.plot_simulation_results()
plt.show()
# sim.visualize_cut(cut_separation=0.01)

# sim.visualize(plot_knife_force_history=False, auto_start=False, render_frequency=10)
# sim.ros_visualizer()
