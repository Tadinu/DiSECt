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
