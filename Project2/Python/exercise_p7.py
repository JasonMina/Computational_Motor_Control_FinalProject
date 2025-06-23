
import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
import matplotlib.pyplot as plt


def exercise_7a_transition(timestep):

    spawn_positions_x=[5,-1]
    spawn_orientation=[-np.pi/2, np.pi/2]

    for i in range(0,2):
        sim_parameters = SimulationParameters(
                duration=30,  # Simulation duration in [s]
                timestep=timestep,  # Simulation timestep in [s]
                spawn_position=[spawn_positions_x[i], 0, 0.0], #change spawn position and orientation for land2water transition
                spawn_orientation=[0, 0, spawn_orientation[i]],
                drive=4.5,  # An example of parameter part of the grid search
                cond_transition=True,
                feedback=True
            )

        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'land', give it a try!
            fast=False,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/ex4." + str(i) + ".mp4",
            camera_id=2,  # camera type: 0=top view, 1=front view, 2=side view,
        )

    pass
    return

if __name__ == '__main__':
    exercise_7a_transition(timestep=1e-2)
    

