"""[Project1] Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
import matplotlib.pyplot as plt


def exercise_4a_transition(timestep):
    """[Project 1] 4a Transitions

    In this exerices, we will implement transitions.
    The salamander robot needs to perform swimming to walking
    and walking to swimming transitions.

    Hint:
        - set the  arena to 'amphibious'
        - use the sensor(gps) values to find the point where
        the robot should transition
        - simulation can be stopped/played in the middle
        by pressing the space bar
        - printing or debug mode of vscode can be used
        to understand the sensor values

    """
    spawn_positions_x=[5,-1]
    spawn_orientation=[-np.pi/2, np.pi/2]

    for i in range(0,2):
        sim_parameters = SimulationParameters(
                duration=30,  # Simulation duration in [s]
                timestep=timestep,  # Simulation timestep in [s]
                spawn_position=[spawn_positions_x[i], 0, 0.0], #change spawn position and orientation for land2water transition
                spawn_orientation=[0, 0, spawn_orientation[i]],
                drive=4.5,  # An example of parameter part of the grid search
                cond_transition=True
            )

        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            record=True,  # Record video
            record_path="videos/ex4." + str(i) + ".mp4",
            camera_id=2,  # camera type: 0=top view, 1=front view, 2=side view,
        )
        
        gps = np.array(data.sensors.links.urdf_positions()[:, :9])
        gps_x_pos=gps[:,4,0]

        joints_positions = joints_positions = np.array(data.sensors.joints.positions_all())
        joints_positions[:, [1,5]] *= -1 #correcting for the wrong sensor data
        
        trajectory_points=np.linspace(0,30, len(gps_x_pos))

        for j in range(0, len(joints_positions[0,:])):
            if j<8:
                label=("Spine joints "+str(j+1))
            else:
                label=("Limb joints "+str(j-7))
            plt.plot(trajectory_points, joints_positions[:,j]-(j*2), label=label)

        font_size=15
        plt.xlabel("Time [s]", fontsize=font_size)
        plt.ylabel("Joint angles for polymander [rad]", fontsize=font_size)
        plt.yticks([])
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    pass
    return

def exercise_4_plot_GPS(timestep):
    """[Project 1] 4a Transitions

    In this exerices, we will implement transitions.
    The salamander robot needs to perform swimming to walking
    and walking to swimming transitions.

    Hint:
        - set the  arena to 'amphibious'
        - use the sensor(gps) values to find the point where
        the robot should transition
        - simulation can be stopped/played in the middle
        by pressing the space bar
        - printing or debug mode of vscode can be used
        to understand the sensor values

    """
    spawn_positions_x=[5, -1]
    spawn_orientation=[-np.pi/2, np.pi/2]

    for i in range(0,2):
        sim_parameters = SimulationParameters(
                duration=30,  # Simulation duration in [s]
                timestep=timestep,  # Simulation timestep in [s]
                spawn_position=[spawn_positions_x[i], 0, 0.0], #change spawn position and orientation for land2water transition
                spawn_orientation=[0, 0, spawn_orientation[i]],
                drive=4.5,  # An example of parameter part of the grid search
                cond_transition=True
            )

        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/ex4_land2water.mp4",
            camera_id=2,  # camera type: 0=top view, 1=front view, 2=side view,
            
        )
        
        gps = np.array(data.sensors.links.urdf_positions()[:, :9])
        gps_x_pos=gps[:,4,0]

        joints_positions = joints_positions = np.array(data.sensors.joints.positions_all())
        joints_positions[:, [1,5]] *= -1 #correcting for the wrong sensor data
        
        for j in range(0, len(joints_positions[0,:])):
            if j<8:
                label=("Spine joints "+str(j+1))
            else:
                label=("Limb joints "+str(j-7))
                
            plt.plot(gps_x_pos, joints_positions[:,j]-(j*2), label=label)

        font_size=15
        
        plt.xlabel("X position of the GPS", fontsize=font_size)
        plt.ylabel("Joint angles for polymander [rad]", fontsize=font_size)
        plt.yticks([])
        plt.legend(loc="upper right")
        plt.tight_layout()
        #To invert X-axis when needed
        if i==0:
            plt.gca().invert_xaxis()
        plt.show()

    pass
    return

if __name__ == '__main__':
    exercise_4a_transition(timestep=1e-2)
    exercise_4_plot_GPS(timestep=1e-2)

