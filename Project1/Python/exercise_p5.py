"""[Project1] Exercise 5: Turning while Swimming & Walking, Backward Swimming & Walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt


def exercise_5a_swim_turn(timestep):
    """[Project1] Exercise 5a: Turning while swimming"""

    # Parameters
    
    sim_parameters = SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.0],
            spawn_orientation=[0, 0, -np.pi/2],
            drive=4.9,  # An example of parameter part of the grid search
            turn=2
        )

    sim, data = simulation(
        sim_parameters=sim_parameters,  # Simulation parameters, see above
        arena='water',  # Can also be 'land', give it a try!
        fast=True,  # For fast mode (not real-time)
        #headless=True,  # For headless mode (No GUI, could be faster)
        record=False,  # Record video
        camera_id=2,  # camera type: 0=top view, 1=front view, 2=side view,
    )
    
    gps = np.array(data.sensors.links.urdf_positions()[:, :9])
    gps_x_pos=gps[:,4,0]
    gps_y_pos=gps[:,4,1]

    joints_positions = joints_positions = np.array(data.sensors.joints.positions_all())
    joints_positions[:, [1,5]] *= -1 #correcting for the wrong sensor data

    trajectory_points=np.linspace(0,30, len(gps_x_pos))

    plt.figure("Figure_5.1 - Spine angles")
    for i in range(0, 16):
        plt.plot(trajectory_points, joints_positions[:,i]-(i*2), label=("Joint "+str(i+1)))

    font_size=15
    plt.xlabel("Time [s]", fontsize=font_size)
    plt.ylabel("Joint angles for spine [rad]", fontsize=font_size)
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    plt.figure("Figure_5.1 - GPS trajectories")
    plt.plot(gps_x_pos, gps_y_pos)
    plt.xlabel("x positions of the GPS", fontsize=font_size)
    plt.ylabel("y positions of the GPS", fontsize=font_size)
    plt.show()
    pass
    return


def exercise_5b_swim_back(timestep):
    """[Project1] Exercise 5b: Backward Swimming"""

    # Parameters
    
    sim_parameters = SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.0],
            spawn_orientation=[0, 0, -np.pi/2],
            drive=3.5,  # An example of parameter part of the grid search
            phase_biases=[-2*np.pi/8, 2*np.pi/8, np.pi, np.pi, np.pi],
            turn=0
        )

    sim, data = simulation(
        sim_parameters=sim_parameters,  # Simulation parameters, see above
        arena='water',  # Can also be 'land', give it a try!
        fast=True,  # For fast mode (not real-time)
        #headless=True,  # For headless mode (No GUI, could be faster)
        record=False,  # Record video
        camera_id=2,  # camera type: 0=top view, 1=front view, 2=side view,
    )
    
    gps = np.array(data.sensors.links.urdf_positions()[:, :9])
    gps_x_pos=gps[:,4,0]
    gps_y_pos=gps[:,4,1]

    joints_positions = joints_positions = np.array(data.sensors.joints.positions_all())
    joints_positions[:, [1,5]] *= -1 #correcting for the wrong sensor data

    trajectory_points=np.linspace(0,30, len(gps_x_pos))

    plt.figure("Figure_5.2 - Spine angles")
    for i in range(0, 16):
        plt.plot(trajectory_points, joints_positions[:,i]-(i*2), label=("Joint "+str(i+1)))

    font_size=15
    plt.xlabel("Time [s]", fontsize=font_size)
    plt.ylabel("Joint angles for spine [rad]", fontsize=font_size)
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    plt.figure("Figure_5.2 - GPS trajectories")
    plt.plot(gps_x_pos, gps_y_pos)
    plt.xlabel("x positions of the GPS", fontsize=font_size)
    plt.ylabel("y positions of the GPS", fontsize=font_size)
    plt.show()
    pass
    return


def exercise_5c_walk_turn(timestep):
    """[Project1] Exercise 5c: Turning while Walking"""

    sim_parameters = SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.0],
            spawn_orientation=[0, 0, -np.pi/2],
            drive=2.9,  # An example of parameter part of the grid search
            turn=2
        )

    sim, data = simulation(
        sim_parameters=sim_parameters,  # Simulation parameters, see above
        arena='land',  # Can also be 'land', give it a try!
        fast=True,  # For fast mode (not real-time)
        #headless=True,  # For headless mode (No GUI, could be faster)
        record=False,  # Record video
        camera_id=2,  # camera type: 0=top view, 1=front view, 2=side view,
    )
    
    gps = np.array(data.sensors.links.urdf_positions()[:, :9])
    gps_x_pos=gps[:,4,0]
    gps_y_pos=gps[:,4,1]

    joints_positions = joints_positions = np.array(data.sensors.joints.positions_all())
    joints_positions[:, [1,5]] *= -1 #correcting for the wrong sensor data
    trajectory_points=np.linspace(0,30, len(gps_x_pos))

    plt.figure("Figure_5.3 - Spine angles")
    for i in range(0, 16):
        plt.plot(trajectory_points, joints_positions[:,i]-(i*2), label=("Joint "+str(i+1)))

    font_size=15
    plt.xlabel("Time [s]", fontsize=font_size)
    plt.ylabel("Joint angles for spine [rad]", fontsize=font_size)
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    plt.figure("Figure_5.3 - GPS trajectories")
    plt.plot(gps_x_pos, gps_y_pos)
    plt.xlabel("x positions of the GPS", fontsize=font_size)
    plt.ylabel("y positions of the GPS", fontsize=font_size)
    plt.show()

    pass
    return


def exercise_5d_walk_back(timestep):
    """[Project1] Exercise 5d: Backward Walking"""
    
    # Parameters
    
    sim_parameters = SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.0],
            spawn_orientation=[0, 0, -np.pi/2],
            drive=2.8,  # An example of parameter part of the grid search
            phase_biases=[2*np.pi/8, -2*np.pi/8, np.pi, -1.35, np.pi],
            turn=0,
            walk_back=True #to comment in order to obtain Figure 28
        )

    sim, data = simulation(
        sim_parameters=sim_parameters,  # Simulation parameters, see above
        arena='land',  # Can also be 'land', give it a try!
        fast=True,  # For fast mode (not real-time)
        #headless=True,  # For headless mode (No GUI, could be faster)
        record=False,  # Record video
        camera_id=2,  # camera type: 0=top view, 1=front view, 2=side view,
    )
    
    gps = np.array(data.sensors.links.urdf_positions()[:, :9])
    gps_x_pos=gps[:,4,0]
    gps_y_pos=gps[:,4,1]

    joints_positions = joints_positions = np.array(data.sensors.joints.positions_all())
    joints_positions[:, [1,5]] *= -1 #correcting for the wrong sensor data

    trajectory_points=np.linspace(0,30, len(gps_x_pos))

    plt.figure("Figure_5.4 - Spine angles")
    for i in range(0, 16):
        plt.plot(trajectory_points, joints_positions[:,i]-(i*2), label=("Joint "+str(i+1)))

    font_size=15
    plt.xlabel("Time [s]", fontsize=font_size)
    plt.ylabel("Joint angles for spine [rad]", fontsize=font_size)
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    plt.figure("Figure_5.4 - GPS trajectories")
    plt.plot(gps_x_pos, gps_y_pos)
    plt.xlabel("x positions of the GPS", fontsize=font_size)
    plt.ylabel("y positions of the GPS", fontsize=font_size)
    plt.show()

    pass
    return


if __name__ == '__main__':
    exercise_5a_swim_turn(timestep=1e-2)
    exercise_5b_swim_back(timestep=1e-2)
    exercise_5c_walk_turn(timestep=1e-2)
    exercise_5d_walk_back(timestep=1e-2)

