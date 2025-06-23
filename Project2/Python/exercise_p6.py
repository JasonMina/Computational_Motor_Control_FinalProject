"""[Project1] Exercise 6: Walking with sensory feedback"""

import os
import pickle
import numpy as np
import matplotlib.animation as manimation
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
from plot_results import main
import farms_pylog

#moving average filter
def smooth_data(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    smoothed_data = cumsum / window_size
    return smoothed_data

def exercise_6a_phase_relation(timestep):
    """Exercise 6a - Relationship between phase of limb oscillator & swing-stance
    (Project 2 Question 1)

    This exercise helps in understanding the relationship between swing-stance
    during walking and phase of the limb oscillators.

    Implement rigid spine with limb movements to understand the relationship.
    Hint:
        - Use the spine's nominal amplitude to make the spine rigid.

    Observer the limb phase output plot versus ground reaction forces.
        - Apply a threshold on ground reaction forces to remove small signals
        - Identify phase at which the limb is in the middle of stance
        - Identify which senstivity function is better suited for implementing
        tegotae feedback for our CPG-controller.
        - Identify if weights for contact feedback should be positive  or negative
        to get the right coordination

    """

    sim_parameters = SimulationParameters(
        duration=10,  # Simulation duration in [s]
        timestep=timestep,  # Simulation timestep in [s]
        spawn_position=[0.0, 0.0, 0.0], #change spawn position and orientation for land2water transition
        spawn_orientation=[0.0, 0.0, -np.pi],
        drive=2.8,  # An example of parameter part of the grid search
        rigid_spine=True
        )

    sim, data = simulation(
        sim_parameters=sim_parameters,  # Simulation parameters, see above
        arena='land',  # Can also be 'land', give it a try!
        fast=True,  # For fast mode (not real-time)
        headless=False,  # For headless mode (No GUI, could be faster)
        record=False,  # Record video
        camera_id=2,  # camera type: 0=top view, 1=front view, 2=side view,
    )

    fig, axs =plt.subplots(4,1)
    phases = data.state.phases()
    
    # Initialize an empty list to store cumulative moving averages
    for i in range(4):
        contact_forces=np.asarray(data.sensors.contacts.reaction_all(i)[:,2])
        contact_forces=smooth_data(contact_forces, 20) #smoothens the data by moving average filter
        contact_forces[contact_forces <= 7] = 0  #lower threshold
        contact_forces[contact_forces >= 20] = 0 #upper threshold
        phases_pi=phases[:, 16+i]/np.pi #phases divided by pi

        axs[i].plot(phases_pi, contact_forces)
        axs[i].set_title('Limb ' + str(i+1))
    fig.supylabel('Contact forces (on z-axis) for each limb')
    fig.supxlabel('Phases of each limb scaled by pi')
    fig.tight_layout()
    plt.show()
    
    pass
    return


def exercise_6b_tegotae_limbs(timestep):
    """Exercise 6b - Implement tegotae feedback
    (Project 2 Question 4)

    This exercise explores the role of local limb feedback. Such that the feedback
    from the limb affects only the oscillator corresponding to the limb.

    Keep the spine rigid and straight to observed the effect of tegotae feedback

    Implement uncoupled oscillators by setting the following:
    weights_body2body = 30
    weights_limb2body = 0
    weights_limb2limb = 0

    Implement only local sensory feedback(same limb) by setting:
    weights_contact_limb_i = (explore positive and negative range of values)

    Hint:
    - Apply weights in a small range, such that the feedback values are not greater than
    the oscilltor's intrinsic frequency
    - Observer the oscillator output plot. Check the phase relationship between
    limbs.

    """
    
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0.0, 0.0, 0.0], #change spawn position and orientation for land2water transition
            spawn_orientation=[0.0, 0.0, -np.pi],
            coupling_weights=[30, 30, 30, 0, 0],
            weights_contact_limb_i = weight,
            drive=2.8,  # An example of parameter part of the grid search
            rigid_spine=True,
            feedback = True
        )
        for weight in np.linspace(0.1, 0.3, 5)
    ]

    os.makedirs('./logs/exercise_6b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_6b/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    for i in range(0,5):
        filename_h5 = "logs/exercise_6b/simulation_{number}.h5".format(number= i)
        filename_pickle = "logs/exercise_6b/simulation_{number}.pickle".format(number= i)
        main(filename_h5, filename_pickle)
        plt.show()

    pass
    return


def exercise_6c_tegotae_spine(timestep):
    """Exercise 6c - Effect of spine undulation with tegotae feedback
    (Project 2 Question 5)

    This exercise explores the role of spine undulation and how
    to combine tegotae sensory feedback with spine undulations.

    We will implement the following cases with tegotae feedback:

    1. spine undulation, with no limb to body coupling, no limb to limb coupling
    2. spine undlation, with limb to body coupling, no limb to limb coupling
    3. spine undlation, with limb to body coupling, with limb to limb coupling

    Comment on the three cases, how they are similar and different.
    """

    parameter_set = [
        SimulationParameters(
            duration= 30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0.0, 0.0, 0.0], #change spawn position and orientation for land2water transition
            spawn_orientation=[0.0, 0.0, -np.pi],
            coupling_weights= coupling, 
            weights_contact_limb_i = 0.15,
            drive=2.8,  # An example of parameter part of the grid search
            rigid_spine=False,
            feedback = True
        )
        for coupling in [[10, 10, 10, 0, 0], 
                         [10, 10, 10, 30, 0],
                         [10, 10, 10, 30, 10]]
    ]
    
    os.makedirs('./logs/exercise_6c/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_6c/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    for i in range(0,3):
        filename_h5 = "logs/exercise_6c/simulation_{number}.h5".format(number= i)
        filename_pickle = "logs/exercise_6c/simulation_{number}.pickle".format(number= i)
        main(filename_h5, filename_pickle)
        plt.show()

    return


def exercise_6d_open_vs_closed(timestep):
    """Exercise 6d - Open loop vs closed loop behaviour
    (Project 2 Question 6)

    This exercise explores the differences in open-loop vs closed loop.

    Implement the following cases
    1. Open loop: spine undulation, with limb to body coupling, no limb to limb coupling
    2. Open loop: spine undlation, with limb to body coupling, with limb to limb coupling
    3. Closed loop: spine undulation, with limb to body coupling, no limb to limb coupling
    4. Closed loop: spine undlation, with limb to body coupling, with limb to limb coupling

    Comment on the three cases, how they are similar and different.
    """
    weights_coupling = [[10, 10, 10, 30, 0],
                        [10, 10, 10, 30, 10],
                        [10, 10, 10, 30, 0],
                        [10, 10, 10, 30, 10]]
    feedbacks = [False, False, True, True]

    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0.0, 0.0, 0.0], #change spawn position and orientation for land2water transition
            spawn_orientation=[0.0, 0.0, -np.pi],
            coupling_weights = weights,
            weights_contact_limb_i = 0.15,
            drive=2.8,  # An example of parameter part of the grid search
            rigid_spine=False,
            feedback = feedback_i
        )
        for weights, feedback_i in zip(weights_coupling, feedbacks)
    ]

    os.makedirs('./logs/exercise_6d/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_6d/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=False,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)

    for i in range(0,4):
        filename_h5 = "logs/exercise_6d/simulation_{number}.h5".format(number= i)
        filename_pickle = "logs/exercise_6d/simulation_{number}.pickle".format(number= i)
        main(filename_h5, filename_pickle)
        plt.show()

    return


if __name__ == '__main__':
    exercise_6a_phase_relation(timestep=1e-2)
    exercise_6b_tegotae_limbs(timestep=1e-2)
    exercise_6c_tegotae_spine(timestep=1e-2)
    exercise_6d_open_vs_closed(timestep=1e-2)

