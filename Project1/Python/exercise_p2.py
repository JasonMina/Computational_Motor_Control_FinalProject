import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
from plot_results import plot_2d
import plot_results as plot_results
from salamandra_simulation.data import SalamandraData
import matplotlib.pyplot as plt
import farms_pylog as pylog

def exercise_2a_swim(timestep):
    """[Project 1] Exercise 2a Swimming

    In this exercise we need to implement swimming for salamander robot.
    Check exericse_example.py to see how to setup simulations.

    Run the simulations for different swimming drives and phase lag between body
    oscillators.
    """
    # Use exercise_example.py for reference
    phase_lags = [np.pi/8, 2*np.pi/8, 3*np.pi/8, 4*np.pi/8, 5*np.pi/8]
    
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.12],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=drive,  # An example of parameter part of the grid search
            phase_biases=[phi, -phi, np.pi, np.pi, np.pi],
            phase_lag_body=phi
        )
        for drive in np.linspace(3.0, 5.0, 5)
        for phi in phase_lags
    ]

    # Grid search
    os.makedirs('./logs/exercise_2a/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_2a/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    pass
    return


def exercise_2b_walk(timestep):
    """[Project 1] Exercise 2a Walking

    In this exercise we need to implement walking for salamander robot.
    Check exericse_example.py to see how to setup simulations.

    Run the simulations for different walking drives and phase lag between body
    oscillators.
    """
    # Use exercise_example.py for reference
    
    phase_lags = [np.pi/8, 2*np.pi/8, 3*np.pi/8, 4*np.pi/8, 5*np.pi/8]
    
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.12],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=drive,  # An example of parameter part of the grid search
            phase_biases=[phi, -phi, np.pi, np.pi, np.pi],
            phase_lag_body=phi
        )
        for drive in np.linspace(1.0, 3.0, 5)
        for phi in phase_lags
    ]

    # Grid search
    os.makedirs('./logs/exercise_2b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_2b/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'land', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            record=False,  # Record video
            record_path="videos/test_video_drive_" + \
            str(simulation_i),  # video savging path
            camera_id=2  # camera type: 0=top view, 1=front view, 2=side view,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    pass
    return


def exercise_test_walk(timestep):
    "[Project 1] Q2 Swimming"
    # Use exercise_example.py for reference
    
    filename = './logs/exercise_2b/simulation_{}.{}'
    
    # Load and plot data
    # Contains 25 combinations of nominal_amps, phase_lags and the resulting characteristic measured:
    total_torques = np.zeros((25,3)) # torques
    forward_speeds =  np.zeros((25,3)) # forward speed 
    lateral_speeds=np.zeros((25,3)) #lateral speed
    max_distances = np.zeros((25,3)) # max traveled distance
    energies=np.zeros((25,3)) #energies 
    inverse_CoTs=np.zeros((25,3)) #inverse of cost of transport
    max_inv_CoT = 0 # max(1/CoT)
    best_params = [] # optimized parameters leading to max(1/CoT)

    for i in range(25):
        data = SalamandraData.from_file(filename.format(i, 'h5'))
        with open(filename.format(i, 'pickle'), 'rb') as param_file:
            parameters = pickle.load(param_file)
        
        drive = parameters.drive
        phase_lag = parameters.phase_biases[0]
        joints_positions = data.sensors.joints.positions_all()
        joints_positions=np.asarray(joints_positions)
        joints_velocities = data.sensors.joints.velocities_all()
        joints_velocities=np.asarray(joints_velocities)
        links_velocities = data.sensors.links.com_lin_velocities()
        links_velocities=np.asarray(links_velocities)
        joints_torques = data.sensors.joints.motor_torques_all()
        joints_torques=np.asarray(joints_torques)
        links_positions = data.sensors.links.urdf_positions()
        links_positions=np.asarray(links_positions)
        joints_positions[:, [1,5]] *= -1
        joints_velocities[:, [1,5]] *= -1
        joints_torques[:, [1,5]] *= -1
        #Traveled Distance
        max_distance= plot_results.max_distance(links_positions)
        max_distances[i]=[phase_lag,  drive, max_distance]
        # Total Torque
        sum_torques=plot_results.sum_torques(joints_torques)
        total_torques[i]=[phase_lag,  drive, sum_torques]
        # forward_speeds
        forward_speed=plot_results.compute_speed(links_positions,links_velocities)[0]
        forward_speeds[i]= [phase_lag,  drive, forward_speed]
        #later_speeds
        lateral_speed= plot_results.compute_speed(links_positions,links_velocities)[1]
        lateral_speeds[i]= [phase_lag,  drive, lateral_speed]
        
        #Energy used
        # Numerical integration with the trapeze method : np.trapz
        energy = np.sum(np.trapz(joints_velocities[:,:]*joints_torques[:,:], dx = timestep))
        energies[i] = [phase_lag, drive, energy]
        ## 1/CoT : defined here as (traveled distance)/energy
        inv_CoT = max_distance/energy
        inverse_CoTs[i] = [phase_lag, drive, inv_CoT]
        # Check if this set of parameters is better than the previous ones for maximizing 1/CoT
        if max_inv_CoT < inv_CoT:
            max_inv_CoT = inv_CoT
            best_params = [phase_lag, drive,i]
        
    
    # Plot the heat maps of the results of the grid search
    plt.figure("Figure_2.2 - grid search results", figsize=(14, 12))

    ## forward_Speed
    plt.subplot(2,2,1)
    plot_2d(forward_speeds, ['Phase lag [rad]', 'Drive', 'Forward_Speed [m/s]'], n_data=5, log=False)
    
    ## lateral_speed
    plt.subplot(2,2,2)
    plot_2d(lateral_speeds, ['Phase lag [rad]', 'Drive', 'Later_Speed [m/s]'], n_data=5, log=False)
    
    ## Total_Torque
    plt.subplot(2,2,3)
    plot_2d(total_torques, ['Phase lag [rad]', 'Drive', 'Total Torque [Nm]'], n_data=5, log=False)

    ## Traveled_Distance
    plt.subplot(2,2,4)
    plot_2d(max_distances, ['Phase lag [rad]', 'Drive', 'Traveled Distance [m]'], n_data=5, log=False)
    plt.show()
    plt.figure("Figure_2.2 - Energy and COT grid search results", figsize=(14, 12))
    ## Energy
    plt.subplot(2,2,1)
    plot_2d(energies, ['Phase lag [rad]', 'Drive', 'Energy [J]'], n_data=5, log=False)

    # iCOT
    plt.subplot(2,2,2)
    plot_2d(inverse_CoTs, ['Phase lag [rad]', 'Drive', 'iCOT [d/E]'], n_data=5, log=False)
    plt.show()
    pylog.info('Best parameters for salamander walking locomotion: Phase Lag = {} [rad], Drive = {}'.format(best_params[0], best_params[1]))

    data = SalamandraData.from_file(filename.format(best_params[2], 'h5'))
    with open(filename.format(best_params[2], 'pickle'), 'rb') as param_file:
        parameters = pickle.load(param_file)
    
    n_iterations = np.shape(data.sensors.links.array)[0]
    times = np.arange(
         start=0,
         stop=timestep*n_iterations,
         step=timestep,
         )

    osc_phases = np.asarray(data.state.phases())
    # Compute the phase lags along the spine (downwards here)
    phase_lags = np.diff(osc_phases[:,:16], axis = 1)
    # Remove the phase difference between oscillators 8-9 that are not coupled
    phase_lags = np.concatenate((phase_lags[:,:7], phase_lags[:,8:]), axis = 1)
    
    joints_positions = np.asanyarray(data.sensors.joints.positions_all()) 
    joints_positions[:, [1,5]] *= -1

    ## PLOT RESULTS
    plt.figure("Figure_2.3 - Spine movement walking ", figsize=(16, 14)) 
    # Plot the spine angles 
    plt.subplot(2, 2, 1)
    plot_results.plot_spine_angles(times, joints_positions, vspace=2)

    # Plot the phase differences along the spine 
    plt.subplot(2, 2, 2)
    for i in range(3):
        if (i==0): 
            plt.plot(times, phase_lags[:,i], color='blue', label='Trunk')
        else: 
            plt.plot(times, phase_lags[:,i], color='blue')
    plt.plot(times, phase_lags[:,3], color='red', label='Trunk/Tail transition')

    for i in range(4,7):
        if(i==4): 
            plt.plot(times, phase_lags[:,i], color='green', label='Tail')
        else: 
            plt.plot(times,  phase_lags[:,i], color='green')

    for i in range(7,10):
        plt.plot(times, phase_lags[:,i], color='blue')

    plt.plot(times, phase_lags[:,10], color='red')

    for i in range(11,14):
        plt.plot(times, phase_lags[:,i], color='green')

    plt.xlabel("Time [s]")
    plt.ylabel("Phase differences")
    plt.grid()
    plt.legend(loc=1)
    
    # Plot the stable phases differences along the spine oscillators 
    oscillators = np.arange(1, 15, 1)
    stable_phase_lags = np.mean(phase_lags[-200:,:], axis = 0)
    # Left spine oscillators 
    plt.subplot(2, 2, 3)
    plt.plot(oscillators[:7], stable_phase_lags[:7], marker='o', label = 'Left side')
    plt.xticks(oscillators[:7], [str(oscillator) for oscillator in oscillators[:7]])
    plt.xlabel("Oscillators coupling index (vertical coupling between 2 body oscillators)")
    plt.ylabel("Stable phase differences")
    plt.grid()
    plt.legend(loc=2)
    plt.subplot(2, 2, 4)
    # Right spine oscillators
    oscillators[7:]=[a+1 for a in oscillators[7:]]
    plt.plot(oscillators[7:], stable_phase_lags[7:], marker='o', label = 'Right side')
    plt.xticks(oscillators[7:], [str(oscillator) for oscillator in oscillators[7:]])
    plt.xlabel("Oscillators coupling index (vertical coupling between 2 body oscillators)")
    plt.ylabel("Stable phase differences")
    plt.grid()
    plt.legend(loc=2)
    plt.show()
    
    pass
    return


def exercise_test_swim(timestep):
    "[Project 1] Q2 Swimming"
    # Use exercise_example.py for reference
    filename = './logs/exercise_2a/simulation_{}.{}'
    
    # Load and plot data
    # Contains 25 combinations of nominal_amps, phase_lags and the resulting characteristic measured:
    total_torques = np.zeros((25,3)) # torques
    forward_speeds =  np.zeros((25,3)) # forward speed 
    lateral_speeds=np.zeros((25,3)) #lateral speed
    max_distances = np.zeros((25,3)) # max traveled distance
    energies=np.zeros((25,3)) #energies 
    inverse_CoTs=np.zeros((25,3)) #inverse of cost of transport
    max_inv_CoT = 0 # max(1/CoT)
    best_params = [] # optimized parameters leading to max(1/CoT)

    for i in range(25):
        data = SalamandraData.from_file(filename.format(i, 'h5'))
        with open(filename.format(i, 'pickle'), 'rb') as param_file:
            parameters = pickle.load(param_file)
        
        drive = parameters.drive
        phase_lag = parameters.phase_biases[0]
        joints_positions = data.sensors.joints.positions_all()
        joints_positions=np.asarray(joints_positions)
        joints_velocities = data.sensors.joints.velocities_all()
        joints_velocities=np.asarray(joints_velocities)
        links_velocities = data.sensors.links.com_lin_velocities()
        links_velocities=np.asarray(links_velocities)
        joints_torques = data.sensors.joints.motor_torques_all()
        joints_torques=np.asarray(joints_torques)
        links_positions = data.sensors.links.urdf_positions()
        links_positions=np.asarray(links_positions)
        joints_positions[:, [1,5]] *= -1
        joints_velocities[:, [1,5]] *= -1
        joints_torques[:, [1,5]] *= -1
        #Traveled Distance
        max_distance= plot_results.max_distance(links_positions)
        max_distances[i]=[phase_lag,  drive, max_distance]
        # Total Torque
        sum_torques=plot_results.sum_torques(joints_torques)
        total_torques[i]=[phase_lag,  drive, sum_torques]
        # forward_speeds
        forward_speed=plot_results.compute_speed(links_positions,links_velocities)[0]
        forward_speeds[i]= [phase_lag,  drive, forward_speed]
        #later_speeds
        lateral_speed= plot_results.compute_speed(links_positions,links_velocities)[1]
        lateral_speeds[i]= [phase_lag,  drive, lateral_speed]
        
        #Energy used
        # Numerical integration with the trapeze method : np.trapz
        energy = np.sum(np.trapz(joints_velocities[:,:]*joints_torques[:,:], dx = timestep))
        energies[i] = [phase_lag, drive, energy]
        ## 1/CoT : defined here as (traveled distance)/energy
        inv_CoT = max_distance/energy
        inverse_CoTs[i] = [phase_lag, drive, inv_CoT]
        # Check if this set of parameters is better than the previous ones for maximizing 1/CoT
        if max_inv_CoT < inv_CoT:
            max_inv_CoT = inv_CoT
            best_params = [phase_lag, drive, i]
    

    # Plot the heat maps of the results of the grid search
    plt.figure("Figure_2.1 - grid search results", figsize=(14, 12))

    ## forward_Speed
    plt.subplot(2,2,1)
    plot_2d(forward_speeds, ['Phase lag [rad]', 'Drive', 'Forward_Speed [m/s]'], n_data=5, log=False)
    
    ## lateral_speed
    plt.subplot(2,2,2)
    plot_2d(lateral_speeds, ['Phase lag [rad]', 'Drive', 'Later_Speed [m/s]'], n_data=5, log=False)
    
    ## Total_Torque
    plt.subplot(2,2,3)
    plot_2d(total_torques, ['Phase lag [rad]', 'Drive', 'Total Torque [Nm]'], n_data=5, log=False)

    ## Traveled_Distance
    plt.subplot(2,2,4)
    plot_2d(max_distances, ['Phase lag [rad]', 'Drive', 'Traveled Distance [m]'], n_data=5, log=False)
    plt.show()
    
    plt.figure("Figure_2.1 - Energy and COT grid search results", figsize=(14, 12))
    ## Energy
    plt.subplot(2,2,1)
    plot_2d(energies, ['Phase lag [rad]', 'Drive', 'Energy [J]'], n_data=5, log=False)

    # iCOT
    plt.subplot(2,2,2)
    plot_2d(inverse_CoTs, ['Phase lag [rad]', 'Drive', 'iCOT [d/E]'], n_data=5, log=False)
    plt.show()
    pylog.info('Best parameters for salamander swimming locomotion: Phase Lag = {} [rad], Drive = {}'.format(best_params[0], best_params[1]))
    
    data = SalamandraData.from_file(filename.format(best_params[2], 'h5'))
    with open(filename.format(best_params[2], 'pickle'), 'rb') as param_file:
        parameters = pickle.load(param_file)
    
    n_iterations = np.shape(data.sensors.links.array)[0]
    times = np.arange(
         start=0,
         stop=timestep*n_iterations,
         step=timestep,
         )

    osc_phases = np.asarray(data.state.phases())
    # Compute the phase lags along the spine (downwards here)
    phase_lags = np.diff(osc_phases[:,:16], axis = 1)
    
    # Remove the phase difference between oscillators 8-9 that are not coupled
    phase_lags = np.concatenate((phase_lags[:,:7], phase_lags[:,8:]), axis = 1)
    
    joints_positions = np.asanyarray(data.sensors.joints.positions_all()) 
    joints_positions[:, [1,5]] *= -1
    ## PLOT RESULTS
    plt.figure("Figure_2.3 - Spine movement swimming ", figsize=(16, 14)) 
    # Plot the spine angles 
    plt.subplot(2, 2, 1)
    plot_results.plot_spine_angles(times, joints_positions, vspace=2)

    # Plot the phase differences along the spine 
    plt.subplot(2, 2, 2)
    for i in range(3):
        if(i==0): 
            plt.plot(times, phase_lags[:,i], color='blue', label='Trunk')
        else: 
            plt.plot(times, phase_lags[:,i], color='blue')
    plt.plot(times, phase_lags[:,3], color='red', label='Trunk/Tail transition')

    for i in range(4,7):
        if(i==4): 
            plt.plot(times, phase_lags[:,i], color='green', label='Tail')
        else: 
            plt.plot(times,  phase_lags[:,i], color='green')

    for i in range(7,10):
        plt.plot(times, phase_lags[:,i], color='blue')
    plt.plot(times, phase_lags[:,10], color='red')

    for i in range(11,14):
        plt.plot(times, phase_lags[:,i], color='green')
        
    plt.xlabel("Time [s]")
    plt.ylabel("Phase differences")
    plt.grid()
    plt.legend(loc=1)
    
    # Plot the stable phases differences along the spine oscillators 
    oscillators = np.arange(1, 15, 1)
    stable_phase_lags = np.mean(phase_lags[-200:,:], axis = 0)
    # Left spine oscillators 
    plt.subplot(2, 2, 3)
    plt.plot(oscillators[:7], stable_phase_lags[:7], marker='o', label = 'Left side')
    plt.xticks(oscillators[:7], [str(oscillator) for oscillator in oscillators[:7]])
    plt.xlabel("Oscillators coupling index (vertical coupling between 2 body oscillators)")
    plt.ylabel("Stable phase differences")
    plt.grid()
    plt.legend(loc=2)
    plt.subplot(2, 2, 4)
    # Right spine oscillators
    oscillators[7:]=[a+1 for a in oscillators[7:]]
    plt.plot(oscillators[7:], stable_phase_lags[7:], marker='o', label = 'Right side')
    plt.xticks(oscillators[7:], [str(oscillator) for oscillator in oscillators[7:]])
    plt.xlabel("Oscillators coupling index (vertical coupling between 2 body oscillators)")
    plt.ylabel("Stable phase differences")
    plt.grid()
    plt.legend(loc=2)

    pass
    return


if __name__ == '__main__':
    exercise_2a_swim(timestep=1e-2)
    exercise_2b_walk(timestep=1e-2)
    
    exercise_test_swim(timestep=1e-2)
    exercise_test_walk(timestep=1e-2)
