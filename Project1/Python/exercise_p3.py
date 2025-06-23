"""[Project1] Exercise 3: Limb and Spine Coordination while walking"""

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


def exercise_3a_coordination(timestep):
    """[Project 1] Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and leg oscillators.

    """
    phase_lags = list(np.linspace(-np.pi-np.pi/9, np.pi+np.pi/9, 10))
    
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.12],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=drive,  # An example of parameter part of the grid search
            phase_biases=[2*np.pi/8, -2*np.pi/8, np.pi, phi, np.pi],
            phase_lag_body=phi
        )
        for drive in np.linspace(1.0, 3.0, 10)
        for phi in phase_lags
    ]

    # Grid search
    os.makedirs('./logs/exercise_3a/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_3a/simulation_{}.{}'
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
    
    filename = './logs/exercise_3a/simulation_{}.{}'
    
    # Load and plot data
    # Contains 25 combinations of nominal_amps, phase_lags and the resulting characteristic measured:
    total_torques = np.zeros((100,3)) # torques
    forward_speeds =  np.zeros((100,3)) # forward speed 
    lateral_speeds=np.zeros((100,3)) #lateral speed
    max_distances = np.zeros((100,3)) # max traveled distance
    energies=np.zeros((100,3)) #energies 
    inverse_CoTs=np.zeros((100,3)) #inverse of cost of transport
    max_inv_CoT = 0 # max(1/CoT)
    best_params = [] # optimized parameters leading to max(1/CoT)
    for i in range(100):
        
        data = SalamandraData.from_file(filename.format(i, 'h5'))
        with open(filename.format(i, 'pickle'), 'rb') as param_file:
            parameters = pickle.load(param_file)
        
        drive = parameters.drive
        phase_lag = parameters.phase_biases[3]
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

    global optimal_phase_offset
    optimal_phase_offset = best_params[0]
    # Plot the heat maps of the results of the grid search
    plt.figure("Figure_3.1 - grid search results", figsize=(14, 12))

    ## forward_Speed
    plt.subplot(2,2,1)
    plot_2d(forward_speeds, ['Phase offset [rad]', 'Drive', 'Forward_Speed [m/s]'], n_data=10, log=False)
    
    ## lateral_speed
    plt.subplot(2,2,2)
    plot_2d(lateral_speeds, ['Phase offset [rad]', 'Drive', 'Later_Speed [m/s]'], n_data=10, log=False)
    
    ## Total_Torque
    plt.subplot(2,2,3)
    plot_2d(total_torques, ['Phase offset [rad]', 'Drive', 'Total Torque [Nm]'], n_data=10, log=False)

    ## Traveled_Distance
    plt.subplot(2,2,4)
    plot_2d(max_distances, ['Phase offset [rad]', 'Drive', 'Traveled Distance [m]'], n_data=10, log=False)
    plt.show()
    plt.figure("Figure_3.2 - Energy and COT grid search results", figsize=(14, 12))
    ## Energy
    plt.subplot(2,2,1)
    plot_2d(energies, ['Phase offset [rad]', 'Drive', 'Energy [J]'], n_data=10, log=False)

    # iCOT
    plt.subplot(2,2,2)
    plot_2d(inverse_CoTs, ['Phase offset [rad]', 'Drive', 'iCOT [d/E]'], n_data=10, log=False)
    plt.show()
    pylog.info('Best parameters for salamander walking locomotion: Phase offset = {} [rad], Drive = {}'.format(best_params[0], best_params[1]))

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
    print(np.mean(osc_phases[-500:,:], axis = 0))
    # Compute the phase lags along the spine (downwards here)
    phase_lags = np.diff(osc_phases[:,:16], axis = 1)
    # Remove the phase difference between oscillators 8-9 that are not coupled
    phase_lags = np.concatenate((phase_lags[:,:7], phase_lags[:,8:]), axis = 1)
    
    joints_positions = np.asanyarray(data.sensors.joints.positions_all()) 
    joints_positions[:, [1,5]] *= -1

    ## PLOT RESULTS
    plt.figure("Figure_3.1 - Spine movement walking ", figsize=(16, 14)) 
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
    stable_phase_lags = np.mean(phase_lags[-500:,:], axis = 0)
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

def exercise_3b_coordination(timestep):
    """[Project 1] Exercise 3b Limb and Spine coordination

    This exercise explores how spine amplitude affects coordination.

    Run the simulations for different walking drives and body amplitude.

    """

    osc_amps = list(np.linspace(0.0, 0.5, 5))
    
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.12],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=drive,  # An example of parameter part of the grid search
            phase_biases=[2*np.pi/8, -2*np.pi/8, np.pi, optimal_phase_offset , np.pi],
            osc_amp=amp
        )
        for drive in np.linspace(1.0, 3.0, 5)
        for amp in osc_amps
    ]

    # Grid search
    os.makedirs('./logs/exercise_3b/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_3b/simulation_{}.{}'
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


def exercise_test_walk_b(timestep):
    
    filename = './logs/exercise_3b/simulation_{}.{}'
    
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
        osc_amp = parameters.osc_amp
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
        max_distances[i]=[osc_amp,  drive, max_distance]
        # Total Torque
        sum_torques=plot_results.sum_torques(joints_torques)
        total_torques[i]=[osc_amp,  drive, sum_torques]
        # forward_speeds
        forward_speed=plot_results.compute_speed(links_positions,links_velocities)[0]
        forward_speeds[i]= [osc_amp,  drive, forward_speed]
        #later_speeds
        lateral_speed= plot_results.compute_speed(links_positions,links_velocities)[1]
        lateral_speeds[i]= [osc_amp,  drive, lateral_speed]
        
        #Energy used
        # Numerical integration with the trapeze method : np.trapz
        energy = np.sum(np.trapz(joints_velocities[:,:]*joints_torques[:,:], dx = timestep))
        energies[i] = [osc_amp, drive, energy]
        ## 1/CoT : defined here as (traveled distance)/energy
        inv_CoT = max_distance/energy
        inverse_CoTs[i] = [osc_amp, drive, inv_CoT]
        # Check if this set of parameters is better than the previous ones for maximizing 1/CoT
        if max_inv_CoT < inv_CoT:
            max_inv_CoT = inv_CoT
            best_params = [osc_amp, drive,i]

    # Plot the heat maps of the results of the grid search
    plt.figure("Figure_3.2 - grid search results", figsize=(14, 12))

    ## forward_Speed
    plt.subplot(2,2,1)
    plot_2d(forward_speeds, ['Oscillation Amplitude', 'Drive', 'Forward_Speed [m/s]'], n_data=10, log=False)
    
    ## lateral_speed
    plt.subplot(2,2,2)
    plot_2d(lateral_speeds, ['Oscillation Amplitude', 'Drive', 'Later_Speed [m/s]'], n_data=10, log=False)
    
    ## Total_Torque
    plt.subplot(2,2,3)
    plot_2d(total_torques, ['Oscillation Amplitude', 'Drive', 'Total Torque [Nm]'], n_data=10, log=False)

    ## Traveled_Distance
    plt.subplot(2,2,4)
    plot_2d(max_distances, ['Oscillation Amplitude', 'Drive', 'Traveled Distance [m]'], n_data=10, log=False)
    plt.show()
    plt.figure("Figure_3.2 - Energy and COT grid search results", figsize=(14, 12))
    ## Energy
    plt.subplot(2,2,1)
    plot_2d(energies, ['Phase offset [rad]', 'Drive', 'Energy [J]'], n_data=10, log=False)

    # iCOT
    plt.subplot(2,2,2)
    plot_2d(inverse_CoTs, ['Phase offset [rad]', 'Drive', 'iCOT [d/E]'], n_data=10, log=False)
    plt.show()
    pylog.info('Best parameters for salamander walking locomotion: Oscillation Amplitude = {} [rad], Drive = {}'.format(best_params[0], best_params[1]))

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
    plt.figure("Figure_3.2  - Spine movement walking ", figsize=(16, 14)) 
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
    stable_phase_lags = np.mean(phase_lags[-500:,:], axis = 0)
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

if __name__ == '__main__':
    exercise_3a_coordination(timestep=1e-2)
    exercise_test_walk(timestep=1e-2)
    exercise_3b_coordination(timestep=1e-2)
    exercise_test_walk_b(timestep=1e-2)
    
