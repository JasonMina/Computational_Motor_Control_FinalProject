"""[Project1] Exercise 1: Implement & run network without MuJoCo"""

import time
import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork
from matplotlib.gridspec import GridSpec


def run_network(duration, update=True, drive=5, timestep=1e-2):
    """ Run network without MuJoCo and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        True: use the prescribed drive parameter, False: update the drive during the simulation
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    cond_coupling=True
    
    for t in range(0,2):
        #Iterate two times, one with coupling, one without
        drive=0
        if t==1:
            cond_coupling=False

        sim_parameters = SimulationParameters(
            drive=drive,
            amplitude_gradient=None,
            phase_lag_body=None,
            turn=0, 
            coupling=cond_coupling
        )

        drive=np.linspace(0,6, n_iterations)
        state = SalamandraState.salamandra_robot(n_iterations)
        network = SalamandraNetwork(sim_parameters, n_iterations, state)

        # Logs
        phases_log = np.zeros([
            n_iterations,
            len(network.state.phases(iteration=0))
        ])
        phases_log[0, :] = network.state.phases(iteration=0)
        amplitudes_log = np.zeros([
            n_iterations,
            len(network.state.amplitudes(iteration=0))
        ])
        amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
        freqs_log = np.zeros([
            n_iterations,
            len(network.robot_parameters.freqs)
        ])
        
        freqs_log[0, :] = network.robot_parameters.freqs

        nom_amps_log = np.zeros([
            n_iterations,
            len(network.robot_parameters.nominal_amplitudes)
        ])
        
        nom_amps_log[0, :] = network.robot_parameters.nominal_amplitudes
        
        outputs_log = np.zeros([
            n_iterations,
            len(network.get_motor_position_output(iteration=0))
        ])
        outputs_log[0, :] = network.get_motor_position_output(iteration=0)

        oscillator_outputs_log = np.zeros([
            n_iterations,
            len(network.outputs(iteration=0))
        ])
        oscillator_outputs_log[0, :] = network.outputs(iteration=0)
        
        # Run network ODE and log data
        if t==0:
            tic = time.time()
        for i, time0 in enumerate(times[1:]):
            if update:
                network.robot_parameters.update(
                    SimulationParameters(drive=drive[i], coupling=cond_coupling)
                )
            network.step(i, time0, timestep)
            phases_log[i+1, :] = network.state.phases(iteration=i+1)
            amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
            outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
            freqs_log[i+1, :] = network.robot_parameters.freqs
            nom_amps_log[i+1, :] = network.robot_parameters.nominal_amplitudes
            oscillator_outputs_log[i+1, :] = network.outputs(iteration=i+1)

        if t==0:
            toc = time.time()

        if t==0:
            #The case where there is coupling
            n_rows=4
            n_columns=1

            fig, ax = plt.subplots(n_rows, n_columns, figsize=(15,15))

            #Plotting the outputs for the body&limb oscillators
            for i in range(0, 8):
                ax[0].plot(times, oscillator_outputs_log[:,i]-(i*2), label=('x'+str(i+1)))

            ax[1].plot(times, oscillator_outputs_log[:,16], label=('x'+str(16)))
            ax[1].plot(times, oscillator_outputs_log[:,18]-3, label=('x'+str(18)))
            ax[1].set_ylim(-5,1.5)

            ax[0].set_yticks([])
            ax[1].set_yticks([])

            #Plotting the instantaneous frequencies
            dt = np.diff(times)
            dphases = np.diff(phases_log, axis=0)
            freqs_from_phases = dphases / dt[:, None]

            for i in range(freqs_from_phases.shape[1]):
                ax[2].plot(times[1:], freqs_from_phases[:, i])

            font_size=15
            #Plotting the drive vs time
            ax[3].plot(times, drive)
            ax[3].text(22,drive[int(22/timestep)], "Swimming", fontsize=font_size)
            ax[3].text(12,drive[int(12/timestep)], "Walking", fontsize=font_size)

            #Setting x,y labels
            ax[0].set_ylabel('x body', fontsize=font_size)
            ax[1].set_ylabel('x limb', fontsize=font_size)
            ax[2].set_ylabel('Freq. [Hz]', fontsize=font_size)
            ax[3].set_ylabel('Drive d', fontsize=font_size)
            ax[3].set_xlabel('Time [s]', fontsize=font_size)

            #Showing legend for all plots
            for i in range(0,n_rows):
                ax[i].axvline(x=20, ls="--")
                ax[i].legend()

            fig.suptitle("Switching from walking to swimming; activity of the CPG model when the drive signal is progressively increased."
                         + "\n(Coupled case)", fontsize=font_size)

        if t == 1:
            #The case where there isnt't coupling

            fig = plt.figure(figsize=(15, 8))

            gs = GridSpec(4, 2)
            
            #Plotting the intrinsic frequencies vs drive
            ax0 = plt.subplot(gs[0:2,0])
            ax0.plot(drive, freqs_log[:,0], label="Body")
            ax0.plot(drive, freqs_log[:,16], label="Limb", ls="--")
            ax0.set_ylabel("ν [Hz]", fontsize=font_size)
            ax0.legend()
            
            #Plotting the nominal amplitudes vs drive
            ax1 = plt.subplot(gs[2:4,0])
            ax1.plot(drive, nom_amps_log[:,0], label="Body")
            ax1.plot(drive, nom_amps_log[:,16], label="Limb", ls="--")
            ax1.set_ylabel("R", fontsize=font_size)
            ax1.set_xlabel("drive", fontsize=font_size)
            ax1.legend()

            #Plotting the outputs of body & limb vs time in the not coupled case
            ax2 = plt.subplot(gs[0:1,1])
            ax2.plot(times, oscillator_outputs_log[:,0], label="Body")
            ax2.plot(times, oscillator_outputs_log[:,16]-2, label="Limb", ls="--")
            ax2.set_ylabel("x", fontsize=font_size)
            ax2.set_yticks([])
            ax2.legend()
            
            #Plotting the instantaneous freqs. of body & limb vs time in the not coupled case
            ax3 = plt.subplot(gs[1:2,1])
            dt = np.diff(times)
            dphases = np.diff(phases_log, axis=0)
            freqs_from_phases = dphases / dt[:, None]

            ax3.plot(times[1:], freqs_from_phases[:, 0], label="Body")
            ax3.plot(times[1:], freqs_from_phases[:, 16], label="Limb", ls="--")
            ax3.set_ylabel("Freq. [Hz]", fontsize=font_size)
            ax3.legend()

            #Plotting the amplitudes of body & limb vs time in the not coupled case
            ax4 = plt.subplot(gs[2:3,1])
            ax4.plot(times, amplitudes_log[:,0], label="Body")
            ax4.plot(times, amplitudes_log[:,16], label="Limb", ls="--")
            ax4.set_ylabel("r", fontsize=font_size)
            ax4.legend()

            #Plotting drive vs time in the uncoupled case
            ax5 = plt.subplot(gs[3:4,1])
            ax5.plot(times, drive)
            ax5.set_ylabel("d (drive)", fontsize=font_size)
            ax5.set_xlabel("Time [s]", fontsize=font_size)
            fig.suptitle("Saturation function and oscillations in isolated (i.e., uncoupled) body and limb oscillators.", fontsize=font_size)

            plt.tight_layout()
        
        # Network performance
        pylog.info('Time to run simulation for {} steps: {} [s]'.format(
            n_iterations,
            toc - tic
        ))

        # Implement plots of network results
        pylog.warning('Implement plots')

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    run_network(duration=40)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
    return

if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())