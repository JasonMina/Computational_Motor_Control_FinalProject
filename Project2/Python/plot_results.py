"""Plot results"""

import pickle
import numpy as np
from requests import head
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from network import motor_output
import matplotlib.colors as colors
import math

# TODO CMC2023: This files needs to be cleaned and correct before final project


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=['x', 'y', 'z'][i])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid(True)


def plot_trajectory(link_data, label=None, color=None):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 1], label=label, color=color)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], 'r.')
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='none',
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


def compute_speed(links_positions, links_vel, nsteps_considered=200):
    '''
    Computes the axial and lateral speed based on the PCA of the links positions
    '''

    links_pos_xy = links_positions[-nsteps_considered:, :, :2]
    joints_vel_xy = links_vel[-nsteps_considered:, :, :2]
    time_idx = links_pos_xy.shape[0]

    speed_forward = []
    speed_lateral = []
    com_pos = []

    for idx in range(time_idx):
        x = links_pos_xy[idx, :9, 0]
        y = links_pos_xy[idx, :9, 1]

        pheadtail = links_pos_xy[idx][0]-links_pos_xy[idx][8]  # head - tail
        pcom_xy = np.mean(links_pos_xy[idx, :9, :], axis=0)
        vcom_xy = np.mean(joints_vel_xy[idx], axis=0)

        covmat = np.cov([x, y])
        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index = np.argmax(eig_values)
        largest_eig_vec = eig_vecs[:, largest_index]

        ht_direction = np.sign(np.dot(pheadtail, largest_eig_vec))
        largest_eig_vec = ht_direction * largest_eig_vec

        v_com_forward_proj = np.dot(vcom_xy, largest_eig_vec)

        left_pointing_vec = np.cross(
            [0, 0, 1],
            [largest_eig_vec[0], largest_eig_vec[1], 0]
        )[:2]

        v_com_lateral_proj = np.dot(vcom_xy, left_pointing_vec)

        com_pos.append(pcom_xy)
        speed_forward.append(v_com_forward_proj)
        speed_lateral.append(v_com_lateral_proj)

    return speed_forward, speed_lateral

#moving average filter
def smooth_data(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    smoothed_data = cumsum / window_size
    return smoothed_data


def main(filename_h5, filename_pickle, plot=True):
    """Main"""
    # Load data
    data = SalamandraData.from_file(filename_h5)
    with open(filename_pickle, 'rb') as param_file:
        parameters = pickle.load(param_file)
    timestep = data.timestep
    n_iterations = np.shape(data.sensors.links.array)[0]
    times = np.arange(
        start=0,
        stop=timestep*n_iterations,
        step=timestep,
    )
    timestep = times[1] - times[0]
    #amplitudes = parameters.amplitudes
    phase_lag_body = parameters.phase_lag_body
    osc_phases = data.state.phases()
    osc_amplitudes = data.state.amplitudes()
    links_positions = data.sensors.links.urdf_positions()
    links_positions=np.asarray(links_positions)
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 8, :]
    joints_positions = data.sensors.joints.positions_all()
    joints_velocities = data.sensors.joints.velocities_all()
    joints_torques = data.sensors.joints.motor_torques_all()
    links_velocities = data.sensors.links.com_lin_velocities()
    links_velocities=np.asarray(links_velocities)
    duration = parameters.duration * 100
    # Notes:
    # For the links arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]
    speed = compute_speed(links_positions, links_velocities, duration)
    forward_speed = speed[0]
    lateral_speed = speed[1]
    phases = data.state.phases()
    average_fwd_speed=np.mean(forward_speed)
    average_lat_speed=np.mean(lateral_speed)

    # Plot data
    head_positions = np.asarray(head_positions)
    #plt.figure('Positions')
    #plot_positions(times, head_positions)
    plt.figure('Trajectory')
    plot_trajectory(head_positions)

    plt.figure('Forward speed')
    plt.plot(times, forward_speed)
    plt.xlabel('Times [s]')
    plt.ylabel('Forward speed [m/s]')
    plt.annotate(f"Average forward speed {average_fwd_speed:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')

    plt.figure('Lateral speed')
    plt.plot(times, lateral_speed)
    plt.xlabel('Times [s]')
    plt.ylabel('Lateral speed [m/s]')
    plt.annotate(f"Average lateral speed {average_lat_speed:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')

    fig, axs =plt.subplots(4,1, figsize=(12,9))
    for i in range(4):
        contact_forces=np.asarray(data.sensors.contacts.reaction_all(i)[:,2])
        contact_forces=smooth_data(contact_forces, 20) #smoothens the data by moving average filter
        contact_forces[contact_forces <= 7] = 0  #lower threshold
        contact_forces[contact_forces >= 20] = 0 #upper threshold
        phases_pi=np.cos(phases[:, 16+i])
        axsi = axs[i].twinx()
        if i ==0:
            axs[i].plot(phases[:, 16+i], contact_forces, color = 'blue', label="Contact forces (on z-axis) for each limb")
            axsi.plot(phases[:, 16+i], phases_pi, color = 'red', label="Cosinus of the phase")
        else:
            axs[i].plot(phases[:, 16+i], contact_forces, color = 'blue')
            axsi.plot(phases[:, 16+i], phases_pi, color = 'red')
        
        axs[i].set_title('Limb ' + str(i+1) + " with weight = " + str(parameters.weights_contact_limb_i))

    axs[2].text(x = 107, y = 5, s = "Cosinus of the phase", rotation = 3*180/2, fontsize=13)
    fig.legend(loc = 'upper right')
    fig.supxlabel('Phases of each limb scaled by pi [rad]')
    fig.supylabel('Contact Forces [N]')
    fig.tight_layout()

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=save_plots())

