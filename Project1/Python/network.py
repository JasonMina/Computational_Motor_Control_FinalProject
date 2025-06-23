"""Oscillator network ODE"""

import numpy as np
from scipy.integrate import ode
from robot_parameters import RobotParameters


def network_ode(_time, state, robot_parameters, loads, contact_sens):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters
    loads: <np.array>
        The lateral forces applied to the body links

    Returns
    -------
    dstate: <np.array>
        Returns derivative of state (phases and amplitudes)

    """
    n_oscillators = robot_parameters.n_oscillators
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]
    freqs=robot_parameters.freqs
    weights=robot_parameters.coupling_weights
    nominal_amplitudes = robot_parameters.nominal_amplitudes

    biases=robot_parameters.phase_bias
    rates=robot_parameters.rates

    dtheta = 2* np.pi* freqs
    dr = rates * (nominal_amplitudes - amplitudes)

    for i in range(0, n_oscillators):

        dtheta[i] += (np.multiply(amplitudes,(weights[i,:])).T @ np.sin(phases-phases[i]-biases[i,:]))
        
    return np.concatenate([dtheta, dr])


def motor_output(phases, amplitudes, iteration, walk_back):
    """Motor output

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    motor_outputs: <np.array>
        Motor outputs for joint in the system.

    """
    motor_outputs = np.zeros(16)

    #spine joint outputs
    motor_outputs[:8] = np.multiply(amplitudes[:8],(1+np.cos(phases[:8]))) - np.multiply(amplitudes[8:16], (1+np.cos(phases[8:16])))

    #shoulder joint ouput
    if walk_back:
        motor_outputs[8::2] = np.multiply(amplitudes[16:20], np.sin(phases[16:20]))

        #Wrist joint output
        motor_outputs[9::2] = np.multiply(amplitudes[16:20], np.cos(phases[16:20]))

    else:
        motor_outputs[8::2] = np.multiply(amplitudes[16:20], np.cos(phases[16:20]))

        #Wrist joint output
        motor_outputs[9::2] = np.multiply(amplitudes[16:20], np.sin(phases[16:20]))
    
    return motor_outputs


class SalamandraNetwork:
    """Salamandra oscillator network"""

    def __init__(self, sim_parameters, n_iterations, state):
        super().__init__()
        self.n_iterations = n_iterations
        self.sim_parameters=sim_parameters
        # States
        self.state = state
        # Parameters
        self.robot_parameters = RobotParameters(sim_parameters)
        # Set initial state
        # Replace your oscillator phases here
        self.state.set_phases(
            iteration=0,
            value=1e-4*np.random.rand(self.robot_parameters.n_oscillators),
        )
        # Set solver
        self.solver = ode(f=network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state.array[0], t=0.0)

    def step(self, iteration, time, timestep, loads=None, contact_sens=None):
        """Step"""
        if loads is None:
            loads = np.zeros(self.robot_parameters.n_joints)
        if iteration + 1 >= self.n_iterations:
            return
        self.solver.set_f_params(self.robot_parameters, loads, contact_sens)
        self.state.array[iteration+1, :] = self.solver.integrate(time+timestep)

    def outputs(self, iteration=None):
        """Oscillator outputs"""
        # Implement equation here
        state=self.state
        phases=state.phases(iteration)
        amplitudes=state.amplitudes(iteration)
        
        outputs = np.multiply(amplitudes, (1+ np.cos(phases)))
        return outputs

    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        oscillator_output = motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
            walk_back=self.sim_parameters.walk_back
        )
        return oscillator_output
