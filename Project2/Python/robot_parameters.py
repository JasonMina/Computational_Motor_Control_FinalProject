"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.sim_parameters=parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([self.n_oscillators,
            self.n_oscillators,
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.feedback_gains_swim = np.zeros(self.n_oscillators)
        self.feedback_gains_walk = np.zeros(self.n_oscillators)

        # gains for final motor output
        self.position_body_gain = parameters.position_body_gain
        self.position_limb_gain = parameters.position_limb_gain
        # for sensory feedback
        self.feedback = parameters.feedback
        self.duration = parameters.duration

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # phi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def step(self, iteration, salamandra_data):
        """Step function called at each iteration

        Parameters
        ----------

        salamanra_data: salamandra_simulation/data.py::SalamandraData
            Contains the robot data, including network and sensors.

        gps (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 representing the GPS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        cond_transition=self.sim_parameters.cond_transition
        if cond_transition:
            contact_forces=np.max([np.mean(np.asarray(salamandra_data.sensors.contacts.reaction_all(i)[iteration-2:iteration+2,2])) for i in range(0,4)])
            if contact_forces >3:
                self.sim_parameters.drive = 2.8
            else:
                self.sim_parameters.drive = 4.5            
            self.set_frequencies(self.sim_parameters)
            self.set_nominal_amplitudes(self.sim_parameters)
        

    def set_frequencies(self, parameters):
        """Set frequencies"""
        drive=parameters.drive
        
        self.nominal_amplitudes=np.zeros_like(self.nominal_amplitudes)
        turn=self.sim_parameters.turn
        
        body_eq = parameters.cv1[0]*drive + parameters.cv0[0]
        limb_eq = parameters.cv1[1]*drive + parameters.cv0[1]
            
        if drive<=parameters.dhigh[0] and drive >= parameters.dlow[0]:
            self.freqs[0:16].fill (body_eq)
        else:
            self.freqs[0:16].fill(parameters.vsat[0])

        if drive<=parameters.dhigh[1] and drive >= parameters.dlow[1]:
            self.freqs[16:20].fill(limb_eq)
        else:
            self.freqs[16:20].fill(parameters.vsat[1])


        """ pylog.warning(
            'Set the frequencies of the spinal and limb oscillators as a function of the drive') """

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""

        self.coupling_weights = np.zeros([self.n_oscillators,self.n_oscillators,])
        
        if parameters.coupling == True:
         # upwards
            upwards = parameters.coupling_weights[0]
            self.coupling_weights[1:8, 0:7] = np.diag([upwards] * 7)
            self.coupling_weights[9:16, 8:15] = np.diag([upwards] * 7)

            # downwards
            downwards = parameters.coupling_weights[1]
            self.coupling_weights[0:7, 1:8] = np.diag([downwards] * 7)
            self.coupling_weights[8:15, 9:16] = np.diag([downwards] * 7)

            # lateral
            lateral = parameters.coupling_weights[2]
            self.coupling_weights[0:8, 8:16] = np.diag([lateral] * 8)
            self.coupling_weights[8:16, 0:8] = np.diag([lateral] * 8)
            
            # limb-body
            limb_body = parameters.coupling_weights[3]
            self.coupling_weights[0:4, 16] = limb_body
            self.coupling_weights[4:8, 18] = limb_body
            self.coupling_weights[8:12, 17] = limb_body
            self.coupling_weights[12:16, 19] = limb_body
            
            # limb-limb
            limb_limb = parameters.coupling_weights[4]
            self.coupling_weights[16, 17] = limb_limb
            self.coupling_weights[16, 18] = limb_limb

            self.coupling_weights[17, 16] = limb_limb
            self.coupling_weights[17, 19] = limb_limb

            self.coupling_weights[18, 16] = limb_limb
            self.coupling_weights[18, 19] = limb_limb

            self.coupling_weights[19, 17] = limb_limb
            self.coupling_weights[19, 18] = limb_limb


    def set_phase_bias(self, parameters):
        """Set phase bias"""

        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        # upwards
        upwards = parameters.phase_biases[0]
        self.phase_bias[1:8, 0:7] = np.diag([upwards] * 7)
        self.phase_bias[9:16, 8:15] = np.diag([upwards] * 7)

        # downwards
        downwards = parameters.phase_biases[1]
        self.phase_bias[0:7, 1:8] = np.diag([downwards] * 7)
        self.phase_bias[8:15, 9:16] = np.diag([downwards] * 7)

        # lateral
        lateral = parameters.phase_biases[2]
        self.phase_bias[0:8, 8:16] = np.diag([lateral] * 8)
        self.phase_bias[8:16, 0:8] = np.diag([lateral] * 8)
        
        # limb-body
        limb_body = parameters.phase_biases[3]
        self.phase_bias[0:4, 16] = limb_body
        self.phase_bias[4:8, 18] = limb_body
        self.phase_bias[8:12, 17] = limb_body
        self.phase_bias[12:16, 19] = limb_body
        
        # limb-limb
        limb_limb = parameters.phase_biases[4]
        self.phase_bias[16, 17] = limb_limb
        self.phase_bias[16, 18] = limb_limb

        self.phase_bias[17, 16] = limb_limb
        self.phase_bias[17, 19] = limb_limb

        self.phase_bias[18, 16] = limb_limb
        self.phase_bias[18, 19] = limb_limb

        self.phase_bias[19, 17] = limb_limb
        self.phase_bias[19, 18] = limb_limb
    
        #pylog.warning('Phase bias must be set')

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates[0:16].fill(parameters.rates[0])
        self.rates[16:20].fill(parameters.rates[1])


    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        drive=parameters.drive
        self.nominal_amplitudes=np.zeros_like(self.nominal_amplitudes)
        turn=self.sim_parameters.turn
        self.sim_parameters.rigid_spine
        
        body_eq = parameters.cr1[0]*drive + parameters.cr0[0]
        limb_eq = parameters.cr1[1]*drive + parameters.cr0[1]

        body_eq_asym = parameters.cr1[0]*(drive+np.abs(turn)) + parameters.cr0[0]
        limb_eq_asym = parameters.cr1[1]*(drive+np.abs(turn)) + parameters.cr0[1]
            
        if drive<=parameters.dhigh[0] and drive >= parameters.dlow[0]:
            
            if np.sign(turn)<0:
                self.nominal_amplitudes[0:8].fill (body_eq)
                self.nominal_amplitudes[8:16].fill (body_eq_asym)

            elif np.sign(turn)>0:
                self.nominal_amplitudes[0:8].fill (body_eq_asym)
                self.nominal_amplitudes[8:16].fill (body_eq)

            else:      
                if self.sim_parameters.rigid_spine:
                    self.nominal_amplitudes[0:16].fill (parameters.rsat[0])
                else:
                    self.nominal_amplitudes[0:16].fill (body_eq)
        else:
            self.nominal_amplitudes[0:16].fill(parameters.rsat[0])

        if drive<=parameters.dhigh[1] and drive >= parameters.dlow[1]:
            if np.sign(turn)<0:
                self.nominal_amplitudes[16::2].fill (limb_eq)
                self.nominal_amplitudes[17::2].fill (limb_eq_asym)

            elif np.sign(turn)>0:
                self.nominal_amplitudes[16::2].fill (limb_eq_asym)
                self.nominal_amplitudes[17::2].fill (limb_eq)

            else:
                self.nominal_amplitudes[16:20].fill(limb_eq)
        else:
            self.nominal_amplitudes[16:20].fill(parameters.rsat[1])

