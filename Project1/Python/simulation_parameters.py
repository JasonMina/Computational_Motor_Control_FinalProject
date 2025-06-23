"""Simulation parameters"""
import numpy as np

class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs): 
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 4
        self.duration = 30
        self.initial_phases = None
        self.position_body_gain = 0.6  # default do not change
        self.position_limb_gain = 1  # default do not change
        self.phase_lag_body = None
        self.amplitude_gradient = None
        self.drive=0
        self.turn=0
        self.walk_back=False
        self.cond_transition=False
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        self.phase_biases=[2*np.pi/8, -2*np.pi/8, np.pi, np.pi, np.pi]
        self.osc_amp=None
        self.coupling_weights=[10, 10, 10, 30, 10]
        self.rates=[20,20]
        self.coupling=True

        #Intrinsic freqs and nominal amplitude parameters, first element for body, second element for limb osc.
        self.dlow = [1,1]
        self.dhigh = [5,3]
        self.cv1 = [0.2, 0.2]
        self.cv0 = [0.3, 0]
        self.cr1 = [0.065, 0.131]
        self.cr0 = [0.196, 0.131]
        self.vsat = [0,0]
        self.rsat = [0,0]
        
        # Disruptions
        self.set_seed = False
        self.randseed = 0
        self.n_disruption_couplings = 0
        self.n_disruption_oscillators = 0
        self.n_disruption_sensors = 0

        # Tegotae
        self.weights_contact_body = 0.0
        self.weights_contact_limb_i = 0.0
        self.weights_contact_limb_c = 0.0

        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

