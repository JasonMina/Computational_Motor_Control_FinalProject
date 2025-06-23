"""Network controller"""

import numpy as np
from farms_core.model.control import AnimatController


class SalamandraController(AnimatController):
    """Salamandra controller"""

    def __init__(self, joints_names, animat_data, network):
        super().__init__(
            joints_names=[joints_names, [], []],
            max_torques=[np.ones(len(joints_names)), [], []],
        )
        self.network = network
        self.animat_data = animat_data
        self.contact_legs = np.zeros((10, 4))

    def step(self, iteration, time, timestep):
        """Control step"""
        # Lateral hydrodynamic forces
        loads = -1 * np.concatenate([
            self.animat_data.sensors.xfrc.array[
                iteration,
                0:self.network.robot_parameters['n_body_joints'],
                1
            ],
            np.zeros(self.network.robot_parameters['n_legs_joints'])
        ])
        # GPS positions
        self.network.robot_parameters.step(
            iteration=iteration,
            salamandra_data=self.animat_data,
        )

        contact_sens = np.asarray(
            self.animat_data.sensors.contacts.array[
                iteration,
                :,
                2
            ]
        )

        # Network integration step
        self.network.step(iteration, time, timestep, loads, contact_sens)

    def positions(self, iteration, time, timestep):
        """Postions"""
        position_values = self.network.get_motor_position_output(iteration)
        # left first & second motor -> 11, 12
        first = [8, 10, 12, 14]
        second = [9, 11, 13, 15]
        left = [8, 9, 12, 13]
        right = [10, 11, 14, 15]

        position_values[1] *= -1
        position_values[5] *= -1
        position_values[0:8] *= self.network.robot_parameters['position_body_gain']
        position_values[8:] *= self.network.robot_parameters['position_limb_gain']

        # print("positions")
        # print("pos: 8:{}, 9:{}".format(position_values[8], position_values[9]))
        # print("pos: 10:{}, 11:{}".format(position_values[10],position_values[11]))
        # print("pos: 12:{}, 13:{}".format(position_values[12],position_values[13]))
        # print("pos: 14:{}, 15:{}".format(position_values[14],position_values[15]))

        # # forward zero for each first motor
        # position_values[8:] *= 0.0
        # position_values[first] *= 2
        position_values[second] *= -1  # forward walking
        # position_values[second] *= 1 # backward walking

        # # add offset
        position_values[8] += np.pi/8
        position_values[10] += np.pi/8
        position_values[12] += - np.pi/8
        position_values[14] += - np.pi/8
        position_values[second] += np.pi/8
        position_values[right] *= -1

        # # correcting the signs for right
        # # position_values[8:] *= -1
        # # position_values[second] *= -0
        # position_values[right] *= -1
        # # position_values[left] *= -1

        # # position_values[8:12] = 0 # debugguing hind zero
        # position_values[9] *= -1
        # position_values[11] *= -1
        # position_values[13] *= -1
        # position_values[15] *= -1

        # position_values[8] *= 2
        # position_values[10] *= 2
        # position_values[12] *= 2
        # position_values[14] *= 2

        # position_values[:] =0
        return dict(zip(
            self.joints_names[0],
            position_values,
        ))

