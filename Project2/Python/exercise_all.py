"""[Project1] Script to call all exercises"""

from farms_core import pylog

from exercise_p6 import (
    exercise_6a_phase_relation,
    exercise_6b_tegotae_limbs,
    exercise_6c_tegotae_spine,
    exercise_6d_open_vs_closed,
)


def exercise_all(arguments):
    """Run all exercises"""

    verbose = 'not_verbose' not in arguments

    if not verbose:
        pylog.set_level('warning')

    # Timestep
    timestep = 1e-2
    if '6a' in arguments:
        exercise_6a_phase_relation(timestep)
    if '6b' in arguments:
        exercise_6b_tegotae_limbs(timestep)
    if '6c' in arguments:
        exercise_6c_tegotae_spine(timestep)
    if '6d' in arguments:
        exercise_6d_open_vs_closed(timestep)

    if not verbose:
        pylog.set_level('debug')


if __name__ == '__main__':
    exercises = ['6a', '6b', '6c', '6d']
    exercise_all(arguments=exercises)

