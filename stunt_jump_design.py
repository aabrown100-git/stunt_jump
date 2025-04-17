from stunt_jump_functions import *
import numpy as np
################################################################################
# ------------------------------------------------------------------------------
# ---------------------------- SIMULATION PARAMETERS ---------------------------
# ------------------------------------------------------------------------------
# This controls how strong the ramp friction is
friction_factor = 0.1

# This is the starting position from the start of the ramp
p_initial = 0.  # inches
p_initial = p_initial * convert_in_to_m

# Animation flag
animation_flag = False

################################################################################
# ------------------------------------------------------------------------------
# ---------------------------- RAMP SHAPE --------------------------------------
# ------------------------------------------------------------------------------
# Define the location of ramp anchors, which serve as spine nodes
anchor_locations = np.array([
    [0, 5], # feet
    [1, 2],
    [2, 2.5],
]) * 12

# Define locations of rings
ring_1 = [5, 1.75] # ft
ring_2 = [6, 1.6] # ft
ring_3 = [7, 0.9] # ft

################################################################################
# ------------------------------------------------------------------------------
# ------------------------ RUN SIMULATION --------------------------------------
# ------------------------------------------------------------------------------

# Create simulation parameters
sim_params = {
    'friction_factor': friction_factor,
    'p_initial': p_initial,
    'ramp_type': 'jump',
    'animation_flag': animation_flag,
    'anchor_locations': anchor_locations,
    'ring_1': ring_1,
    'ring_2': ring_2,
    'ring_3': ring_3,
    'save_inverted_image': True
}

# Run simulation
results = simulate_stunt_jump(sim_params)