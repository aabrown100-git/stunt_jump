# ------------------------------------------------------------------------------
# ------------------------------------ SETUP -----------------------------------
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from PIL import Image, ImageOps
from stunt_jump_functions import *

################################################################################
# ------------------------------------------------------------------------------
# ---------------------------- SIMULATION PARAMETERS ---------------------------
# ------------------------------------------------------------------------------
# This controls how strong the ramp friction is
friction_factor = 0.1

################################################################################

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------ BEGIN MAIN SCRIPT -----------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Initial track position
p_initial = 0.  # inches
p_initial = p_initial * convert_in_to_m

# Simulation mode (halfpipe vs jump)
ramp_type = 'jump'

# Animation flag
animation_flag = False

# ------------------------------------------------------------------------------
# ---------------------------- RAMP SHAPE --------------------------------------
# ------------------------------------------------------------------------------
# Define the ramp shape using control points for a cubic spine interpolation.
# Define the location of ramp anchors, which serve as spine nodes
anchor_locations = np.array([
    [0, 5], # feet
    [1, 2],
    [2, 2.5],
]) * 12

# Extract positions of spline nodes
xnodes = np.array(anchor_locations[:,0])
ynodes = np.array(anchor_locations[:,1])

# Create a cubic spline between anchor locations
cs = CubicSpline(xnodes, ynodes, bc_type = 'not-a-knot')
xs = np.linspace(np.min(xnodes), np.max(xnodes), 1000)
ys = cs(xs)

# Convert to meters
xnodes = xnodes * convert_in_to_m # m
ynodes = ynodes * convert_in_to_m # m
xs = xs * convert_in_to_m
ys = ys * convert_in_to_m

# ------------------------------------------------------------------------------
# ---------------------------- RING LOCATIONS ----------------------------------
# ------------------------------------------------------------------------------

# Define locations of rings
ring_1 = [5, 1.75] # ft
ring_2 = [6, 1.6] # ft
ring_3 = [7, 0.9] # ft
# ------------------------------------------------------------------------------
# ------------------------ RUN SIMULATION --------------------------------------
# ------------------------------------------------------------------------------

# Create simulation parameters
sim_params = {
    'friction_factor': friction_factor,
    'p_initial': p_initial,
    'ramp_type': ramp_type,
    'animation_flag': animation_flag,
    'anchor_locations': anchor_locations,
    'ring_1': ring_1,
    'ring_2': ring_2,
    'ring_3': ring_3,
    'save_inverted_image': True
}

# Run simulation
results = simulate_stunt_jump(sim_params)
