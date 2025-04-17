from stunt_jump_functions import *
import numpy as np
################################################################################

# ------------------------------------------------------------------------------
# ---------------------------- SIMULATION PARAMETERS ---------------------------
# ------------------------------------------------------------------------------
# This controls how strong the ramp friction is
friction_factor = 0.045

# This is the starting position from the start of the ramp
p_initial = 0.; # inches
p_initial = p_initial * convert_in_to_m

# Simulation mode (halfpipe vs jump)
ramp_type = 'halfpipe'

# Animation flag
animation_flag = True
################################################################################

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------ BEGIN MAIN SCRIPT -----------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------------------------- RAMP SHAPE --------------------------------------
# ------------------------------------------------------------------------------
# Define a halfpipe ramp shape using control points for a cubic spine interpolation.

# Define the location of ramp anchors, which serve as spine nodes
anchor_locations = np.array([
    [0, 1], # feet
    [1, 0],
    [2, 1],
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
xs = xs * convert_in_to_m # m
ys = ys * convert_in_to_m # m

# ------------------------------------------------------------------------------
# ---------------------------- RING LOCATIONS ----------------------------------
# ------------------------------------------------------------------------------

# Define locations of rings
ring_1 = [5, 1.75] # ft
ring_2 = [6, 1.6] # ft
ring_3 = [7, 0.9] # ft


# ------------------------------------------------------------------------------
# ---------------------------- PATH LENGTH AND MAPS ----------------------------
# ------------------------------------------------------------------------------
path_maps = compute_path_length_and_maps(xs, ys, long_track_segment_in, convert_in_to_m)

ps = path_maps["ps"]
p2x = path_maps["p2x"]
p2y = path_maps["p2y"]
p2k = path_maps["p2k"]
p2kappa = path_maps["p2kappa"]
truncated_length = path_maps["truncated_length"]
n_long_track_segments = path_maps["n_long_track_segments"]


# ------------------------------------------------------------------------------
# ---------------------------- SIMULATION --------------------------------------
# ------------------------------------------------------------------------------

# ------------------------- INTEGRATE RAMP ODE ---------------------------------
# Integrate the ODE for the car on the ramp. The ODE is defined in ramp_ode_fun.
# The integration stops when the car reaches the end of the track or comes to rest.

# Define the ramp parameters
ramp_params = {
    'friction_factor': friction_factor,
    'p_initial': p_initial,
    'ps': ps,
    'p2x': p2x,
    'p2y': p2y,
    'p2k': p2k,
    'p2kappa': p2kappa,
}

ramp_result = integrate_ramp_ode(
    ramp_params=ramp_params,
    ramp_ode_fun=ramp_ode_fun,
    end_track=end_track,
    at_rest=at_rest,
)

x_track = ramp_result["x_track"]
y_track = ramp_result["y_track"]
t_track = ramp_result["t_track"]
end_track_flag = ramp_result["end_track_flag"]
at_rest_flag = ramp_result["at_rest_flag"]
t_to_rest = ramp_result["t_to_rest"]
x_end = ramp_result["x_end"]
y_end = ramp_result["y_end"]
speed_end = ramp_result["speed_end"]
theta_end = ramp_result["theta_end"]


# ------------------------- INTEGRATE BALLISTIC ODE -----------------------------
# If we reached the end of the track, then integrate the ballistic ODE to get
# the motion of the car in the air. If we did not reach the end of the track, skip.
# The ODE is defined in ballistic_ode_fun.

ballistic_result = integrate_ballistic_ode(
    end_track_flag=end_track_flag,
    x_end=x_end,
    y_end=y_end,
    speed_end=speed_end,
    theta_end=theta_end,
    x_track=x_track,
    y_track=y_track,
    t_track=t_track,
    ballistic_ode_fun=ballistic_ode_fun,
    hit_target=hit_target,
)

x_full = ballistic_result["x_full"]
y_full = ballistic_result["y_full"]
t_full = ballistic_result["t_full"]

# ------------------------------------------------------------------------------
# ---------------------------- PLOTTING ----------------------------------------
# ------------------------------------------------------------------------------

# Plot/animate motion of car on track (and in air, if applicable)
plot_and_animate_car_motion(
    xnodes=xnodes,
    ynodes=ynodes,
    xs=xs,
    ys=ys,
    x_full=x_full,
    y_full=y_full,
    ps=ps,
    p2k=p2k,
    n_long_track_segments=n_long_track_segments,
    long_track_segment_in=long_track_segment_in,
    convert_in_to_m=convert_in_to_m,
    ring_1=ring_1,
    ring_2=ring_2,
    ring_3=ring_3,
    end_track_flag=end_track_flag,
    x_end=x_end,
    y_end=y_end,
    ramp_type=ramp_type,
    at_rest_flag=at_rest_flag,
    t_to_rest=t_to_rest,
    t_full=t_full,
    animation_flag=animation_flag
)