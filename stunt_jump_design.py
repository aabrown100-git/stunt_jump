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
from spline_editor_qt import edit_splines

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
temp, traj_coordinates = edit_splines()
if temp is not None:
    df = pd.DataFrame(temp)
    df.to_excel("nodelocs.xlsx", index=False)
    anchor_locations = temp * 12

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
# ---------------------------- PATH LENGTH AND MAPS ----------------------------
# ------------------------------------------------------------------------------
# Here, we compute the pathlength (distance along the track) from the left-most
# point of the track. Pathlength (ps) is computed for every spline point (xs, ys).
# We assume each point on the spline is connected by straight lines. The error in
# this approximation is quadratic.
# We then create maps from p to x, p to y, p to k, where k is the slope of the
# track, and p to kappa, where kappa is the curvature of the track.

# Compute path length
ps = np.zeros_like(xs)
for i in range(len(xs)-1):
  ps[i+1] = ps[i] + np.sqrt( (xs[i+1] - xs[i])**2 + (ys[i+1] - ys[i])**2)

# Truncate path length to nearest smaller multiple of long track length
truncated_length = np.floor(ps[-1] / (long_track_segment_in * convert_in_to_m)) * (long_track_segment_in * convert_in_to_m) # m
n_long_track_segments = np.floor(ps[-1] / (long_track_segment_in * convert_in_to_m)) # number of long track segments
ps = ps[ps <= truncated_length]
xs = xs[0:len(ps)]
ys = ys[0:len(ps)]
print(f"The track is {np.max(ps):.2f} meters ({int(n_long_track_segments)} long segments) long.")

# Create maps between p, x, and y. Use cubic interpolation
p2x = interp1d(ps,xs, fill_value="extrapolate", kind = 'cubic')
x2p = interp1d(xs,ps, fill_value="extrapolate", kind = 'cubic')
p2y = interp1d(ps,ys, fill_value="extrapolate", kind = 'cubic')
x2y = interp1d(xs,ys, fill_value="extrapolate", kind = 'cubic')

# Create map from p to k (slope)
dp = 1e-6 # finite difference for slope calculation (dy/dx)
ks = (p2y(ps+dp) - p2y(ps)) / (p2x(ps+dp) - p2x(ps))
p2k = interp1d(ps,ks, fill_value="extrapolate")

# Compute curvature
dx = 1e-6
d2ydx2s = (x2y(xs + dx) - 2*x2y(xs) + x2y(xs - dx)) / dx**2
dydxs = ks
kappas = d2ydx2s / (1 + dydxs**2)**(3/2)
p2kappa = interp1d(ps, kappas, fill_value="extrapolate")

# ------------------------------------------------------------------------------
# ---------------------------- SIMULATION --------------------------------------
# ------------------------------------------------------------------------------

# ------------------------- INTEGRATE RAMP ODE ---------------------------------
# Integrate the ODE for the car on the ramp. The ODE is defined in ramp_ode_fun.
# The integration stops when the car reaches the end of the track or comes to rest.

p0 = p_initial # Initial path length position
v0 = 0 # Initial velocity
state0 = np.array([p0, v0])
t_span = [0,15] # Time span of integration

# Create parameters dictionary for the ODE
params = {
    'friction_factor': friction_factor,
    'p2k': p2k,
    'p2kappa': p2kappa,
    'ps': ps  # Add the path length array
}

soln = solve_ivp(ramp_ode_fun, t_span, state0, method = 'RK45', dense_output = True, 
                 events = [end_track, at_rest], args=(params,))

# Did the car reach the end of the track, or did it come to rest?
end_track_flag = len(soln.t_events[0]) > 0
at_rest_flag = len(soln.t_events[1]) > 0

# Get the car position solution at specified times
h = 0.01
if end_track_flag:
  t = np.arange(t_span[0], soln.t_events[0][0], h)
elif at_rest_flag:
  t_to_rest = soln.t_events[1][0] # Record time to come to rest
  t = np.arange(t_span[0], t_to_rest, h) # Keep time to come to rest, plus 2 seconds
else:
  t = np.arange(t_span[0], t_span[1], h)

sol = soln.sol(t)
p = sol[0] # Extracting position in path length variable
v = sol[1] # Extracting velocity

# Save x, y position of car on track. We will add ballistic portion next.
x_track = p2x(p)
y_track = p2y(p)
t_track = t

if end_track_flag:
  # Get position, velocity, and slope at end of the ramp
  x_end = x_track[-1]
  y_end = y_track[-1]
  speed_end = v[-1]
  theta_end = np.arctan2(p2k(p[-1]), 1.0)
else:
  x_end = x_track[-1]

# ------------------------- INTEGRATE BALLISTIC ODE -----------------------------
# If we reached the end of the track, then integrate the ballistic ODE to get
# the motion of the car in the air. If we did not reach the end of the track, skip.
# The ODE is defined in ballistic_ode_fun.

if end_track_flag:
  # Initial speed and launch angle
  speed0 = speed_end
  theta0 = theta_end

  # Initial conditions: x0, y0, vx0, vy0
  x0 = x_end
  y0 = y_end
  vx0 = np.cos(theta0) * speed0
  vy0 = np.sin(theta0) * speed0
  state0 = [x0,y0,vx0,vy0]

  # Integrate ballistic ODE
  t_span = [0,2] # Time span of integration
  soln = solve_ivp(ballistic_ode_fun, t_span, state0, method = 'RK45', dense_output = True, events=hit_target)

  # Plot solution at specified times
  h = 0.01
  t = np.arange(t_span[0], soln.t_events[0][0], h)
  sol = soln.sol(t)
  x_ballistic = sol[0]
  y_ballistic = sol[1]
  vx_ballistic = sol[2]
  vy_ballistic = sol[3]

  t_ballistic = t

  # Add x,y_ballistic to x,y_full
  x_full = np.concatenate((x_track, x_ballistic[1:-1]))
  y_full = np.concatenate((y_track, y_ballistic[1:-1]))
  t_full = np.concatenate((t_track, t_track[-1] + t_ballistic[1:-1]))
else:
  x_full = x_track
  y_full = y_track
  t_full = t_track

# ------------------------------------------------------------------------------
# ---------------------------- PLOTTING ----------------------------------------
# ------------------------------------------------------------------------------

# Plot motion of car on track (and in air, if applicable)

# Convert units to feet for plotting
xnodes_ft = xnodes/convert_in_to_m / 12
ynodes_ft = ynodes/convert_in_to_m / 12
xs_ft = xs/convert_in_to_m / 12
ys_ft = ys/convert_in_to_m / 12

# Create ellipse objects for rings
ring_1_ellipse = Ellipse(xy=ring_1, width=0.2, height=6/12, facecolor="none", edgecolor="k")
ring_2_ellipse = Ellipse(xy=ring_2, width=0.2, height=6/12, facecolor="none", edgecolor="k")
ring_3_ellipse = Ellipse(xy=ring_3, width=0.2, height=6/12, facecolor="none", edgecolor="k")

# Plot track, nodes, long segments, rings, and car trajectory
fig, ax = plt.subplots(figsize=(10, 5), dpi = 200, layout = 'constrained')
ax.plot(xs_ft,ys_ft, color = 'orange', linewidth = 2) # Plot track
ax.scatter(xnodes_ft, ynodes_ft, color = 'gray') # Plot spline nodes

# Label the angel at the end of the ramp
if end_track_flag:
    theta_end = np.arctan2(p2k(p[-1]), 1.0)
    offset_x = 0.15
    offset_y = 0.04
    ax.text(x_end/convert_in_to_m/12 + offset_x, y_end/convert_in_to_m/12 + offset_y, f'{theta_end*180/np.pi:.2f}\u00b0', fontsize=8, color='black', ha='left')
    # Add an angle symbol to show the angle
    ax.plot([x_end/convert_in_to_m/12, x_end/convert_in_to_m/12 + 0.3*np.cos(theta_end)], [y_end/convert_in_to_m/12, y_end/convert_in_to_m/12 + 0.3*np.sin(theta_end)], color = 'black')
    ax.plot([x_end/convert_in_to_m/12, x_end/convert_in_to_m/12 + 0.3], [y_end/convert_in_to_m/12, y_end/convert_in_to_m/12], color = 'black')

# Plot diamond marker at each long track segment junction
for i in range(int(n_long_track_segments)+1):
    idx = np.argmin(np.abs(ps - i * long_track_segment_in * convert_in_to_m))
    ax.scatter(xs_ft[idx], ys_ft[idx], color = 'black', marker = 'd')

#plt.axhline(y = 0.0, color = 'k', linestyle = '-', linewidth = 5) # Plot ground (y=0)

x_full_ft=traj_coordinates[:, 0]
y_full_ft=traj_coordinates[:, 1]

if ramp_type == 'jump':
    ax.add_patch(ring_1_ellipse)  # Plot ring 1
    ax.add_patch(ring_2_ellipse)  # Plot ring 2
    ax.add_patch(ring_3_ellipse)  # Plot ring 3
ax.plot(x_full_ft, y_full_ft, color = 'green', linestyle = ':') # Plot car trajectory
ax.set_aspect('equal', 'box') # Set x,y aspect ratio equal
plt.xticks(np.arange(0, 10, 1))
plt.yticks(np.arange(0, 5, 1))
plt.grid(visible=True)
plt.ylim(0, 5)
plt.xlim(0, 10)
plt.xlabel('x [feet]')
plt.ylabel('y [feet]')
if ramp_type == 'halfpipe' and at_rest_flag:
    plt.title(f'Halfpipe: {t_to_rest:.2f}s')
    print(f'The car took approximately {t_to_rest:.2f} seconds to come to rest.')
elif ramp_type == 'jump' and end_track_flag:
    x_landing = x_full[-1]
    plt.title(f'Jump: {(x_landing-x_end)/convert_in_to_m:.2f} inches')
plt.savefig('stunt_jump.png', )



# Convert units to feet for plotting
xnodes_ft = xnodes/convert_in_to_m / 12
ynodes_ft = ynodes/convert_in_to_m / 12
xs_ft = xs/convert_in_to_m / 12
ys_ft = ys/convert_in_to_m / 12

# Create ellipse objects for rings
ring_1_ellipse = Ellipse(xy=ring_1, width=0.2, height=6/12, facecolor="none", edgecolor="k")
ring_2_ellipse = Ellipse(xy=ring_2, width=0.2, height=6/12, facecolor="none", edgecolor="k")
ring_3_ellipse = Ellipse(xy=ring_3, width=0.2, height=6/12, facecolor="none", edgecolor="k")
fig2, ax2 = plt.subplots(figsize=(10, 5), dpi = 100, layout = 'constrained')
ax2.plot(xs_ft,ys_ft, color = 'orange', linewidth = 2) # Plot track
ax2.scatter(xnodes_ft, ynodes_ft, color = 'gray') # Plot spline nodes

# Label the angel at the end of the ramp
if end_track_flag:
  theta_end = np.arctan2(p2k(p[-1]), 1.0)
  offset_x = 0.15
  offset_y = 0.04
  ax2.text(x_end/convert_in_to_m/12 + offset_x, y_end/convert_in_to_m/12 + offset_y, f'{theta_end*180/np.pi:.2f}\u00b0', fontsize=8, color='black', ha='left')
  # Add an angle symbol to show the angle
  ax2.plot([x_end/convert_in_to_m/12, x_end/convert_in_to_m/12 + 0.3*np.cos(theta_end)], [y_end/convert_in_to_m/12, y_end/convert_in_to_m/12 + 0.3*np.sin(theta_end)], color = 'black')
  ax2.plot([x_end/convert_in_to_m/12, x_end/convert_in_to_m/12 + 0.3], [y_end/convert_in_to_m/12, y_end/convert_in_to_m/12], color = 'black')

# Plot diamond marker at each long track segment junction
for i in range(int(n_long_track_segments)+1):
  idx = np.argmin(np.abs(ps - i * long_track_segment_in * convert_in_to_m))
  ax2.scatter(xs_ft[idx], ys_ft[idx], color = 'black', marker = 'd')

#plt.axhline(y = 0.0, color = 'k', linestyle = '-', linewidth = 5) # Plot ground (y=0)

if ramp_type == 'jump':
  ax2.add_patch(ring_1_ellipse)  # Plot ring 1
  ax2.add_patch(ring_2_ellipse)  # Plot ring 2
  ax2.add_patch(ring_3_ellipse)  # Plot ring 3
ax2.plot(x_full_ft, y_full_ft, color = 'green', linestyle = ':') # Plot car trajectory
ax2.set_aspect('equal', 'box') # Set x,y aspect ratio equal
plt.xticks(np.arange(0, 10, 1))
plt.yticks(np.arange(0, 5, 1))
plt.grid(visible=True)
plt.ylim(0, 5)
plt.xlim(0, 10)
plt.xlabel('x [feet]')
plt.ylabel('y [feet]')
if ramp_type == 'halfpipe' and at_rest_flag:
  plt.title(f'Halfpipe: {t_to_rest:.2f}s')
  print(f'The car took approximately {t_to_rest:.2f} seconds to come to rest.')
elif ramp_type == 'jump' and end_track_flag:
  x_landing = x_full[-1]
  plt.title(f'Jump: {(x_landing-x_end)/convert_in_to_m:.2f} inches')
  print(f'The car traveled {(x_landing-x_end)/convert_in_to_m:.2f} inches from the end of the track!')
plt.savefig('background_image.png', )

# Save inverted image for dark mode
img = Image.open(r'stunt_jump.png').convert('RGB')
img_inv = ImageOps.invert(img)
img_inv.save('stunt_jump_inv.png')


# ------------------------------------------------------------------------------
# ---------------------------- ANIMATION ---------------------------------------
# ------------------------------------------------------------------------------
# Optional

if animation_flag:
  from matplotlib import rc
  rc('animation', html='jshtml')

  # Make frames for animation
  scatter = ax.scatter(x_full_ft, y_full_ft, color = 'k')
  label_string = f'Time = {t_full[0]}s'
  label = ax.text(np.median(x_full_ft), np.max(y_full_ft), label_string, ha='center', va='center', fontsize=12, color="Red")
  def update(i):
      scatter.set_offsets((x_full_ft[i], y_full_ft[i]))
      label.set_text(f'Time = {t_full[i]:.2f}s')
      return scatter,

  # Animate with desired framerate
  playback = 1 # 1 = real time
  fps = 24 # frames per second

  T_sim = t_full[-1] - t_full[0] # Total simulation time

  nframes = fps * T_sim / playback
  step = int(len(x_full_ft) // nframes)
  frames = np.arange(0, len(x_full_ft), step)

  interval = 1/fps*1000 # frame delay in ms
  ani = animation.FuncAnimation(
      fig, update, frames = frames, interval=interval, blit = True)

  ani.save('stunt_jump.mp4', writer='ffmpeg', fps=fps)
