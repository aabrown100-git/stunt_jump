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

################################################################################

# Friction factor
friction_factor = 0.1

################################################################################

# Convert inches to meters
convert_in_to_m = 0.0254; # in to m

# Length of one track length
short_track_segment_in = 8. + 15./16.; # length of short track segment in inches
long_track_segment_in = 11. + 13./16.; # length of long track segment in inches

# ------------------------------------------------------------------------------
# ------------------------------------ FUNCTIONS -------------------------------
# ------------------------------------------------------------------------------

def ramp_ode_fun(t, state):
  '''
  The differential equation for the car on the ramp
  Derivative of path length = velocity
  Mass * derivative of velocity = gravitational force along slope - friction force

  See https://www.myphysicslab.com/roller/roller-single-en.html

  For friction force, we use a Coulomb-tanh model (see
  https://www.sciencedirect.com/science/article/pii/S0301679X05003154#aep-section-id25
  Eq. 9)

  '''

  # Unpack variables from state
  p = state[0] # Path length
  v = state[1] # Velocity

  # ODEs
  dpdt = v
  p2k = interp1d(ps, ks, fill_value="extrapolate")
  k = p2k(p)
  kappa = p2kappa(p)
  g = 9.81 # m/s^2
  m = 1.0 # kg (does not matter for this simulation)
  b = friction_factor
  mu = friction_factor

  # Compute normal force
  N = m*g*1/np.sqrt(1 + k**2) + m * v**2 * kappa

  # Compute friction force
  Fc = np.max([mu * N, 0.0])
  steep = 10
  F = Fc * np.tanh(steep * v)

  dvdt = -g * k / np.sqrt(1 + k**2) - F/m

  # Return dydt = [dpdt, dvdt]
  dstate_dt = np.array([dpdt, dvdt])
  return dstate_dt

def end_track(t,state):
  '''
  The event function to determine if the car has reached the end of the track.
  The car has reached the end of the track if the path length (p) coordinate is pmax
  '''
  return state[0] - np.max(ps)

# Stop the integration when we hit the target
end_track.terminal = True
# We must be moving forward
end_track.direction = 1

def at_rest(t,state):
  '''
  The event function to determine if the car has come to rest.
  The car is at rest if both velocity and acceleration are zero
  '''

  # Unpack variables from state
  p = state[0] # Path length
  v = state[1] # Velocity

  k = p2k(p)
  kappa = p2kappa(p)
  g = 9.81 # m/s^2
  m = 1.0 # kg (does not matter for this simulation)
  b = friction_factor
  mu = friction_factor

  # Compute normal force
  N = m*g*1/np.sqrt(1 + k**2) + m * v**2 * kappa

  # Compute friction force
  Fc = np.max([mu * N, 0.0])
  steep = 10
  F = Fc * np.tanh(steep * v)

  dvdt = -g * k / np.sqrt(1 + k**2) - F/m

  # Tolerance for closeness to zero
  tol = 1e-1

  return np.abs(v) + np.abs(dvdt) - tol

# Don't stop the integration when car comes to rest
at_rest.terminal = False
# Event direction doesn't matter in this case
at_rest.direction = 0

def ballistic_ode_fun(t, state):
  '''
  The differential equation for ballistic motion
  Derivative of x position = x velocity
  Derivative of y position = y velocity
  Mass * derivative of x velocity = -drag force
  Mass * derivative of y velocity = gravitational force - drag force
  '''

  # Unpack variables from state
  x = state[0]
  y = state[1]
  vx = state[2]
  vy = state[3]
  speed = np.sqrt(vx**2 + vy**2)

  # ODEs
  dxdt = vx
  dydt = vy

  # Parameters
  g = 9.81 # Gravitational acceleration, m/s^2
  c = 0.0 # Drag coefficient, dimensionless
  r = 0.05 # Effective radius, m
  A = np.pi * r**2 # Effective area, m^2
  m = 0.2 # Mass, kg
  rho_air = 1.28 # Density of air, kg/m^3
  k = 0.5 * c * rho_air * A # Drag constant

  dvxdt = -k/m * speed * vx
  dvydt = -g  - k/m * speed * vy

  # Return dstate_dt = [dxdt, dydt, dvxdt, dvydt]
  dstate_dt = [dxdt, dydt, dvxdt, dvydt]
  return dstate_dt

def hit_target(t,state):
  '''
  The event function to determine if the car has hit the target.
  The car has hit the target if the y-coordinate is zero.
  '''
  # We've hit the target if the y-coordinate is 0
  return state[1]

# Stop the integration when we hit the target
hit_target.terminal = True
# We must be moving downards (don't stop before we begin moving upwards!)
hit_target.direction = -1

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------ BEGIN MAIN SCRIPT -----------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------------------------- SIMULATION PARAMETERS ---------------------------
# ------------------------------------------------------------------------------
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
class SplineEditor:
    def __init__(self, master):
        self.traj = None
        self.master = master
        self.load_background_image("background_image.png")
        self.canvas = tk.Canvas(master, width=self.background_image.width(), height=self.background_image.height(), bg="white")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.background_image)
        anchor_locations = np.array([
                 [0, 2.5], # feet
                 [1, 1],
                 [2, 0.25],
                 [3, 0.25],
                 [4, 1]])
        ppi = 85.5
        df = pd.read_excel("nodelocs.xlsx")
        anchor_locations = df.values
        anchor_locations=anchor_locations*ppi
        anchor_locations[:, 0]=anchor_locations[:, 0]+74
        anchor_locations[:, 1]=453-anchor_locations[:, 1]

        self.spline_color = "blue"
        self.traj_color = "green"
        self.node_color = "black"
        self.bad_node_color = "red"

        self.node_size = 10
        self.nodes = [self.canvas.create_oval(anchor_locations[i, 0]-self.node_size,\
         anchor_locations[i, 1]-self.node_size,\
         anchor_locations[i, 0]+self.node_size,\
         anchor_locations[i, 1]+self.node_size,\
         fill=self.node_color, tags="nodes") for i in range(5)]
        # self.nodes = [self.canvas.create_oval(74+i*50-5, 453-5, 74+i*50+5, 453+5, fill="blue", tags="nodes") for i in range(6)]
        self.drag_data = {"item": None, "x": 0, "y": 0}
        self.canvas.tag_bind("nodes", "<ButtonPress-1>", self.on_drag_start)
        self.canvas.tag_bind("nodes", "<ButtonRelease-1>", self.on_drag_stop)
        self.canvas.tag_bind("nodes", "<B1-Motion>", self.on_drag)
        self.done_button = tk.Button(master, text="Simulate", command=self.return_nodes)
        self.done_button.pack()
        self.node_positions = None

        self.max_node_to_node = 2*long_track_segment_in * ppi / 12
        self.warning = None
        self.bad_nodes = None
        self.display_warning = False

    def load_background_image(self, filename):
        image = Image.open(filename)
        self.background_image = ImageTk.PhotoImage(image)

    def on_drag_start(self, event):
        self.drag_data["item"] = self.canvas.find_closest(event.x, event.y)[0]
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_drag_stop(self, event):
        self.drag_data["item"] = None

    def on_drag(self, event):
        d_x = event.x - self.drag_data["x"]
        d_y = event.y - self.drag_data["y"]
        node_id = self.drag_data["item"]
        x1, y1, x2, y2 = self.canvas.coords(node_id)

        self.canvas.move(self.drag_data["item"], d_x, d_y)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.update_spline()

    def update_spline(self):
        spline_x = [self.canvas.coords(node)[0] + self.node_size for node in self.nodes]
        spline_y = [self.canvas.coords(node)[1] + self.node_size for node in self.nodes]
        spline = CubicSpline(spline_x, spline_y, bc_type='not-a-knot')
        t = np.linspace(min(spline_x), max(spline_x), 1000)
        spline_points = [(x, spline(x)) for x in t]
        if hasattr(self, 'spline_line') & hasattr(self, 'traj'):
            self.canvas.delete(self.spline_line)
            self.canvas.delete(self.traj)


        def update_traj(self):

            def ramp_ode_fun1(t, state):
                '''
                The differential equation for the car on the ramp
                Derivative of path length = velocity
                Mass * derivative of velocity = gravitational force along slope - friction force

                See https://www.myphysicslab.com/roller/roller-single-en.html

                For friction force, we use a Coulomb-tanh model (see
                https://www.sciencedirect.com/science/article/pii/S0301679X05003154#aep-section-id25
                Eq. 9)

                '''

                # Unpack variables from state
                p = state[0]  # Path length
                v = state[1]  # Velocity

                # ODEs
                dpdt = v
                p2k = interp1d(ps, ks, fill_value="extrapolate")
                k = p2k(p)
                kappa = p2kappa(p)
                g = 9.81  # m/s^2
                m = 1.0  # kg (does not matter for this simulation)
                b = friction_factor
                mu = friction_factor

                # Compute normal force
                N = m * g * 1 / np.sqrt(1 + k ** 2) + m * v ** 2 * kappa

                # Compute friction force
                Fc = np.max([mu * N, 0.0])
                steep = 10
                F = Fc * np.tanh(steep * v)

                dvdt = -g * k / np.sqrt(1 + k ** 2) - F / m

                # Return dydt = [dpdt, dvdt]
                dstate_dt = np.array([dpdt, dvdt])
                return dstate_dt

            def at_rest1(t, state):
                '''
                The event function to determine if the car has come to rest.
                The car is at rest if both velocity and acceleration are zero
                '''

                # Unpack variables from state
                p = state[0]  # Path length
                v = state[1]  # Velocity

                k = p2k(p)
                kappa = p2kappa(p)
                g = 9.81  # m/s^2
                m = 1.0  # kg (does not matter for this simulation)
                b = friction_factor
                mu = friction_factor

                # Compute normal force
                N = m * g * 1 / np.sqrt(1 + k ** 2) + m * v ** 2 * kappa

                # Compute friction force
                Fc = np.max([mu * N, 0.0])
                steep = 10
                F = Fc * np.tanh(steep * v)

                dvdt = -g * k / np.sqrt(1 + k ** 2) - F / m

                # Tolerance for closeness to zero
                tol = 1e-1

                return np.abs(v) + np.abs(dvdt) - tol

                # Don't stop the integration when car comes to rest
                at_rest1.terminal = False
                # Event direction doesn't matter in this case
                at_rest1.direction = 0

            def ballistic_ode_fun1(t, state):
                '''
                The differential equation for ballistic motion
                Derivative of x position = x velocity
                Derivative of y position = y velocity
                Mass * derivative of x velocity = -drag force
                Mass * derivative of y velocity = gravitational force - drag force
                '''

                # Unpack variables from state
                x = state[0]
                y = state[1]
                vx = state[2]
                vy = state[3]
                speed = np.sqrt(vx ** 2 + vy ** 2)

                # ODEs
                dxdt = vx
                dydt = vy

                # Parameters
                g = 9.81  # Gravitational acceleration, m/s^2
                c = 0.0  # Drag coefficient, dimensionless
                r = 0.05  # Effective radius, m
                A = np.pi * r ** 2  # Effective area, m^2
                m = 0.2  # Mass, kg
                rho_air = 1.28  # Density of air, kg/m^3
                k = 0.5 * c * rho_air * A  # Drag constant

                dvxdt = -k / m * speed * vx
                dvydt = -g - k / m * speed * vy

                # Return dstate_dt = [dxdt, dydt, dvxdt, dvydt]
                dstate_dt = [dxdt, dydt, dvxdt, dvydt]
                return dstate_dt

            def end_track1(t, state):
                '''
                The event function to determine if the car has reached the end of the track.
                The car has reached the end of the track if the path length (p) coordinate is pmax
                '''
                return state[0] - np.max(ps)

                # Stop the integration when we hit the target
                end_track1.terminal = True
                # We must be moving forward
                end_track1.direction = 1

            def hit_target1(t, state):
                '''
                The event function to determine if the car has hit the target.
                The car has hit the target if the y-coordinate is zero.
                '''
                # We've hit the target if the y-coordinate is 0
                return state[1]

                # Stop the integration when we hit the target
                hit_target1.terminal = True
                # We must be moving downards (don't stop before we begin moving upwards!)
                hit_target1.direction = -1

            ppi = 85.5
            anchor_locations = np.array(
                [(self.canvas.coords(node)[0] + self.node_size, self.canvas.coords(node)[1] + self.node_size) for node in self.nodes])
            anchor_locations[:, 0] = anchor_locations[:, 0] - 74
            anchor_locations[:, 1] = 453 - anchor_locations[:, 1]
            anchor_locations = anchor_locations * 12 / ppi
            xnodes = np.array(anchor_locations[:, 0])
            ynodes = np.array(anchor_locations[:, 1])

            # Create a cubic spline between anchor locations
            cs = CubicSpline(xnodes, ynodes, bc_type='not-a-knot')
            xs = np.linspace(np.min(xnodes), np.max(xnodes), 1000)
            ys = cs(xs)

            # Convert to meters
            xnodes = xnodes * convert_in_to_m  # m
            ynodes = ynodes * convert_in_to_m  # m
            xs = xs * convert_in_to_m
            ys = ys * convert_in_to_m

            ps = np.zeros_like(xs)
            for i in range(len(xs) - 1):
                ps[i + 1] = ps[i] + np.sqrt((xs[i + 1] - xs[i]) ** 2 + (ys[i + 1] - ys[i]) ** 2)

            # Truncate path length to nearest smaller multiple of long track length
            truncated_length = np.floor(ps[-1] / (long_track_segment_in * convert_in_to_m)) * (
                    long_track_segment_in * convert_in_to_m)  # m
            n_long_track_segments = np.floor(
                ps[-1] / (long_track_segment_in * convert_in_to_m))  # number of long track segments
            s = ps[ps <= truncated_length]
            xs = xs[0:len(ps)]
            ys = ys[0:len(ps)]

            # Create maps between p, x, and y. Use cubic interpolation
            p2x = interp1d(ps, xs, fill_value="extrapolate", kind='cubic')
            x2p = interp1d(xs, ps, fill_value="extrapolate", kind='cubic')
            p2y = interp1d(ps, ys, fill_value="extrapolate", kind='cubic')
            x2y = interp1d(xs, ys, fill_value="extrapolate", kind='cubic')

            # Create map from p to k (slope)
            dp = 1e-6  # finite difference for slope calculation (dy/dx)
            ks = (p2y(ps + dp) - p2y(ps)) / (p2x(ps + dp) - p2x(ps))
            p2k = interp1d(ps, ks, fill_value="extrapolate")

            # Compute curvature
            dx = 1e-6
            d2ydx2s = (x2y(xs + dx) - 2 * x2y(xs) + x2y(xs - dx)) / dx ** 2
            dydxs = ks
            kappas = d2ydx2s / (1 + dydxs ** 2) ** (3 / 2)
            p2kappa = interp1d(ps, kappas, fill_value="extrapolate")
            p0 = p_initial  # Initial path length position
            v0 = 0  # Initial velocity
            state0 = np.array([p0, v0])
            t_span = [0, 15]  # Time span of integration
            soln = solve_ivp(ramp_ode_fun1, t_span, state0, method='RK45', dense_output=True,
                             events=[end_track1, at_rest1])

            # Did the car reach the end of the track, or did it come to rest?
            end_track1_flag = len(soln.t_events[0]) > 0
            at_rest1_flag = len(soln.t_events[1]) > 0

            # Get the car position solution at specified times
            h = 0.01
            if end_track1_flag:
                t = np.arange(t_span[0], soln.t_events[0][0], h)
            elif at_rest1_flag:
                t_to_rest = soln.t_events[1][0]  # Record time to come to rest
                t = np.arange(t_span[0], t_to_rest, h)  # Keep time to come to rest, plus 2 seconds
            else:
                t = np.arange(t_span[0], t_span[1], h)

            sol = soln.sol(t)
            p = sol[0]  # Extracting position in path length variable
            v = sol[1]  # Extracting velocity

            # Save x, y position of car on track. We will add ballistic portion next.
            x_track = p2x(p)
            y_track = p2y(p)
            t_track = t

            if end_track1_flag:
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

            if end_track1_flag:
                # Initial speed and launch angle
                speed0 = speed_end
                theta0 = theta_end

                # Initial conditions: x0, y0, vx0, vy0
                x0 = x_end
                y0 = y_end
                vx0 = np.cos(theta0) * speed0
                vy0 = np.sin(theta0) * speed0
                state0 = [x0, y0, vx0, vy0]

                # Integrate ballistic ODE
                t_span = [0, 2]  # Time span of integration
                soln = solve_ivp(ballistic_ode_fun1, t_span, state0, method='RK45', dense_output=True,
                                 events=hit_target)

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
            xnodes_ft = xnodes / convert_in_to_m / 12
            ynodes_ft = ynodes / convert_in_to_m / 12
            xs_ft = xs / convert_in_to_m / 12
            ys_ft = ys / convert_in_to_m / 12
            x_full_ft = x_full / convert_in_to_m / 12
            y_full_ft = y_full / convert_in_to_m / 12
            ppi = 85.5
            x_full_ft = x_full_ft * ppi
            y_full_ft = y_full_ft * ppi
            x_full_ft = x_full_ft + 74
            y_full_ft = 453 - y_full_ft
            traj_points=np.array([x_full_ft, y_full_ft]).T
            return traj_points
        traj_points=update_traj(self)
        plot_traj=traj_points.flatten().tolist()
        self.coordinates = traj_points
        self.traj=self.canvas.create_line(plot_traj, fill=self.traj_color, width=2, dash=(2, 2))  # Create a small oval (point) at each coordinate
        self.spline_line = self.canvas.create_line(spline_points, fill=self.spline_color, tags="spline", smooth=True, splinesteps=100, width=2)


        too_far_from_next = np.zeros_like(self.nodes)
        too_far_from_prev = np.zeros_like(self.nodes)
        self.canvas.delete(self.warning)
        # if self.bad_nodes!=None:
        #     tot_objs = self.canvas.find_all()
        #     self.canvas.delete(self.bad_nodes)
        # Calculate distance to previous node
        for node_id in self.nodes:
            x1, y1, x2, y2 = self.canvas.coords(node_id)
            i = node_id - self.nodes[0]
            if node_id > self.nodes[0]:
                prev_node_id = node_id - 1
                px1, py1, px2, py2 = self.canvas.coords(prev_node_id)
                distance_to_prev = np.sqrt((x1 - px1)**2 + (y1 - py1)**2)
                if distance_to_prev > self.max_node_to_node:
                    too_far_from_prev[i] = True
                    too_far_from_prev[i-1] = True
            # Calculate distance to next node
            if node_id < self.nodes[-1]:
                next_node_id = node_id + 1
                nx1, ny1, nx2, ny2 = self.canvas.coords(next_node_id)
                distance_to_next = np.sqrt((nx1 - x2)**2 + (ny1 - y2)**2)
                if distance_to_next> self.max_node_to_node:
                    too_far_from_next[i] = True
                    too_far_from_next[i+1] = True
        # mask = np.logical_or(distance_to_next > self.max_node_to_node, distance_to_prev > self.max_node_to_node)
        mask = np.logical_or(too_far_from_next, too_far_from_prev)
        bad_node_ids = np.array(self.nodes)[mask]
        good_node_ids = np.array(self.nodes)[~mask]
        for node_id in bad_node_ids:
            self.canvas.itemconfig(node_id, fill=self.bad_node_color)
        for node_id in good_node_ids:
            self.canvas.itemconfig(node_id, fill=self.node_color)
        if np.any(mask):
            # self.display_warning = True
            self.warning = self.canvas.create_text(350, 50, text="INVALID TRACK WARNING: Track is too long!", fill="red", font=("Arial",24))

    def return_nodes(self):
        self.node_positions = np.array([(self.canvas.coords(node)[0] + self.node_size, self.canvas.coords(node)[1] + self.node_size) for node in self.nodes])
        self.master.destroy()


def edit_splines():
    root = tk.Tk()
    ppi=85.5
    root.title("Spline Editor")
    app = SplineEditor(root)
    root.mainloop()
    app.node_positions[:, 0] = app.node_positions[:, 0] - 74
    app.node_positions[:, 1] = 453 - app.node_positions[:, 1]
    app.coordinates[:, 0]=app.coordinates[:, 0]-74
    app.coordinates[:, 1]=453-app.coordinates[:, 1]
    return app.node_positions/ppi, app.coordinates/ppi


# Initial track position
p_initial = 0.; # inches
p_initial = p_initial * convert_in_to_m

# Simulation mode (halfpipe vs jump)
ramp_type = 'jump'

# Animation flag
animation_flag = False

# ------------------------------------------------------------------------------
# ---------------------------- RAMP SHAPE --------------------------------------
# ------------------------------------------------------------------------------
# Define the ramp shape using control points for a cubic spine interpolation.

# Halfpipe shape for tuning parameters
if ramp_type == 'halfpipe':

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

# Ramp shape for jump simulation
elif ramp_type == 'jump':
  temp, traj_coordinates=edit_splines()
  df = pd.DataFrame(temp)
  df.to_excel("nodelocs.xlsx", index=False)
  anchor_locations=temp*12
  # anchor_locations = np.array([
  #     [0, 2.5 ], # feet
  #     [1, 1 ],
  #     [2, 0.25],
  #     [3, 0.25],
  #     [4, 1 ]
  # ]) * 12


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

else:
  raise ValueError('Invalid ramp_type')


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

#plt.plot(xs, p2kappa(ps), '-o', label = 'kappa')
#plt.plot(xs,ys, label = 'Track')
#plt.axvline(xs2[0]*convert_in_to_m, color = 'k')
#plt.axvline(xs3[0]*convert_in_to_m, color = 'k')
#plt.legend()


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
soln = solve_ivp(ramp_ode_fun, t_span, state0, method = 'RK45', dense_output = True, events = [end_track, at_rest])

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
