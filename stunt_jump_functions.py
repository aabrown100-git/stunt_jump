# ------------------------------------------------------------------------------
# ---------------------------------- IMPORTS -----------------------------------
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from PIL import Image, ImageOps

# ------------------------------------------------------------------------------
# ---------------------------- CONSTANTS ---------------------------------------
# ------------------------------------------------------------------------------

# Convert inches to meters
convert_in_to_m = 0.0254; # in to m

# Length of one track length
short_track_segment_in = 8. + 15./16.; # length of short track segment in inches
long_track_segment_in = 11. + 13./16.; # length of long track segment in inches

# ------------------------------------------------------------------------------
# ------------------------------------ FUNCTIONS -------------------------------
# ------------------------------------------------------------------------------

def total_energy(state, params):
    '''
    Computes the total energy of the car at a given state.
    The total energy is the sum of kinetic and potential energy.
    '''

    # Unpack variables from state
    p = state[0] # Path length
    v = state[1] # Velocity

    # Unpack parameters
    p2y = params['p2y'] # Function to get y-coordinate from path length
    y_track_min = params['y_track_min'] # Minimum y-coordinate of the track
    
    # Compute kinetic energy
    KE = 0.5 * v**2

    # Compute potential energy
    h = p2y(p) - y_track_min
    PE = 9.81 * h

    return KE + PE, KE, PE

def ramp_ode_fun(t, state, params):
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

    # Unpack parameters
    friction_factor = params['friction_factor']
    p2k = params['p2k']
    p2kappa = params['p2kappa']
    
    # ODEs
    dpdt = v

    k = p2k(p)
    kappa = p2kappa(p)
    g = 9.81 # m/s^2
    m = 1.0 # kg (does not matter for this simulation)
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

def end_track(t, state, params):
    '''
    The event function to determine if the car has reached the end of the track.
    The car has reached the end of the track if the path length (p) coordinate is pmax
    '''
    return state[0] - np.max(params['ps'])

# Stop the integration when we hit the target
end_track.terminal = True
# We must be moving forward
end_track.direction = 1

def at_rest(t, state, params):
    '''
    The event function to determine if the car has come to rest.
    The car is at rest if the total energy is close to zero.
    '''

    # Compute kinetic + potential energy (divided by mass)
    E, _, _ = total_energy(state, params)

    # Tolerance for closeness to zero
    tol = 1e-3

    return E - tol
    # return np.max([np.abs(v), np.abs(dvdt)]) - tol

# Don't stop the integration when car comes to rest
at_rest.terminal = False
# Event direction doesn't matter in this case
at_rest.direction = -1

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
    c = 0.47 # Drag coefficient, dimensionless
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

def compute_path_length_and_maps(xs, ys):
    """
    Computes the path length and creates maps for a track defined by spline points.

    Parameters:
        xs (np.ndarray): x-coordinates of the spline points (in meters).
        ys (np.ndarray): y-coordinates of the spline points (in meters).

    Returns:
        dict: A dictionary containing:
            - ps (np.ndarray): Path length array.
            - p2x (interp1d): Map from path length to x-coordinates.
            - p2y (interp1d): Map from path length to y-coordinates.
            - p2k (interp1d): Map from path length to slope (dy/dx).
            - p2kappa (interp1d): Map from path length to curvature.
            - truncated_length (float): Truncated path length (in meters).
            - n_long_track_segments (int): Number of long track segments.
    """
    # Compute path length
    ps = np.zeros_like(xs)
    for i in range(len(xs) - 1):
        ps[i + 1] = ps[i] + np.sqrt((xs[i + 1] - xs[i])**2 + (ys[i + 1] - ys[i])**2)

    # Truncate path length to nearest smaller multiple of long track length
    truncated_length = (
        np.floor(ps[-1] / (long_track_segment_in * convert_in_to_m))
        * (long_track_segment_in * convert_in_to_m)
    )  # m
    n_long_track_segments = int(
        np.floor(ps[-1] / (long_track_segment_in * convert_in_to_m))
    )  # number of long track segments
    ps = ps[ps <= truncated_length]
    xs = xs[: len(ps)]
    ys = ys[: len(ps)]
    print(f"The track is {np.max(ps):.2f} meters ({n_long_track_segments} long segments) long.")

    # Create maps between p, x, and y. Use cubic interpolation
    p2x = interp1d(ps, xs, fill_value="extrapolate", kind="cubic")
    x2p = interp1d(xs, ps, fill_value="extrapolate", kind="cubic")
    p2y = interp1d(ps, ys, fill_value="extrapolate", kind="cubic")
    x2y = interp1d(xs, ys, fill_value="extrapolate", kind="cubic")

    # Calculate minimum y-coordinate of the track
    y_track_min = np.min(ys)

    # Create map from p to k (slope)
    dp = 1e-6  # finite difference for slope calculation (dy/dx)
    ks = (p2y(ps + dp) - p2y(ps)) / (p2x(ps + dp) - p2x(ps))
    p2k = interp1d(ps, ks, fill_value="extrapolate")

    # Compute curvature
    dx = 1e-6
    d2ydx2s = (x2y(xs + dx) - 2*x2y(xs) + x2y(xs - dx)) / dx**2
    dydxs = ks
    kappas = d2ydx2s / (1 + dydxs**2)**(3/2)
    p2kappa = interp1d(ps, kappas, fill_value="extrapolate")

    return {
        "ps": ps,
        "p2x": p2x,
        "p2y": p2y,
        "p2k": p2k,
        "p2kappa": p2kappa,
        "truncated_length": truncated_length,
        "n_long_track_segments": n_long_track_segments,
        "xs": xs,
        "ys": ys,
        "y_track_min": y_track_min,
    }

def integrate_ramp_ode(ramp_params, ramp_ode_fun, end_track, at_rest):
    """
    Integrates the ODE for the car on the ramp and determines its motion.

    Parameters:
        ramp_params (dict): A dictionary containing:
            - friction_factor (float): Friction factor.
            - p_initial (float): Initial path length position (in meters).
            - p2x (callable): Function mapping path length to x-coordinates.
            - p2y (callable): Function mapping path length to y-coordinates.
            - p2k (callable): Function mapping path length to slope (dy/dx).
            - p2kappa (callable): Function mapping path length to curvature.
        ramp_ode_fun (callable): Function defining the ramp ODE.
        end_track (callable): Event function for reaching the end of the track.
        at_rest (callable): Event function for the car coming to rest.

    Returns:
        dict: A dictionary containing:
            - x_track (np.ndarray): x-coordinates of the car on the track.
            - y_track (np.ndarray): y-coordinates of the car on the track.
            - t_track (np.ndarray): Time points corresponding to the track motion.
            - end_track_flag (bool): Whether the car reached the end of the track.
            - at_rest_flag (bool): Whether the car came to rest.
            - x_end (float): x-coordinate at the end of the ramp.
            - y_end (float): y-coordinate at the end of the ramp.
            - speed_end (float): Speed of the car at the end of the ramp (if applicable).
            - theta_end (float): Slope angle at the end of the ramp (if applicable).
    """

    # Initial conditions
    p0 = ramp_params['p_initial']  # Initial path length position
    v0 = 0  # Initial velocity
    state0 = np.array([p0, v0])
    t_span = [0, 100]  # Time span of integration

    # Solve the ODE
    soln = solve_ivp(
        ramp_ode_fun, t_span, state0, method='RK45', dense_output=True, events=[end_track, at_rest], args=(ramp_params,),
        rtol = 1e-6, atol = 1e-9
    )

    # Check if the car reached the end of the track or came to rest
    end_track_flag = len(soln.t_events[0]) > 0
    at_rest_flag = len(soln.t_events[1]) > 0

    # Get the car position solution at specified times
    h = 0.01
    if end_track_flag:
        t = np.arange(t_span[0], soln.t_events[0][0], h)
    elif at_rest_flag:
        t_to_rest = soln.t_events[1][0]  # Record time to come to rest
        t = np.arange(t_span[0], t_to_rest + 2, h)  # Keep time to come to rest, plus 2 seconds
    else:
        t = np.arange(t_span[0], t_span[1], h)

    sol = soln.sol(t)
    p = sol[0]  # Extracting position in path length variable
    v = sol[1]  # Extracting velocity

    # Unpack path length maps
    p2x = ramp_params['p2x']
    p2y = ramp_params['p2y']
    p2k = ramp_params['p2k']

    # Save x, y position of car on track
    x_track = p2x(p)
    y_track = p2y(p)
    t_track = t

    if end_track_flag:
        # Get position, velocity, and slope at the end of the ramp
        x_end = x_track[-1]
        y_end = y_track[-1]
        speed_end = v[-1]
        theta_end = np.arctan2(p2k(p[-1]), 1.0)
    else:
        x_end = x_track[-1]
        y_end = y_track[-1]
        speed_end = None
        theta_end = None

    return {
        "state_track": sol,
        "x_track": x_track,
        "y_track": y_track,
        "t_track": t_track,
        "end_track_flag": end_track_flag,
        "at_rest_flag": at_rest_flag,
        "t_to_rest": t_to_rest if at_rest_flag else None,
        "x_end": x_end,
        "y_end": y_end,
        "speed_end": speed_end,
        "theta_end": theta_end,
    }

def integrate_ballistic_ode(params):
    """
    Integrates the ballistic ODE for the car's motion in the air after leaving the ramp.

    Parameters:
        params (dict): Dictionary containing:
            - end_track_flag (bool): Whether the car reached the end of the track.
            - x_end (float): x-coordinate at the end of the ramp.
            - y_end (float): y-coordinate at the end of the ramp.
            - speed_end (float): Speed of the car at the end of the ramp.
            - theta_end (float): Launch angle of the car at the end of the ramp.
            - x_track (np.ndarray): x-coordinates of the car on the track.
            - y_track (np.ndarray): y-coordinates of the car on the track.
            - t_track (np.ndarray): Time points corresponding to the track motion.
            - ballistic_ode_fun (callable): Function defining the ballistic ODE.
            - hit_target (callable): Event function for detecting when the car hits a target.

    Returns:
        dict: A dictionary containing:
            - x_full (np.ndarray): Full x-coordinates of the car (track + air).
            - y_full (np.ndarray): Full y-coordinates of the car (track + air).
            - t_full (np.ndarray): Full time points of the car's motion.
    """
    # Unpack parameters
    end_track_flag = params['end_track_flag']
    x_end = params['x_end']
    y_end = params['y_end']
    speed_end = params['speed_end']
    theta_end = params['theta_end']
    x_track = params['x_track']
    y_track = params['y_track']
    t_track = params['t_track']
    ballistic_ode_fun = params['ballistic_ode_fun']
    hit_target = params['hit_target']

    if end_track_flag:
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
        soln = solve_ivp(
            ballistic_ode_fun, t_span, state0, method='RK45', dense_output=True, events=hit_target,
            rtol = 1e-6, atol = 1e-9
        )

        # Plot solution at specified times
        h = 0.01
        t = np.arange(t_span[0], soln.t_events[0][0], h)
        sol = soln.sol(t)
        x_ballistic = sol[0]
        y_ballistic = sol[1]
        vx_ballistic = sol[2]
        vy_ballistic = sol[3]

        t_ballistic = t

        # Add x, y_ballistic to x, y_full
        x_full = np.concatenate((x_track, x_ballistic[1:-1]))
        y_full = np.concatenate((y_track, y_ballistic[1:-1]))
        t_full = np.concatenate((t_track, t_track[-1] + t_ballistic[1:-1]))
    else:
        x_full = x_track
        y_full = y_track
        t_full = t_track

    return {
        "x_full": x_full,
        "y_full": y_full,
        "t_full": t_full,
    }

def plot_and_animate_car_motion(params, ax=None):
    """
    Plots the motion of the car on the track and in the air, and optionally creates an animation.

    Parameters:
        params (dict): Dictionary containing:
            - xnodes, ynodes (np.ndarray): Spline nodes (in meters).
            - xs, ys (np.ndarray): Track coordinates (in meters).
            - x_full, y_full (np.ndarray): Full trajectory coordinates (in meters).
            - ps (np.ndarray): Path length array.
            - p2k (callable): Function mapping path length to slope (dy/dx).
            - n_long_track_segments (int): Number of long track segments.
            - ring_1, ring_2, ring_3 (list): Ring locations (in feet).
            - end_track_flag (bool): Whether the car reached the end of the track.
            - x_end, y_end (float): Coordinates at the end of the ramp (in meters).
            - ramp_type (str): Type of ramp ('halfpipe' or 'jump').
            - at_rest_flag (bool): Whether the car came to rest.
            - t_to_rest (float): Time taken for the car to come to rest.
            - t_full (np.ndarray): Full time points of the car's motion.
            - animation_flag (bool): Whether to create an animation.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If None, creates new figure and axes.

    Returns:
        None
    """
    # Unpack parameters
    xnodes = params['xnodes']
    ynodes = params['ynodes']
    xs = params['xs']
    ys = params['ys']
    x_full = params['x_full']
    y_full = params['y_full']
    ps = params['ps']
    p2k = params['p2k']
    n_long_track_segments = params['n_long_track_segments']
    ring_1 = params['ring_1']
    ring_2 = params['ring_2']
    ring_3 = params['ring_3']
    end_track_flag = params['end_track_flag']
    x_end = params['x_end']
    y_end = params['y_end']
    ramp_type = params['ramp_type']
    at_rest_flag = params['at_rest_flag']
    t_to_rest = params['t_to_rest']
    t_full = params['t_full']
    animation_flag = params['animation_flag']

    # Convert units to feet for plotting
    xnodes_ft = xnodes / convert_in_to_m / 12
    ynodes_ft = ynodes / convert_in_to_m / 12
    xs_ft = xs / convert_in_to_m / 12
    ys_ft = ys / convert_in_to_m / 12
    x_full_ft = x_full / convert_in_to_m / 12
    y_full_ft = y_full / convert_in_to_m / 12

    # Create ellipse objects for rings
    ring_1_ellipse = Ellipse(xy=ring_1, width=0.2, height=6 / 12, facecolor="none", edgecolor="k")
    ring_2_ellipse = Ellipse(xy=ring_2, width=0.2, height=6 / 12, facecolor="none", edgecolor="k")
    ring_3_ellipse = Ellipse(xy=ring_3, width=0.2, height=6 / 12, facecolor="none", edgecolor="k")

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200, layout='constrained')
    else:
        fig = ax.figure

    # Plot track, nodes, long segments, rings, and car trajectory
    ax.plot(xs_ft, ys_ft, color='orange', linewidth=2)  # Plot track
    ax.scatter(xnodes_ft, ynodes_ft, color='gray')  # Plot spline nodes

    # Plot initial position marker as an open green circle
    ax.scatter(x_full_ft[0], y_full_ft[0], color='green', marker='o', s=100, facecolors='none', edgecolors='green', linewidth=2, zorder=5)

    # Label the angle at the end of the ramp
    if end_track_flag:
        theta_end = np.arctan2(p2k(ps[-1]), 1.0)
        offset_x = 0.2  
        offset_y = -0.1 
        ax.text(
            x_end / convert_in_to_m / 12 + offset_x,
            y_end / convert_in_to_m / 12 + offset_y,
            f'{theta_end * 180 / np.pi:.2f}\u00b0',
            fontsize=8,
            color='black',
            ha='left'
        )

    # Plot diamond marker at each long track segment junction
    for i in range(int(n_long_track_segments) + 1):
        idx = np.argmin(np.abs(ps - i * long_track_segment_in * convert_in_to_m))
        ax.scatter(xs_ft[idx], ys_ft[idx], color='black', marker='d')

    # Plot rings if ramp type is 'jump'
    if ramp_type == 'jump':
        ax.add_patch(ring_1_ellipse)  # Plot ring 1
        ax.add_patch(ring_2_ellipse)  # Plot ring 2
        ax.add_patch(ring_3_ellipse)  # Plot ring 3

    # Plot car trajectory
    ax.plot(x_full_ft, y_full_ft, color='green', linestyle=':')  # Plot car trajectory
    ax.set_aspect('equal', 'box')  # Set x,y aspect ratio equal
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.grid(visible=True)
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 10)
    ax.set_xlabel('x [feet]')
    ax.set_ylabel('y [feet]')
    if at_rest_flag:
        ax.set_title(f'Time to come to rest: {t_to_rest:.2f}s')
        print(f'The car took approximately {t_to_rest:.2f} seconds to come to rest.')
    elif end_track_flag:
        x_landing = x_full[-1]
        ax.set_title(f'Distance traveled from end of track: {(x_landing - x_end) / convert_in_to_m:.2f} inches')
        print(f'The car traveled {(x_landing - x_end) / convert_in_to_m:.2f} inches from the end of the track!')

    # Save the plot
    plt.savefig('stunt_jump.png')

    # Optional animation
    if animation_flag:
        from matplotlib import rc
        rc('animation', html='jshtml')

        # Make frames for animation
        scatter = ax.scatter(x_full_ft, y_full_ft, color='k')
        label_string = f'Time = {t_full[0]}s'
        label = ax.text(np.median(x_full_ft), np.max(y_full_ft), label_string, ha='center', va='center', fontsize=12, color="Red")

        def update(i):
            scatter.set_offsets((x_full_ft[i], y_full_ft[i]))
            label.set_text(f'Time = {t_full[i]:.2f}s')
            return scatter,

        # Animate with desired framerate
        playback = 1  # 1 = real time
        fps = 24  # frames per second

        T_sim = t_full[-1] - t_full[0]  # Total simulation time

        nframes = fps * T_sim / playback
        step = int(len(x_full_ft) // nframes)
        frames = np.arange(0, len(x_full_ft), step)

        interval = 1 / fps * 1000  # frame delay in ms
        ani = animation.FuncAnimation(
            fig, update, frames=frames, interval=interval, blit=True
        )

        ani.save('stunt_jump.mp4', writer='ffmpeg', fps=fps)

def simulate_stunt_jump(params):
    """
    Performs a complete simulation of the stunt jump, including:
    1. Computing path length and maps
    2. Integrating the ramp ODE
    3. Integrating the ballistic ODE (if applicable)
    4. Plotting and animating the car motion

    Parameters:
        params (dict): Dictionary containing:
            - friction_factor (float): Friction factor for the ramp
            - p_initial (float): Initial position on the ramp (in meters)
            - ramp_type (str): Type of ramp ('halfpipe' or 'jump')
            - animation_flag (bool): Whether to create an animation
            - anchor_locations (np.ndarray): Array of [x,y] coordinates for ramp anchors (in feet)
            - ring_1, ring_2, ring_3 (list): Ring locations (in feet)
            - save_inverted_image (bool): Whether to save an inverted version of the plot (default: False)

    Returns:
        dict: A dictionary containing all simulation results
    """
    # Unpack parameters
    friction_factor = params['friction_factor']
    p_initial = params['p_initial']
    ramp_type = params['ramp_type']
    animation_flag = params['animation_flag']
    anchor_locations = params['anchor_locations']
    ring_1 = params['ring_1']
    ring_2 = params['ring_2']
    ring_3 = params['ring_3']
    save_inverted_image = params.get('save_inverted_image', False)  # Default to False if not specified

    # Extract positions of spline nodes
    xnodes = np.array(anchor_locations[:,0])
    ynodes = np.array(anchor_locations[:,1])

    # Create a cubic spline between anchor locations
    cs = CubicSpline(xnodes, ynodes, bc_type = 'not-a-knot')
    xs = np.linspace(np.min(xnodes), np.max(xnodes), int(1e3))
    ys = cs(xs)

    # Convert to meters
    xnodes = xnodes * convert_in_to_m # m
    ynodes = ynodes * convert_in_to_m # m
    xs = xs * convert_in_to_m # m
    ys = ys * convert_in_to_m # m

    # Compute path length and maps.
    path_maps = compute_path_length_and_maps(xs, ys)

    # Define ramp parameters
    ramp_params = {
        'friction_factor': friction_factor,
        'p_initial': p_initial,
        'ps': path_maps["ps"],
        'p2x': path_maps["p2x"],
        'p2y': path_maps["p2y"],
        'p2k': path_maps["p2k"],
        'p2kappa': path_maps["p2kappa"],
        'y_track_min': path_maps["y_track_min"],
    }

    # Integrate ramp ODE
    ramp_result = integrate_ramp_ode(
        ramp_params=ramp_params,
        ramp_ode_fun=ramp_ode_fun,
        end_track=end_track,
        at_rest=at_rest,
    )

    # Integrate ballistic ODE
    ballistic_params = {
        'end_track_flag': ramp_result["end_track_flag"],
        'x_end': ramp_result["x_end"],
        'y_end': ramp_result["y_end"],
        'speed_end': ramp_result["speed_end"],
        'theta_end': ramp_result["theta_end"],
        'x_track': ramp_result["x_track"],
        'y_track': ramp_result["y_track"],
        't_track': ramp_result["t_track"],
        'ballistic_ode_fun': ballistic_ode_fun,
        'hit_target': hit_target,
    }

    ballistic_result = integrate_ballistic_ode(ballistic_params)

    # Plot and animate results
    plot_params = {
        'xnodes': xnodes,
        'ynodes': ynodes,
        'xs': path_maps["xs"],
        'ys': path_maps["ys"],
        'x_full': ballistic_result["x_full"],
        'y_full': ballistic_result["y_full"],
        'ps': path_maps["ps"],
        'p2k': path_maps["p2k"],
        'n_long_track_segments': path_maps["n_long_track_segments"],
        'ring_1': ring_1,
        'ring_2': ring_2,
        'ring_3': ring_3,
        'end_track_flag': ramp_result["end_track_flag"],
        'x_end': ramp_result["x_end"],
        'y_end': ramp_result["y_end"],
        'ramp_type': ramp_type,
        'at_rest_flag': ramp_result["at_rest_flag"],
        't_to_rest': ramp_result["t_to_rest"],
        't_full': ballistic_result["t_full"],
        'animation_flag': animation_flag
    }

    plot_and_animate_car_motion(plot_params)

    # Plot kinetic and potential energy over time
    E, KE, PE = total_energy(ramp_result["state_track"], ramp_params)
    E_tolerance = 1e-3
    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(ramp_result["t_track"], E, label='Total Energy', color='blue')
    plt.plot(ramp_result["t_track"], KE, label='Kinetic Energy', color='red')
    plt.plot(ramp_result["t_track"], PE, label='Potential Energy', color='green')
    plt.axhline(E_tolerance, color='black', linestyle='--', linewidth=1)
    plt.axvline(ramp_result["t_to_rest"], color='black', linestyle='--', linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J/kg]')
    plt.title('Energy vs Time')
    plt.legend()
    plt.grid()
    plt.savefig('energy_vs_time.png')
    
    # Zoom in on the plot
    plt.xlim(0.9*ramp_result["t_to_rest"], 1.1*ramp_result["t_to_rest"])
    plt.ylim(0, 2*E_tolerance)
    plt.savefig('energy_vs_time_zoomed.png')
    plt.close()

    # Save inverted image if requested
    if save_inverted_image:
        img = Image.open('stunt_jump.png').convert('RGB')
        img_inv = ImageOps.invert(img)
        img_inv.save('stunt_jump_inv.png')

    # Return all results
    return {
        'path_maps': path_maps,
        'ramp_result': ramp_result,
        'ballistic_result': ballistic_result,
        'plot_params': plot_params
    }