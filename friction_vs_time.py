# friction_vs_time.py
import numpy as np
import matplotlib.pyplot as plt
from stunt_jump_functions import simulate_stunt_jump, convert_in_to_m


def main():
    # Define anchor locations (in feet converted to inches)
    anchor_locations = np.array([
        [0, 1],
        [1, 0],
        [2, 1],
    ]) * 12

    # Ring positions (feet)
    ring_1 = [5, 1.75]
    ring_2 = [6, 1.6]
    ring_3 = [7, 0.9]

    # Initial position on ramp (in meters)
    p_initial = 0.0 * convert_in_to_m

    # Sweep friction factors
    friction_factors = np.linspace(0.04, 0.06, 50)
    times_to_rest = []

    for mu in friction_factors:
        print(f'mu: {mu}')
        params = {
            'friction_factor': mu,
            'p_initial':       p_initial,
            'ramp_type':       'halfpipe',    # stops on ramp
            'animation_flag':  False,
            'anchor_locations': anchor_locations,
            'ring_1':           ring_1,
            'ring_2':           ring_2,
            'ring_3':           ring_3
        }

        results = simulate_stunt_jump(params)
        t_rest = results['ramp_result']['t_to_rest']
        times_to_rest.append(t_rest if t_rest is not None else np.nan)

    # Plot results
    plt.figure(figsize=(8,5))
    plt.plot(friction_factors, times_to_rest, 'o-', color='tab:blue')
    plt.xlabel('Friction factor μ')
    plt.ylabel('Time to come to rest [s]')
    plt.title('Half-pipe: Time to rest vs. friction')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('friction_vs_time.png', dpi=150)


if __name__ == '__main__':
    main() 