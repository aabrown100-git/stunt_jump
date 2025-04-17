import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QHBoxLayout, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from stunt_jump_functions import (simulate_stunt_jump, convert_in_to_m, 
                                compute_path_length_and_maps, long_track_segment_in,
                                plot_and_animate_car_motion)
from scipy.interpolate import CubicSpline

class StuntJumpGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stunt Jump Designer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize simulation parameters
        self.friction_factor = 0.1
        self.p_initial = 0.0 * convert_in_to_m
        self.ramp_type = 'jump'
        self.animation_flag = False
        self.show_ramp = True  # Flag to control ramp visibility
        
        # Store last simulation results
        self.last_simulation_results = None
        
        # Ring locations in feet
        self.ring_1 = [5, 1.75]
        self.ring_2 = [6, 1.6]
        self.ring_3 = [7, 0.9]
        
        # Initial anchor points in feet
        self.anchor_locations = np.array([
            [1, 3],
            [2, 1],
            [3, 1.5],
        ]) * 12  # Convert to inches
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Add control buttons
        self.add_point_button = QPushButton("Add Point")
        self.add_point_button.clicked.connect(self.add_point)
        control_layout.addWidget(self.add_point_button)
        
        self.remove_point_button = QPushButton("Remove Point")
        self.remove_point_button.clicked.connect(self.remove_point)
        control_layout.addWidget(self.remove_point_button)
        
        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.simulate)
        control_layout.addWidget(self.simulate_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        control_layout.addWidget(self.reset_button)

        # Add friction factor control
        friction_layout = QHBoxLayout()
        friction_layout.addWidget(QLabel("Friction Factor:"))
        self.friction_input = QLineEdit(str(self.friction_factor))
        friction_layout.addWidget(self.friction_input)
        control_layout.addLayout(friction_layout)

        # Add ramp visibility toggle
        self.toggle_ramp_button = QPushButton("Hide Ramp")
        self.toggle_ramp_button.setCheckable(True)
        self.toggle_ramp_button.clicked.connect(self.toggle_ramp)
        control_layout.addWidget(self.toggle_ramp_button)

        # Add ring controls
        ring_control_panel = QWidget()
        ring_control_layout = QVBoxLayout(ring_control_panel)
        
        # Ring 1 controls
        ring1_layout = QHBoxLayout()
        ring1_layout.addWidget(QLabel("Ring 1:"))
        self.ring1_x = QLineEdit(str(self.ring_1[0]))
        self.ring1_y = QLineEdit(str(self.ring_1[1]))
        ring1_layout.addWidget(QLabel("x:"))
        ring1_layout.addWidget(self.ring1_x)
        ring1_layout.addWidget(QLabel("y:"))
        ring1_layout.addWidget(self.ring1_y)
        ring_control_layout.addLayout(ring1_layout)
        
        # Ring 2 controls
        ring2_layout = QHBoxLayout()
        ring2_layout.addWidget(QLabel("Ring 2:"))
        self.ring2_x = QLineEdit(str(self.ring_2[0]))
        self.ring2_y = QLineEdit(str(self.ring_2[1]))
        ring2_layout.addWidget(QLabel("x:"))
        ring2_layout.addWidget(self.ring2_x)
        ring2_layout.addWidget(QLabel("y:"))
        ring2_layout.addWidget(self.ring2_y)
        ring_control_layout.addLayout(ring2_layout)
        
        # Ring 3 controls
        ring3_layout = QHBoxLayout()
        ring3_layout.addWidget(QLabel("Ring 3:"))
        self.ring3_x = QLineEdit(str(self.ring_3[0]))
        self.ring3_y = QLineEdit(str(self.ring_3[1]))
        ring3_layout.addWidget(QLabel("x:"))
        ring3_layout.addWidget(self.ring3_x)
        ring3_layout.addWidget(QLabel("y:"))
        ring3_layout.addWidget(self.ring3_y)
        ring_control_layout.addLayout(ring3_layout)
        
        # Add update button for rings
        self.update_rings_button = QPushButton("Update Rings")
        self.update_rings_button.clicked.connect(self.update_rings)
        ring_control_layout.addWidget(self.update_rings_button)
        
        control_layout.addWidget(ring_control_panel)
        
        layout.addWidget(control_panel)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Add status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Create single axis
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 5)
        self.ax.set_xlabel('x [feet]')
        self.ax.set_ylabel('y [feet]')
        self.ax.grid(True)
        
        # Initialize plot
        self.plot_initial_ramp(run_simulation=True)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.dragging = False
        self.dragged_point = None
        
    def plot_last_simulation(self):
        if self.last_simulation_results is not None:
            # Clear the axis
            self.ax.clear()
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 5)
            self.ax.set_xlabel('x [feet]')
            self.ax.set_ylabel('y [feet]')
            self.ax.grid(True)
            
            # Plot the last simulation results
            plot_params = {
                'xnodes': self.last_simulation_results['anchor_locations'][:, 0] * convert_in_to_m,
                'ynodes': self.last_simulation_results['anchor_locations'][:, 1] * convert_in_to_m,
                'xs': self.last_simulation_results['path_maps']['xs'],
                'ys': self.last_simulation_results['path_maps']['ys'],
                'x_full': self.last_simulation_results['ballistic_result']['x_full'],
                'y_full': self.last_simulation_results['ballistic_result']['y_full'],
                'ps': self.last_simulation_results['path_maps']['ps'],
                'p2k': self.last_simulation_results['path_maps']['p2k'],
                'n_long_track_segments': self.last_simulation_results['path_maps']['n_long_track_segments'],
                'ring_1': self.ring_1,
                'ring_2': self.ring_2,
                'ring_3': self.ring_3,
                'end_track_flag': self.last_simulation_results['ramp_result']['end_track_flag'],
                'x_end': self.last_simulation_results['ramp_result']['x_end'],
                'y_end': self.last_simulation_results['ramp_result']['y_end'],
                'ramp_type': self.ramp_type,
                'at_rest_flag': self.last_simulation_results['ramp_result']['at_rest_flag'],
                't_to_rest': self.last_simulation_results['ramp_result']['t_to_rest'],
                't_full': self.last_simulation_results['ballistic_result']['t_full'],
                'animation_flag': False
            }
            plot_and_animate_car_motion(plot_params, ax=self.ax)
            
    def plot_initial_ramp(self, run_simulation=True, clear_plot=True):
        if clear_plot:
            # Clear the axis
            self.ax.clear()
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 5)
            self.ax.set_xlabel('x [feet]')
            self.ax.set_ylabel('y [feet]')
            self.ax.grid(True)
            
            # Plot last simulation results if they exist
            self.plot_last_simulation()
        
        # Get ramp coordinates
        xs_ft, ys_ft = self.get_ramp_coordinates()
        
        if run_simulation:
            # Run simulation
            sim_params = {
                'friction_factor': self.friction_factor,
                'p_initial': self.p_initial,
                'ramp_type': self.ramp_type,
                'animation_flag': self.animation_flag,
                'anchor_locations': self.anchor_locations,
                'ring_1': self.ring_1,
                'ring_2': self.ring_2,
                'ring_3': self.ring_3,
                'save_inverted_image': True
            }
            
            results = simulate_stunt_jump(sim_params)
            
            # Store the results
            self.last_simulation_results = {
                'path_maps': results['path_maps'],
                'ballistic_result': results['ballistic_result'],
                'ramp_result': results['ramp_result'],
                'anchor_locations': self.anchor_locations.copy()
            }
            
            # Clear and update the plot
            self.plot_last_simulation()
            
            # Update status
            if results['ramp_result']['end_track_flag']:
                x_landing = results['ballistic_result']['x_full'][-1] / convert_in_to_m / 12
                x_end = results['ramp_result']['x_end'] / convert_in_to_m / 12
                distance = (x_landing - x_end) * 12  # Convert to inches
                self.status_label.setText(f"Simulation complete! Car traveled {distance:.2f} inches from the end of the track.")
            else:
                self.status_label.setText("Simulation complete! Car did not reach the end of the track.")
        
        # Plot the current ramp shape and control points if visible
        if self.show_ramp:
            self.ax.plot(xs_ft, ys_ft, color='orange', linewidth=2, label='Ramp', zorder=2)
            self.ax.plot(self.anchor_locations[:, 0]/12, self.anchor_locations[:, 1]/12, 'ro', label='Control Points', zorder=3)
        
        self.canvas.draw()
        
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
            
        # Check if we clicked near any anchor point
        for i, (x, y) in enumerate(self.anchor_locations):
            x_ft = x / 12
            y_ft = y / 12
            if abs(event.xdata - x_ft) < 0.2 and abs(event.ydata - y_ft) < 0.2:
                self.dragging = True
                self.dragged_point = i
                break
                
    def on_release(self, event):
        self.dragging = False
        self.dragged_point = None
        
    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
            
        # Update the dragged point's position
        if self.dragged_point is not None:
            self.anchor_locations[self.dragged_point] = [
                event.xdata * 12,  # Convert to inches
                event.ydata * 12
            ]
            
            # Clear only the control points and ramp
            for artist in self.ax.lines:
                if artist.get_label() in ['Control Points', 'Ramp']:
                    artist.remove()
            
            # Get new ramp coordinates
            xs_ft, ys_ft = self.get_ramp_coordinates()
            
            # Plot new ramp and control points if visible
            if self.show_ramp:
                self.ax.plot(xs_ft, ys_ft, color='orange', linewidth=2, label='Ramp', zorder=2)
                self.ax.plot(self.anchor_locations[:, 0]/12, self.anchor_locations[:, 1]/12, 'ro', label='Control Points', zorder=3)
            
            self.canvas.draw()
            
    def add_point(self):
        if len(self.anchor_locations) >= 2:
            # Get the last two points
            last_point = self.anchor_locations[-1]
            second_last_point = self.anchor_locations[-2]
            
            # Calculate midpoint
            new_x = (last_point[0] + second_last_point[0]) / 2
            new_y = (last_point[1] + second_last_point[1]) / 2
            
            # Add new point
            self.anchor_locations = np.vstack([self.anchor_locations, [new_x, new_y]])
            
            # Sort points by x-coordinate
            sort_indices = np.argsort(self.anchor_locations[:, 0])
            self.anchor_locations = self.anchor_locations[sort_indices]
            
            # Update the plot
            self.plot_initial_ramp(run_simulation=False, clear_plot=True)  # Clear plot to show only last simulation
            self.status_label.setText(f"Added new point at ({new_x/12:.2f}, {new_y/12:.2f}) feet")
        else:
            self.status_label.setText("Need at least 2 points to add a new one")
            
    def remove_point(self):
        if len(self.anchor_locations) > 3:
            # Remove the last point
            removed_point = self.anchor_locations[-1]
            self.anchor_locations = self.anchor_locations[:-1]
            
            # Update the plot
            self.plot_initial_ramp(run_simulation=False, clear_plot=True)  # Clear plot to show only last simulation
            self.status_label.setText(f"Removed point at ({removed_point[0]/12:.2f}, {removed_point[1]/12:.2f}) feet")
        else:
            self.status_label.setText("Need at least 3 points to remove one")
            
    def get_ramp_coordinates(self):
        # Extract positions of spline nodes
        xnodes = self.anchor_locations[:, 0]  # inches
        ynodes = self.anchor_locations[:, 1]  # inches
        
        # Create a cubic spline between anchor locations
        cs = CubicSpline(xnodes, ynodes, bc_type='not-a-knot')
        xs = np.linspace(np.min(xnodes), np.max(xnodes), 1000)
        ys = cs(xs)
        
        # Convert to meters for compute_path_length_and_maps
        xs_m = xs * convert_in_to_m
        ys_m = ys * convert_in_to_m
        
        # Get the correct spline coordinates
        path_maps = compute_path_length_and_maps(xs_m, ys_m)
        
        # Convert back to feet for plotting
        xs_ft = path_maps["xs"] / convert_in_to_m / 12
        ys_ft = path_maps["ys"] / convert_in_to_m / 12
        
        return xs_ft, ys_ft
        
    def simulate(self):
        try:
            # Update friction factor from input
            self.friction_factor = float(self.friction_input.text())
            self.status_label.setText("Simulating...")
            QApplication.processEvents()
            
            # Clear last simulation results before running new simulation
            self.last_simulation_results = None
            
            # Update the plot with simulation
            self.plot_initial_ramp(run_simulation=True)
        except ValueError:
            self.status_label.setText("Invalid friction factor value")

    def reset(self):
        # Reset to initial anchor points
        self.anchor_locations = np.array([
            [1, 3],
            [2, 1],
            [3, 1.5],
        ]) * 12  # Convert to inches
        
        # Reset friction factor
        self.friction_factor = 0.1
        self.friction_input.setText(str(self.friction_factor))
        
        # Reset ramp visibility
        self.show_ramp = True
        self.toggle_ramp_button.setChecked(False)
        self.toggle_ramp_button.setText("Hide Ramp")
        
        # Clear last simulation results
        self.last_simulation_results = None
        
        # Redraw the plot
        self.plot_initial_ramp(run_simulation=True, clear_plot=True)
        self.status_label.setText("Reset to initial configuration")

    def update_rings(self):
        try:
            # Update ring locations from input fields
            self.ring_1 = [float(self.ring1_x.text()), float(self.ring1_y.text())]
            self.ring_2 = [float(self.ring2_x.text()), float(self.ring2_y.text())]
            self.ring_3 = [float(self.ring3_x.text()), float(self.ring3_y.text())]
            
            # Redraw the plot with new ring positions
            self.plot_initial_ramp(run_simulation=False, clear_plot=True)
            self.status_label.setText("Ring positions updated")
        except ValueError:
            self.status_label.setText("Invalid ring position values")

    def toggle_ramp(self):
        self.show_ramp = not self.show_ramp
        self.toggle_ramp_button.setText("Show Ramp" if not self.show_ramp else "Hide Ramp")
        self.plot_initial_ramp(run_simulation=False, clear_plot=True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StuntJumpGUI()
    window.show()
    sys.exit(app.exec()) 