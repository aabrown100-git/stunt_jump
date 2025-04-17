import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QPainterPath
from scipy.interpolate import CubicSpline

class SplineEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spline Editor")
        self.setMinimumSize(800, 600)
        
        # Initialize variables
        self.nodes = []
        self.dragging_node = None
        self.spline_path = None
        self.trajectory_path = None
        self.coordinates = None
        
        # Default node locations in feet
        self.default_nodes = np.array([
            [0, 2.5],  # feet
            [1, 1],
            [2, 0.25],
            [3, 0.25],
            [4, 1]
        ])
        
        # Convert to pixels
        self.ppi = 85.5
        self.nodes = self.default_nodes * self.ppi
        self.nodes[:, 0] = self.nodes[:, 0] + 74
        self.nodes[:, 1] = 453 - self.nodes[:, 1]
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create canvas widget
        self.canvas = Canvas(self)
        layout.addWidget(self.canvas)
        
        # Create simulate button
        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.simulate)
        layout.addWidget(self.simulate_button)
        
        # Load background image
        self.background = QPixmap("background_image.png")
        if self.background.isNull():
            print("Warning: Could not load background image")
        
        # Update spline
        self.update_spline()
    
    def update_spline(self):
        # Create spline through nodes
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        cs = CubicSpline(x, y, bc_type='not-a-knot')
        
        # Create path for spline
        self.spline_path = QPainterPath()
        t = np.linspace(min(x), max(x), 1000)
        points = [(t[0], cs(t[0]))]
        for i in range(1, len(t)):
            points.append((t[i], cs(t[i])))
        
        self.spline_path.moveTo(*points[0])
        for point in points[1:]:
            self.spline_path.lineTo(*point)
        
        # Update canvas
        self.canvas.update()
    
    def simulate(self):
        # Convert node positions back to feet
        node_positions = self.nodes.copy()
        node_positions[:, 0] = (node_positions[:, 0] - 74) / self.ppi
        node_positions[:, 1] = (453 - node_positions[:, 1]) / self.ppi
        
        # Store coordinates for later use
        self.coordinates = node_positions
        
        # Close the window
        self.close()

class Canvas(QWidget):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.setMinimumSize(800, 500)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        if not self.editor.background.isNull():
            painter.drawPixmap(0, 0, self.editor.background)
        
        # Draw spline
        if self.editor.spline_path:
            painter.setPen(QPen(QColor("blue"), 2))
            painter.drawPath(self.editor.spline_path)
        
        # Draw nodes
        painter.setPen(QPen(QColor("black"), 2))
        for node in self.editor.nodes:
            painter.drawEllipse(QPointF(node[0], node[1]), 5, 5)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position()
            for i, node in enumerate(self.editor.nodes):
                if (pos.x() - node[0])**2 + (pos.y() - node[1])**2 <= 25:  # 5px radius
                    self.editor.dragging_node = i
                    break
    
    def mouseMoveEvent(self, event):
        if self.editor.dragging_node is not None:
            pos = event.position()
            self.editor.nodes[self.editor.dragging_node] = [pos.x(), pos.y()]
            self.editor.update_spline()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.editor.dragging_node = None

def edit_splines():
    app = QApplication(sys.argv)
    editor = SplineEditor()
    editor.show()
    app.exec()
    
    # Return node positions and coordinates
    if editor.coordinates is not None:
        # Create trajectory coordinates from the spline
        x = editor.nodes[:, 0]
        y = editor.nodes[:, 1]
        cs = CubicSpline(x, y, bc_type='not-a-knot')
        t = np.linspace(min(x), max(x), 1000)
        traj_coords = np.column_stack((t, cs(t)))
        return editor.coordinates, traj_coords
    return None, None

if __name__ == "__main__":
    edit_splines() 