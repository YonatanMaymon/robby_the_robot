import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING
from consts import WIDTH, HEIGHT
if TYPE_CHECKING:
    from model import Model
    from agent import Agent

class Front:
    def __init__(self, model: Model, agent: Agent):
        self.model = model
        self.agent = agent

    def define_grid(self):
        fig, ax = plt.subplots(figsize=(6, 6))

        # CRITICAL: Lock vmin and vmax so color 0, 1, and 2 always map to the exact same colors
        self.image = ax.imshow(self.model.grid, cmap='Blues', vmin=0, vmax=2)

        # Add lines between cells (grid grid lines sit at half-step boundaries)
        ax.set_xticks(np.arange(-0.5, WIDTH, 1))
        ax.set_yticks(np.arange(-0.5, HEIGHT, 1))
        ax.grid(color='black', linestyle='-', linewidth=1)

        # Remove number labels on axes for a clean board look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.invert_yaxis()
        # Overlay Agent marker on top
        self.agent_marker, = ax.plot([], [], marker='o', color='red', markersize=16, label='Agent')
        ax.legend(loc='upper right')
        plt.title("2D Matrix Board")
        plt.ion()
        plt.show()

    def step(self):
        self.image.set_data(self.model.grid_copy)
        x, y = self.agent.position
        self.agent_marker.set_data([x],[y])
        plt.pause(0.1)
    def show(self):
        plt.ioff()
        plt.show()