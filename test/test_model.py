import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from model import Model, Action, CAN_BONUS, EMPTY_SITE_PICKUP_PENALTY, WALL_CRASH_PENALTY
from agent import Agent, DNA_SIZE

class TestModel(unittest.TestCase):

    def setUp(self):
        """Set up a model and an agent for testing."""
        # Use a fixed seed for predictable random can placement
        np.random.seed(42)
        self.model = Model(10, 10, 20)
        self.agent = Agent()

    def test_init_and_populate_grid(self):
        """Test model initialization and grid population."""
        self.assertEqual(self.model.width, 10)
        self.assertEqual(self.model.height, 10)
        self.assertEqual(self.model.grid.shape, (10, 10))
        # Check if the correct number of cans were placed
        self.assertEqual(np.sum(self.model.grid), 20)

    def test_define_state(self):
        """Test the state definition logic."""
        # Create a specific grid scenario
        self.model.grid_copy = np.zeros((10, 10), dtype=bool)
        # Place cans in specific locations for testing
        self.model.grid_copy[1, 2] = True  # Up
        self.model.grid_copy[2, 1] = True  # Left
        self.model.grid_copy[2, 2] = True  # Current
        self.model.grid_copy[3, 2] = True  # Down
        # Right is a wall

        # Test state at an edge
        agent_pos = (9, 2)
        # State array: [Left, Right, Up, Down, Current]
        # Expected: [grid(8,2)=0, Wall=2, grid(9,1)=0, grid(9,3)=0, grid(9,2)=0] -> [0, 2, 0, 0, 0]
        # Base 3 conversion: 0*81 + 2*27 + 0*9 + 0*3 + 0*1 = 54
        self.assertEqual(self.model.define_state(agent_pos), 54)

        # Test state surrounded by cans
        agent_pos = (2, 2)
        # Expected: [grid(1,2)=1, grid(3,2)=1, grid(2,1)=1, grid(2,3)=0, grid(2,2)=1] -> [1, 1, 1, 0, 1]
        # Base 3 conversion: 1*81 + 1*27 + 1*9 + 0*3 + 1*1 = 81 + 27 + 9 + 1 = 118
        self.assertEqual(self.model.define_state(agent_pos), 118)

    def test_do_action_move_up(self):
        """Test moving up."""
        self.agent.position = (5, 5)
        self.model.do_action(self.agent, Action.UP)
        self.assertEqual(self.agent.position, (5, 4))
        self.assertEqual(self.agent.score, 0)

    def test_do_action_move_up_into_wall(self):
        """Test crashing into the top wall."""
        self.agent.position = (5, 0)
        self.model.do_action(self.agent, Action.UP)
        self.assertEqual(self.agent.position, (5, 0)) # Position doesn't change
        self.assertEqual(self.agent.score, WALL_CRASH_PENALTY)

    def test_do_action_move_left(self):
        """Test moving left."""
        self.agent.position = (5, 5)
        self.model.do_action(self.agent, Action.LEFT)
        self.assertEqual(self.agent.position, (4, 5))
        self.assertEqual(self.agent.score, 0)

    def test_do_action_move_left_into_wall(self):
        """Test crashing into the left wall."""
        self.agent.position = (0, 5)
        self.model.do_action(self.agent, Action.LEFT)
        self.assertEqual(self.agent.position, (0, 5))
        self.assertEqual(self.agent.score, WALL_CRASH_PENALTY)

    def test_do_action_pickup_can(self):
        """Test picking up a can."""
        self.model.grid_copy = self.model.grid.copy()
        pos_with_can = (5, 5)
        self.model.grid_copy[pos_with_can] = True
        self.agent.position = pos_with_can

        self.model.do_action(self.agent, Action.PICK_UP)

        self.assertEqual(self.agent.score, CAN_BONUS)
        self.assertFalse(self.model.grid_copy[pos_with_can]) # Can should be gone

    def test_do_action_pickup_empty(self):
        """Test picking up from an empty spot."""
        self.model.grid_copy = self.model.grid.copy()
        pos_without_can = (5, 5)
        self.model.grid_copy[pos_without_can] = False
        self.agent.position = pos_without_can

        self.model.do_action(self.agent, Action.PICK_UP)

        self.assertEqual(self.agent.score, EMPTY_SITE_PICKUP_PENALTY)
        self.assertFalse(self.model.grid_copy[pos_without_can])

    @patch('random.randint', return_value=Action.RIGHT.value)
    def test_do_action_random(self, mock_randint):
        """Test a random action."""
        self.agent.position = (5, 5)
        self.model.do_action(self.agent, Action.RANDOM)
        mock_randint.assert_called_once_with(1, 4)
        self.assertEqual(self.agent.position, (6, 5)) # Moves right based on mocked return

    def test_act(self):
        """Test the main act method."""
        # Set up a known state and DNA
        self.model.grid_copy = np.zeros((10, 10), dtype=bool)
        self.agent.position = (0, 0)
        state_num = self.model.define_state(self.agent.position)

        # Set DNA for the computed state to be Action.DOWN
        self.agent.dna[state_num] = Action.DOWN.value

        # Mock do_action to verify it's called correctly
        self.model.do_action = MagicMock()

        self.model.act(self.agent)

        self.model.do_action.assert_called_once_with(agent=self.agent, action=Action.DOWN)

    def test_score_agent_loop(self):
        """Test the agent scoring loop."""
        # Create an agent that always picks up
        dna = np.full(DNA_SIZE, Action.PICK_UP.value, dtype=int)
        agent = Agent(dna=dna)
        
        # Create a grid with one can where the agent starts
        self.model.grid = np.zeros((10, 10), dtype=bool)
        self.model.grid[0, 0] = True

        # Score the agent (without visualization)
        self.model.score_agent(agent, is_show=False)

        # Agent should pick up the can on the first step (time=0) and get 10 points.
        # The loop condition `np.any(self.grid_copy)` will then be false, so it terminates.
        self.assertEqual(agent.score, CAN_BONUS)


if __name__ == '__main__':
    unittest.main()