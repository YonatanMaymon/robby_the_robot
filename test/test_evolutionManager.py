import unittest
from unittest.mock import MagicMock, patch, call
from evolutionManager import EvolutionManager, POPULATION
from agent import Agent
import numpy as np

class TestEvolutionManager(unittest.TestCase):

    def setUp(self):
        """Set up a mock model and an EvolutionManager instance for each test."""
        self.mock_model = MagicMock()
        # The manager's __init__ calls generate_next_generation, which calls score_agent.
        # We patch 'agent.Agent' to prevent this initial scoring from complicating our setup.
        with patch('evolutionManager.Agent', side_effect=[Agent() for _ in range(POPULATION)]):
            self.evolution_manager = EvolutionManager(self.mock_model)
        
        # Reset the mock for the actual test assertions
        self.mock_model.reset_mock()

    def test_init(self):
        """Test the initialization of the EvolutionManager."""
        # __init__ creates an initial population.
        self.assertEqual(len(self.evolution_manager.agents), POPULATION)
        # Check that all agents are instances of the Agent class
        self.assertTrue(all(isinstance(agent, Agent) for agent in self.evolution_manager.agents))

    @patch('evolutionManager.Agent')
    def test_generate_next_generation_initial(self, mock_agent_class):
        """Test creating the first generation of agents."""
        # We want to control the agent instances created
        mock_agents = [MagicMock() for _ in range(POPULATION)]
        mock_agent_class.side_effect = mock_agents

        # Call the method to test
        self.evolution_manager.generate_next_generation()

        # Assertions
        self.assertEqual(mock_agent_class.call_count, POPULATION)
        self.assertEqual(self.mock_model.score_agent.call_count, POPULATION)
        # Ensure score_agent was called on each new agent
        self.mock_model.score_agent.assert_has_calls([call(agent) for agent in mock_agents])
        self.assertEqual(len(self.evolution_manager.agents), POPULATION)

    def test_step(self):
        """Test a single evolution step."""
        # Create a population with predictable scores
        for i, agent in enumerate(self.evolution_manager.agents):
            agent.score = i

        # The two agents with the highest scores will have scores POPULATION-1 and POPULATION-2
        best_agent_1 = self.evolution_manager.agents[POPULATION - 1]
        best_agent_2 = self.evolution_manager.agents[POPULATION - 2]

        # Mock the generate_next_generation method to check if it's called correctly
        self.evolution_manager.generate_next_generation = MagicMock()

        # Run the step
        self.evolution_manager.step()

        # Check that best_agent is set correctly
        self.assertIs(self.evolution_manager.best_agent, best_agent_1)

        # Check that the next generation is created from the two best agents
        self.evolution_manager.generate_next_generation.assert_called_once_with(best_agent_1, best_agent_2)

    @patch('agent.Agent.sexually_reproduce')
    def test_generate_next_generation_with_parents(self, mock_reproduce):
        """Test creating a new generation from parent agents."""
        agent1 = Agent(dna=np.zeros(243))
        agent2 = Agent(dna=np.ones(243))

        self.evolution_manager.generate_next_generation(agent1, agent2)

        self.assertEqual(len(self.evolution_manager.agents), POPULATION)
        # Check that the parents' DNA is carried over to the new generation
        np.testing.assert_array_equal(self.evolution_manager.agents[0].dna, agent1.dna)
        np.testing.assert_array_equal(self.evolution_manager.agents[1].dna, agent2.dna)
        # Check that reproduction was called for the rest of the population
        self.assertEqual(mock_reproduce.call_count, POPULATION - 2)
        # Check that new children were scored
        self.assertEqual(self.mock_model.score_agent.call_count, POPULATION - 2)

if __name__ == '__main__':
    unittest.main()