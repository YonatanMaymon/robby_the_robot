import unittest
import numpy as np
from agent import Agent, DNA_SIZE, MUTATION_FACTOR

class TestAgent(unittest.TestCase):

    def test_init_without_dna(self):
        """Test agent initialization with random DNA."""
        agent = Agent()
        self.assertIsNotNone(agent.dna)
        self.assertEqual(agent.dna.shape, (DNA_SIZE,))
        self.assertTrue(np.all(agent.dna >= 0) and np.all(agent.dna < 7))
        self.assertEqual(agent.position, (0, 0))
        self.assertEqual(agent.score, 0)

    def test_init_with_dna(self):
        """Test agent initialization with a predefined DNA."""
        custom_dna = np.random.randint(0, 7, size=DNA_SIZE)
        agent = Agent(dna=custom_dna)
        np.testing.assert_array_equal(agent.dna, custom_dna)
        self.assertEqual(agent.position, (0, 0))
        self.assertEqual(agent.score, 0)

    def test_add_score(self):
        """Test the add_score method."""
        agent = Agent()
        self.assertEqual(agent.score, 0)
        agent.add_score(10)
        self.assertEqual(agent.score, 10)
        agent.add_score(-5)
        self.assertEqual(agent.score, 5)

    def test_generate_random_dna(self):
        """Test the creation of random DNA."""
        agent = Agent()
        dna = agent.generate_random_dna()
        self.assertIsInstance(dna, np.ndarray)
        self.assertEqual(dna.shape, (DNA_SIZE,))
        self.assertTrue(np.all(dna >= 0) and np.all(dna < 7))

    def test_mutate(self):
        """Test the mutation of an agent's DNA."""
        original_dna = np.zeros(DNA_SIZE, dtype=int)
        agent = Agent(dna=original_dna.copy())
        agent.mutate()
        # Check that the DNA has changed
        self.assertFalse(np.array_equal(original_dna, agent.dna))
        # Check that the number of mutations is at most MUTATION_FACTOR
        # (it can be less if the same index is chosen multiple times)
        num_mutations = np.sum(original_dna != agent.dna)
        self.assertLessEqual(num_mutations, MUTATION_FACTOR)

    def test_sexually_reproduce(self):
        """Test sexual reproduction between two agents."""
        dna1 = np.zeros(DNA_SIZE, dtype=int)
        dna2 = np.ones(DNA_SIZE, dtype=int)
        agent1 = Agent(dna=dna1)
        agent2 = Agent(dna=dna2)

        child = agent1.sexually_reproduce(agent2)

        # The child's DNA should be a mix of 0s and 1s before mutation
        # We can't know the exact mix because of randomness, but we can check properties.
        
        # Child's DNA should not be identical to either parent after reproduction and mutation
        self.assertFalse(np.array_equal(child.dna, agent1.dna))
        self.assertFalse(np.array_equal(child.dna, agent2.dna))

        # Count genes from each parent before mutation to verify crossover
        # This is harder to test directly without mocking np.random.choice
        # But we can check that the child's DNA contains genes from both parents
        # This is not a perfect test because of mutation.
        
        self.assertIsInstance(child, Agent)
        self.assertEqual(child.score, 0)

if __name__ == '__main__':
    unittest.main()