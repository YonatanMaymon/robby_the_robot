from agent import Agent
from model import Model
import heapq

POPULATION = 500

class EvolutionManager:
    def __init__(self, model: Model):
        self.model = model
        self.generate_next_generation()

        

    def step(self):
        # finding the two best performing agents
        agent1, agent2 = heapq.nlargest(2, self.agents, key= lambda a: a.score)
        # saving the best agent so far
        print("best score is " + str(agent1.score))
        self.best_agent = Agent(agent1.dna)
        self.generate_next_generation(agent1, agent2)

    def generate_next_generation(self, agent1: Agent = None, agent2: Agent = None):
        agents = []
        population = POPULATION
        # save the best performing individuals from previous generation
        if agent1 and agent2:
            agents.append(Agent(agent1.dna))
            agents.append(Agent(agent2.dna))
            population -= 2
        # populate next generations with children of the two best preforming agents
        for _ in range(population):
            agent = agent1.sexually_reproduce(agent2) if agent1 else Agent()
            agents.append(agent)
        for agent in agents:
            self.model.score_agent(agent)
        self.agents = agents
        print("generated " + str(len(self.agents)) + " agents")
