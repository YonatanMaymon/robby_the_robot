
from evolutionManager import EvolutionManager
from model import Model

model = Model()

evolution_manager = EvolutionManager(model)

for i in range(2):
    print("generation: " + str(i))
    evolution_manager.step()


print(evolution_manager.best_agent.dna)
print("with score of " + str(evolution_manager.best_agent.score))

model.score_agent(evolution_manager.best_agent, True)