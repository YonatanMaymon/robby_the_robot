import numpy as np

MUTATION_FACTOR = 30
DNA_SIZE = 243

class Agent:
    def __init__(self, dna: np.ndarray = None):
        self.dna = dna if dna is not None else self.generate_random_dna()
        self.position = (0,0)
        self.score = 0

    def add_score(self, add):
        self.score += add

    def sexually_reproduce(self, partner: Agent):
        partners_genes_locations = np.random.choice(DNA_SIZE, (DNA_SIZE+1)//2)
        # self copy of genes
        child_dna = self.dna.copy()
        # adds partners share
        child_dna[partners_genes_locations] = partner.dna[partners_genes_locations]
        # create child and add random mutations to his genes
        child = Agent(dna= child_dna)
        child.mutate()
        return child

    def mutate(self):
        mutation_locations = np.random.choice(DNA_SIZE, MUTATION_FACTOR)
        self.dna[mutation_locations] = np.random.randint(0, 7, size= MUTATION_FACTOR)

    def generate_random_dna(self):
        return np.random.randint(0,7,size= DNA_SIZE)