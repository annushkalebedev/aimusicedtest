import sys
import streamlit as st
import music21 as m2 
import streamlit as st
import numpy as np
from params import *
from tqdm import tqdm
import crash


class GA(object):
    def __init__(self, n_measures, n_per_measure, weights, m1):
        self.n_measures = n_measures
        self.n_per_measure = n_per_measure
        self.weights = weights
        self.m1 = m1
        self.crossover_type = "uniform"

        self.key_scale = [p.pitchClass for p in key_sigs[st.session_state.key_signature].pitches]

        return 

    # pop: a list of midi pitches: (#of measures * #eighths in measures)
    def cal_pop_fitness(self, pop):

        fitness = []
        for sol in pop:
            score = 0
            notes = sol[np.nonzero(sol)] # pitched notes in the list
            pitch_classes = notes % 12

            # in range: [-64, 0]
            score -= sum([abs(note) for note in sol if (
                (note != 0) and (note < 24) or (note > 96))])

            # note-rest percentage: [-400, 0]
            note_len = len(notes)
            rest_len = len(sol) - note_len
            score -= (abs(note_len - rest_len) / len(sol)) * 400

            # in key: [0, 1280]
            score += (len([pitch for pitch in pitch_classes if pitch in self.key_scale])
                * 80 * self.weights['key_weight'])

            # interval smooth: [-256, 0]
            score -= sum(np.abs(np.diff(notes))) * 4 * self.weights['smoothing_weight']


            # note on beat and downbeat: [0, 60]
            beats = sol[np.arange(0, len(sol), 2)]
            pitched_beats = beats[np.nonzero(beats)]
            score += (len(pitched_beats) / len(beats)) * 30 * self.weights['rhythm_weight']

            downbeats = sol[np.arange(0, len(sol), self.n_per_measure)]
            pitched_downbeats = beats[np.nonzero(downbeats)]
            score += (len(pitched_beats) / len(beats)) * 30 * self.weights['rhythm_weight']     

            # similarity to existing melody: []
            m1 = self.m1[:len(sol)]
            score -= sum(np.abs(sol - np.array(m1))) * self.weights['similarity_weight']

            # cadence

            # long note ending: [0, 450]
            i = len(sol) - 1
            while sol[i] == 0:
                i -= 1
            score += min((len(sol) - 1 - i) * 150, 450)


            fitness.append(score)

        return np.array(fitness).T

    def select_mating_pool(self, pop, fitness, num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -1e8
        return parents

    def crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.uint8(offspring_size[1]/2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]

            if self.crossover_type == 'uniform':
                for i in range(offspring_size[1]):
                    parent_idx = np.random.choice([parent1_idx, parent2_idx])
                    offspring[k, i] = parents[parent_idx, i]
                    # crash()

            elif self.crossover_type == "one_point":
                # The new offspring will have its first half of its genes taken from the first parent.
                offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
                # The new offspring will have its second half of its genes taken from the second parent.
                offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

            
        return offspring

    def mutation(self, offspring_crossover, num_mutations=1):
        mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)

        # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
        for idx in range(offspring_crossover.shape[0]):
            gene_idx = mutations_counter - 1
            for mutation_num in range(num_mutations):

                # The random value to be added to the gene.
                random_value = np.random.randint(low=-6.0, high=6.0, size=1) * 2
                # if offspring_crossover[idx, gene_idx]:
                offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
                gene_idx = gene_idx + mutations_counter
        return offspring_crossover



# Number of the weights we are looking to optimize.
# n bars, each bar has m tokens, n * m tokens in general
def generate_development(ga, bridge=False):
    num_weights = int((ga.n_measures * ga.n_per_measure) / (2 if bridge else 1))
    pop_size = (st.session_state.sol_per_pop,num_weights) 

    #Creating the initial population.
    new_population = np.random.randint(low=48, high=72, size=pop_size)
    rests = np.random.choice([0, 1], new_population.shape)
    new_population = np.multiply(rests, new_population)
    # print(new_population)

    best_outputs = []
    for generation in tqdm(range(st.session_state.num_gen)):
        # Measuring the fitness of each chromosome in the population.
        fitness = ga.cal_pop_fitness(new_population)
        # print("Fitness")
        # print(fitness)

        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(new_population, fitness, 
                                          st.session_state.num_parents_mating)
        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents,
                offspring_size=(pop_size[0]-parents.shape[0], num_weights))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        
    # Getting the best solution after iterating finishing all generations.
    # At first, the fitness is calculated for each solution in the final generation.
    fitness = ga.cal_pop_fitness(new_population)
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = np.where(fitness == np.max(fitness))

    print("Best solution : ", new_population[best_match_idx, :])
    print("Best solution fitness : ", fitness[best_match_idx])


    solution = new_population[best_match_idx, :][0][0]

    return solution


