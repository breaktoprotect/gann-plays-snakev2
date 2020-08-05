# Author        : JS @breaktoprotect
# Date started  : 4 August 2020
# Description   : Attempting to implementic Genetic Algorithm with Neural Network to play Snake game 50 by 50 space
import gym
import numpy as np 
import gym_snake2
import time
import random

class GANNAgent:
    def __init__(self, fitness_top_percentile=0.05):
        # Environment
        self.env = gym.make('snake-v1', render=True)
        self.env.reset()

        # Genetic Algorithm
        self.current_generation_index = 0
        self.generation_fitness = [] # list of average fitness score per that generation
        self.fitness_top_percentile = fitness_top_percentile


    def generate_training_data(self, generation_population = 1000, acceptance_criteria = 0.05, model_path=False):
        training_data = []
        score_list = []
        accepted_score_list = []
        game_memory_lol = [] # game_memory list of lists

        print("[*] Generating training data...started.")

        if model_path:
            pass

        for episode in range(generation_population):
            # Reset parameters for each episode
            self.env.reset()
            cur_game_memory_list = []
            cur_score = 0
            prev_observation = []
            cur_choices_list = []

            #* Run the game until it is done (using break)
            while True:
                if model_path and prev_observation:
                    pass
                    #TODO use current brain/model to predict action
                else:
                    action = self.env.action_space.sample()

                # Execute a step
                observation, reward, done, info = self.env.step(action)

                # Record game memory
                if prev_observation:
                    cur_game_memory.append([prev_observation, action])
                    choices.append(action)

                prev_observation = observation
                cur_score += reward

                # Terminate when game has ended
                if done:
                    break

            # Each end of game calculations
            score_list.append(cur_score)
            game_memory_lol.append(cur_game_memory_list)

            # For progression notification
            if (episode+1) % 500 == 0:
                print("[*]", episode+1, "has completed.")
        
        #* Pick top scorers and select for next generation
        print("[*] Calculating fitness of Generation {GENERATION_INDEX}...".format(GENERATION_INDEX=self.current_generation_index))
        minimum_score_requirement = self.fitness(score_list)

        for index, score in enumerate(score_list):
            if score >= minimum_score_requirement:
                # Add to list of accepted scores
                accepted_score_list.append(score)

                # Add observations and output to training data
                for data in game_memory_lol[index]:
                    if data[1] == 0: # data[1] is action; where data[0] is prev_observation
                        output = [1, 0, 0, 0] # Up
                    elif data[1] == 1:
                        output = [0, 1, 0, 0] # Right
                    elif data[1] == 2:
                        output = [0, 0, 1, 0] # Down
                    elif data[1] == 3:
                        output = [0, 0, 0, 1] # Left

                    # Training data instance is 1) Previous observaiton; 2) Action carried out
                    training_data.append(data[0], output)

        self.current_generation_index += 1
        self.generation_fitness[self.current_generation_index] = self.calculate_fitness() #TODO

    def get_current_generation_index(self):
        return self.current_generation_index

    def set_current_generation_index(self, generation_number):
        #TODO truncate/reset old generation

        if generation_number >= self.current_generation_index:
            print("[-] Generation number must be smaller than current generation.")
            return False

        self.current_generation_index = generation_number
        return True
        
    def get_winners(self, score):
        pass

    # Fitness criteria defined as top 5% percentile
    def fitness(self, score_list):
        index_of_min_score = round(len(score_list)*self.fitness_top_percentile)

        # High score to lowest score, starting index 0 to last 
        score_list.sort(reverse=True)

        min_score = score_list[index_of_min_score-1]

        return min_score

    def crossover(self):
        pass

    def mutate(self):
        pass


    # Use current model (brain) to predict the next move
    def get_action(self,current_state):
        pass
    
    # Just return a random action without thinking
    def get_random_action(self):
        return random.randint(0,3)
        
        