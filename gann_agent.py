# Author        : JS @breaktoprotect
# Date started  : 4 August 2020
# Description   : Attempting to implementic Genetic Algorithm with Neural Network to play Snake game 50 by 50 space
import gym
import numpy as np 
import gym_snake2
import time
import random
import tflearn

# Standard Deep Neural Network 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression

# Convo Neural Network
from tflearn.layers.conv import conv_2d, max_pool_2d

import tensorflow as tf

class GANNAgent:
    def __init__(self, fitness_top_percentile=0.05, segment_width=25, mutation_rate=0.02, learning_rate=1e-3):
        # Environment
        self.env = gym.make('snake2-v0', render=True, segment_width=25)
        self.env.reset()

        # Genetic Algorithm
        self.model_list = [None, None] # Model or the "brain" of the agent to make decisions; Only Two are loaded at the same time - Parents
        self.current_snake_model = None
        self.current_generation_index = 0 # Generation starts from 1. Only gets += after each generate_training_data()
        self.generation_fitness = [] # list of average fitness score per that generation
        self.generation_scores = [] # top 5 of sorted from highest to lowest snakes
        self.fitness_top_percentile = fitness_top_percentile
        self.learning_rate = learning_rate # or 0.01 (or 1e-3 0.001)
        self.mutation_rate = mutation_rate


    def generate_training_data(self, generation_population = 1000, acceptance_criteria = 0.05, verbose=False, debug_render=False, debug_frequency=100, num_of_epoch=5):
        training_data = []
        score_list = []
        accepted_score_list = []
        game_memory_lol = [] # game_memory list of lists

        if self.model_list[0] and self.model_list[0]:
            parent_model_1 = self.model_list[0] # Load the two parents model
            parent_model_2 = self.model_list[1]
            is_crossover = True
            print("[+] Models loaded with Two Parents from Generation: {GEN}".format(GEN=self.current_generation_index))
        else:
            is_crossover = False

        print("[*] Generating training data...started.")

        #* Start of training each episode for defined number of population in the generation
        for episode in range(generation_population):
            #* Verbose
            if verbose and episode > 0:
                print("[*] Completion summary - {EPS} out of {TOTAL} - {COMPLETION}%".format(COMPLETION=episode/generation_population*100, EPS=episode, TOTAL=generation_population))

                #debug 
                '''
                print("len(game_memory_lol) -->", len(game_memory_lol))
                print("len(score_list) -->", len(score_list))
                print("len(training_data) -->", len(training_data))
                print("")
                '''

            #* Reset parameters for each episode
            self.env.reset()
            cur_game_memory_list = []
            cur_score = 0
            prev_observation = []
            cur_choices_list = []

            #* Decide if this snake agent is "mutated"
            if random.random() < self.mutation_rate:
                is_mutated = True
            else:
                is_mutated = False

            #* Run the game until it is done (using break)
            while True:
                if debug_render:
                    self.env.render()
                    time.sleep(1/debug_frequency)

                if is_crossover and len(prev_observation) and is_mutated == False > 0:
                    #* Use current brain/model to predict action
                    self._get_parents_action(parent_model_1, parent_model_2, prev_observation)
                else:
                    # Random
                    action = self.env.action_space.sample()

                # Execute a step
                observation, reward, done, info = self.env.step(action)

                if verbose:
                    print(observation)
                    print("***action taken->", action)
                    print("")
                    
                
                #debug
                #print(observation)
                #print("")
                #time.sleep(1)
                #print("len(self._convert_board_to_inputs(observation)) -->", len(self._convert_board_to_inputs(observation)))
                #/debug

                # Record game memory
                if len(prev_observation) > 0:
                    cur_game_memory_list.append([prev_observation, action])
                    cur_choices_list.append(action)

                prev_observation = observation # normalized to a sequential 400 inputs (20 x 20)
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
        
        #? After each episode
        #* Pick top scorers and select for next generation
        print("[*] Calculating fitness of Generation {GENERATION_INDEX}...".format(GENERATION_INDEX=self.current_generation_index))
        minimum_score_requirement = self.fitness_min_score(score_list)

        #! Deprecated
        '''
        #debug
        print("debug: minimum_score_requirement ->",minimum_score_requirement)

        for index, score in enumerate(score_list):
            if score >= minimum_score_requirement:
                # Debug
                print("[+] Generation {GEN} Snake {INDEX} has been selected for score:{SCORE}.".format(GEN=self.current_generation_index,INDEX=index,SCORE=score))

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
                    training_data.append([data[0], output])
        '''
        # Take top 2 as parents
        tmp_score_list = score_list.copy()
        tmp_score_list.sort(reverse=True) # Highest to lowest
        index_of_score1 = score_list.index(tmp_score_list[0])
        index_of_score2 = score_list.index(tmp_score_list[1])
        indexes_of_score = [index_of_score1, index_of_score2]

        # Add to accepted_score_list for record
        accepted_score_list.append(tmp_score_list[0])
        accepted_score_list.append(tmp_score_list[1])

        #debug
        print("debug: index_of_score1:", index_of_score1)
        print("debug: index_of_score2:", index_of_score2)
        

        # Add observations and output to training data
        for i, list_index in enumerate(indexes_of_score):
            training_data = []
            for data in game_memory_lol[list_index]:
                if data[1] == 0: # data[1] is action; where data[0] is prev_observation
                    output = [1, 0, 0, 0] # Up
                elif data[1] == 1:
                    output = [0, 1, 0, 0] # Right
                elif data[1] == 2:
                    output = [0, 0, 1, 0] # Down
                elif data[1] == 3:
                    output = [0, 0, 0, 1] # Left

                # Training data instance is 1) Previous observaiton; 2) Action carried out
                training_data.append([data[0], output])

            # Perform neural network fit (training the agent's brain)
            self.model_list[i] = self._train_fitted_model(training_data, num_of_epoch=num_of_epoch)

        #debug
        print("debug** self.model_list length:", len(self.model_list))

        # Current Snake Model Takes after the Stronger Parent
        self.current_snake_model = self.model_list[0]

        # Perform neural network fitting (or training the agent's "brain")
        #self.model_list.append(self._train_fitted_model(training_data, num_of_epoch=num_of_epoch))

        # Update generation
        self._update_generation_information(accepted_score_list)

        # Select new parents to create future generation
        #self.parent_index_1, self.parent_index_2 = self._select_parents(minimum_generation=10)
        
        # Summary of each generation
        print("[*] Summary of Generation {GEN}".format(GEN=self.current_generation_index))
        accepted_score_list.sort(reverse=True)
        print("    Top scores(*):", accepted_score_list[:5])
        print("    Average scores(~):", self.generation_fitness)
        accepted_score_list.sort()
        print("    Minimum scores(-):", accepted_score_list[:5])

        return training_data

    # Update generation information
    def _update_generation_information(self, accepted_score_list):
        self.current_generation_index += 1
        self.generation_scores.append(accepted_score_list)
        self.generation_fitness.append(self.calculate_topscorers_fitness(accepted_score_list)) 
        return

    #* Train the model / brain
    def _train_fitted_model(self, training_data, num_of_epoch, model=False): 
        #X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1) # Observations
        X = np.array([i[0] for i in training_data]).reshape([-1, 20, 20, 1]) #!experimental
        y = [i[1] for i in training_data]

        # Create a new graph / session
        new_graph = tf.Graph()
        with new_graph.as_default(): #? working :)
            if not model:
                model = self._create_neural_network_model(input_size = len(X[0]), output_size= len(y[0]))

                #debug
                print("len(X) -->",len(X[0]))
                print("len(y) -->",len(y[0]))

            model.fit({'input': X}, {'targets': y}, n_epoch=num_of_epoch,snapshot_step=1000, show_metric=True, run_id='gann_agent')

        return model

    # Defining the neural network
    def _create_neural_network_model(self, input_size, output_size):
        #network = input_data(shape=[None, input_size, 1], name='input')
        network = input_data(shape=[None, 20, 20, 1], name='input') #!experimental hardcoded

        #!experimental
        network = conv_2d(network, 320, 2, activation="relu")
        network = max_pool_2d(network, 2)

        network = conv_2d(network, 160, 2, activation="relu")
        network = max_pool_2d(network, 2)
        #/experiemntal

        network = fully_connected(network, 80, activation="relu") # activation function = rectified linear
        network = dropout(network,0.8) # 80% retention

        network = fully_connected(network, output_size, activation='softmax') # 4 outputs or actions of agent
        network = regression(network, optimizer='adam', learning_rate=self.learning_rate, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(network, tensorboard_dir='log')

        return model

    #! Experimental: Convolution Neural Network
    '''
    def _conv_neural_network_model(self, input_size, output_size):
        convnet = input_data(shape=[None, 400,400])
    '''
    # Convert the board observation matrix into inputs
    def _convert_board_to_inputs(self, board):
        inputs = []
        for cell in board:
            inputs.extend(cell)

        return inputs # in tuple

    def get_current_generation_index(self):
        return self.current_generation_index

    def set_current_generation_index(self, generation_number):
        #TODO truncate/reset old generation

        if generation_number >= self.current_generation_index:
            print("[-] Generation number must be smaller than current generation.")
            return False

        self.current_generation_index = generation_number
        return True
    
    # Average fitness of current selected generation 
    def calculate_topscorers_fitness(self, accepted_score_list):
        average_topscorers_fitness = sum(accepted_score_list) / len(accepted_score_list)
        return average_topscorers_fitness

    # Fitness criteria defined as top percentile
    def fitness_min_score(self, score_list):
        index_of_min_score = round(len(score_list)*self.fitness_top_percentile)

        # High score to lowest score, starting index 0 to last 
        score_list.sort(reverse=True)

        min_score = score_list[index_of_min_score]

        #debug
        #print("score_list:",score_list)
        #print("index_of_min_score:",index_of_min_score)
        #print("sorted score_list:",score_list)

        return min_score

    '''
    def _select_parents(self, minimum_generation):
        # Ensure that first enough initial generations to pick from 
        if len(self.generation_fitness) < minimum_generation:
            print("[*] Skipping parents selection - currently insufficient minimum generation. Expecting at least {MIN}. Currently only {CUR}.".format(MIN=minimum_generation, CUR=len(self.generation_fitness)))
            return None, None

        gf_list = self.generation_fitness.copy()

        # Sort to obtain highest to lowest
        gf_list.sort(reverse=True)

        # Select Top two from sorted list and find out the generation index from original list
        gen_index1 = self.generation_fitness.index(gf_list[0])
        gen_index2 = self.generation_fitness.index(gf_list[1])

        # Debug
        print("[+] Selected two generation index-> Gen Index 1:{GEN1}, Gen Index 2:{GEN2}".format(GEN1=gen_index1, GEN2=gen_index2))

        return gen_index1, gen_index2
    '''

    def most_fit_generation_index(self):
        if not len(self.generation_fitness) > 0:
            print("[!] Cannot find most fit generation. Currently generation is less than 1.")
            return

        gf_list = self.generation_fitness.copy()

        gf_list.sort(reverse=True)

        return self.generation_fitness.index(gf_list[0])


    def crossover(self):
        pass

    # Use a single model to predict the next move
    def _get_action(self, model, prev_observation):
        #action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])
        action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, 20, 20, 1))[0]) #!experimental

        return action
    
    # Use current two parent models (brains) to predict the next move
    def _get_parents_action(self, parent_model_1, parent_model_2, prev_observation):
        # 50% chance of either parent 1 or parent 2
        if random.random() > 0.5: 
            #action = np.argmax(parent_model_1.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])
            action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, 20, 20, 1))[0]) #!experimental
        else:
            #action = np.argmax(parent_model_2.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])
            action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, 20, 20, 1))[0]) #!experimental
        return action
        
    #* Play the selected model based on generation index
    def play(self, model_generation_index=None, num_of_games=5, frequency=50, random_game=False):
        # If model not defined, choose the best fitness generation
        '''
        if not model_generation_index:
            cur_model_index = self.most_fit_generation_index()
            cur_model = self.model_list[cur_model_index]
            print("[*] No model generation index defined, selected most fit generation index: {GEN}.".format(GEN=cur_model_index))
        else:
            cur_model = self.model_list[model_generation_index-1]
        '''
        if not self.current_snake_model:
            print("[!] Error. No existing trained snake model.")
            return
        cur_model = self.current_snake_model

        # Init
        cur_score_list = []

        for episode in range(num_of_games):
            #* Reset parameters for each episode
            self.env.reset()
            #cur_game_memory_list = []
            cur_score = 0
            prev_observation = []
            #cur_choices_list = []

            #* In each game
            while True:
                time.sleep(1/frequency)
                self.env.render()

                if len(prev_observation)  > 0 and not random_game:
                    #* Use current brain/model to predict action
                    action = self._get_action(cur_model, prev_observation)
                else:
                    # Random move if not yet have any observation
                    action = self.env.action_space.sample()

                # Execute a step
                observation, reward, done, info = self.env.step(action)

                prev_observation = observation # normalized to a sequential 400 inputs (20 x 20)
                cur_score += reward

                if done:
                    time.sleep(1) # Pause for a second
                    break

            # End of current game
            cur_score_list.append(cur_score)
        
        print("[*] Summary of scores:",cur_score_list)