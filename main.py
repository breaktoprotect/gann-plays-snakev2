
#debug
import gym
import numpy as np 
import gym_snake2
import time

from gann_agent import *

def main():
    gann_player = GANNAgent(fitness_top_percentile=0.005, mutation_rate=0.01,learning_rate=1e-2) # 1e-3*5 0.005

    watch_every = 1

    for i in range(500):
        gann_player.generate_training_data(generation_population=2000, debug_render=False, debug_frequency=1,verbose=False, num_of_epoch=20)
        
        if ((i+1) % watch_every) == 0:
            gann_player.play(num_of_games=5, model_generation_index=i+1)
    
    #time.sleep(2)
    #gann_player.play(random_game=True)

    #!program prematurely ends here
    return


# Test snake v2 env
'''
env = gym.make('snake-v1', render=True, segment_width=30)
env.reset()
FREQUENCY = 10

def main():

    done = False
    while(True):
        print(env._get_board())
        print("")
        env.render()
        
        _, _, done, _ = env.step(env.action_space.sample())
        time.sleep(1/FREQUENCY)

        if done:
            env.reset()
'''

if __name__ == "__main__":
    main()