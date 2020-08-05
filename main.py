
#debug
import gym
import numpy as np 
import gym_snake2
import time

from gann_agent import *

def main():
    gann_player = GANNAgent()

    gann_player.generate_training_data()

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