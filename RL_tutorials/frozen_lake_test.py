import numpy as np
import gym
import random
import time
import pathlib
import pickle

env = gym.make("FrozenLake-v0")
max_steps_per_episode = 100

cur_path = pathlib.Path(__file__).parent.absolute()
file_name = str(cur_path) + "/q_table_pickle"
pick_file = open(file_name, "rb")
q_table = pickle.load(pick_file)

for episode in range(3):
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):        
        # clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])        
        new_state, reward, done, info = env.step(action)

        if done:
            # clear_output(wait=True)
            env.render()
            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
                # clear_output(wait=True)
            break

        state = new_state

env.close()