import numpy as np
import gym
from utils import *
# from example import example_use_of_gym_env

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def BFS(queue,cost_to_goal,env_grid):

    new_queue = []
    cost = 1
    y_max = cost_to_goal.shape[0]-1
    x_max = cost_to_goal.shape[1]-1
    print('Here')
    if queue == []:
        return cost_to_goal

    for i in range(len(queue)):

        current_idx,current_idy = queue.pop(0)

        # Check all four directions

        if current_idx+1 >=0 and current_idx+1 <= x_max and env_grid[current_idy,current_idx+1]==1:

            current_cost = cost_to_goal[current_idy,current_idx]
            next_cost = cost_to_goal[current_idy,current_idx+1]
            if current_cost + cost < next_cost:
                cost_to_goal[current_idy,current_idx+1] = current_cost + cost
                new_queue.append([current_idx+1,current_idy])


        if current_idx-1 >=0 and current_idx-1 <= x_max and env_grid[current_idy,current_idx-1]==1:

            current_cost = cost_to_goal[current_idy,current_idx]
            next_cost = cost_to_goal[current_idy,current_idx-1]
            if current_cost + cost < next_cost:
                cost_to_goal[current_idy,current_idx-1] = current_cost + cost   
                new_queue.append([current_idx-1,current_idy])     

        if current_idy+1 >=0 and current_idy+1 <= y_max and env_grid[current_idy+1,current_idx]==1:

            current_cost = cost_to_goal[current_idy,current_idx]
            next_cost = cost_to_goal[current_idy+1,current_idx]
            if current_cost + cost < next_cost:
                cost_to_goal[current_idy+1,current_idx] = current_cost + cost
                new_queue.append([current_idx,current_idy+1])

        if current_idy-1 >=0 and current_idy-1 <= y_max and env_grid[current_idy-1,current_idx]==1:

            current_cost = cost_to_goal[current_idy,current_idx]
            next_cost = cost_to_goal[current_idy-1,current_idx]
            if current_cost + cost < next_cost:
                cost_to_goal[current_idy-1,current_idx] = current_cost + cost
                new_queue.append([current_idx,current_idy-1])

    return BFS(new_queue,cost_to_goal,env_grid)

def get_policy(start,goal,cost_to_goal):

    y_max = cost_to_goal.shape[0]-1
    x_max = cost_to_goal.shape[1]-1

    for i in range(y_max):

        for j in range(x_max):

            cos

def get_shortest_path(env_grid,start,goal):

    policy
    cost_to_goal = np.full(env_grid.shape,np.inf)


def doorkey_problem(env):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env
        
        doorkey-6x6-direct.env
        doorkey-8x8-direct.env
        
        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
        
    Feel Free to modify this fuction
    '''
    optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq


def main():
    env_path = './envs/example-8x8.env'
    env, info = load_env(env_path) # load an environment
    seq = doorkey_problem(env) # find the optimal action sequence
    print('Now printing')
    #print(env.grid.decode)
    plot_env(env)
    print(gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0])
    print('-------------')
    obs, reward, done, info = env.step(env.actions.forward)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    print('Done Printing')
    plot_env(env)
    print('-------------')
    obs, reward, done, info = env.step(env.actions.left)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    plot_env(env)
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save



if __name__ == '__main__':
    # example_use_of_gym_env()
    # main()
    env_path = './envs/example-8x8.env'
    env, info = load_env(env_path) # load an environment
    print(info)
    env_grid = gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T
    print(env_grid)
    start = [(2,6)]
    cost_to_goal = np.full((8,8),np.inf)
    cost_to_goal[6,2] = 0
    print(BFS(start,cost_to_goal,env_grid))

        
        
    
