import numpy as np
import gym
from utils import *
# from example import example_use_of_gym_env

MF = 0 # Move Forward
TL = 1 # Turn Left
TR = 2 # Turn Right
PK = 3 # Pickup Key
UD = 4 # Unlock Door

def BFS(queue,cost_to_goal,env_grid,policy):

    new_queue = []
    cost = 1
    y_max = cost_to_goal.shape[0]-1
    x_max = cost_to_goal.shape[1]-1
    print('Here')
    if queue == []:
        return cost_to_goal,policy

    for i in range(len(queue)):

        current_idx,current_idy = queue.pop(0)

        # Check all four directions

        if current_idx+1 >=0 and current_idx+1 <= x_max and env_grid[current_idy,current_idx+1]==1:

            current_cost = cost_to_goal[current_idy,current_idx]
            next_cost = cost_to_goal[current_idy,current_idx+1]
            if current_cost + cost < next_cost:
                cost_to_goal[current_idy,current_idx+1] = current_cost + cost
                new_queue.append([current_idx+1,current_idy])
                policy[(current_idx+1,current_idy)] = [(current_idx,current_idy)]
            elif current_cost + cost == next_cost:
                policy[(current_idx+1,current_idy)].append((current_idx,current_idy))



        if current_idx-1 >=0 and current_idx-1 <= x_max and env_grid[current_idy,current_idx-1]==1:

            current_cost = cost_to_goal[current_idy,current_idx]
            next_cost = cost_to_goal[current_idy,current_idx-1]
            if current_cost + cost < next_cost:
                cost_to_goal[current_idy,current_idx-1] = current_cost + cost   
                new_queue.append([current_idx-1,current_idy])
                policy[(current_idx-1,current_idy)] = [(current_idx,current_idy)]
            elif current_cost + cost == next_cost:
                policy[(current_idx-1,current_idy)].append((current_idx,current_idy))     

        if current_idy+1 >=0 and current_idy+1 <= y_max and env_grid[current_idy+1,current_idx]==1:

            current_cost = cost_to_goal[current_idy,current_idx]
            next_cost = cost_to_goal[current_idy+1,current_idx]
            if current_cost + cost < next_cost:
                cost_to_goal[current_idy+1,current_idx] = current_cost + cost
                new_queue.append([current_idx,current_idy+1])
                policy[(current_idx,current_idy+1)] = [(current_idx,current_idy)]
            elif current_cost + cost == next_cost:
                policy[(current_idx,current_idy+1)].append((current_idx,current_idy))

        if current_idy-1 >=0 and current_idy-1 <= y_max and env_grid[current_idy-1,current_idx]==1:

            current_cost = cost_to_goal[current_idy,current_idx]
            next_cost = cost_to_goal[current_idy-1,current_idx]
            if current_cost + cost < next_cost:
                cost_to_goal[current_idy-1,current_idx] = current_cost + cost
                new_queue.append([current_idx,current_idy-1])
                policy[(current_idx,current_idy-1)] = [(current_idx,current_idy)]
            elif current_cost + cost == next_cost:
                policy[(current_idx,current_idy-1)].append((current_idx,current_idy))

    return BFS(new_queue,cost_to_goal,env_grid,policy)

def get_shortest_path(shortest_path,shortest_path_controls,policy,start,start_ori,goal):

    print('Ala re')
    if start == goal:

        return

    # if len(policy[start]) == 1:
    #     x1,y1 = start
    #     x2,y2 = policy[start]
    #     next_ori = (x2-x1,y2-y1)
    #     change_ori = tuple(np.subtract(next_ori,start_ori))
    #     shortest_path_controls[start] = [(x2,y2),change_ori]
    #     get_shortest_path(shortest_path_controls,policy,(x2,y2),next_ori,goal)
    
    # else:
    shortest_path.append((start))
    next_ = tuple( np.add(start,start_ori) ) 
    try:
        policy[start].index(next_)
        shortest_path_controls.append(0)
        get_shortest_path(shortest_path,shortest_path_controls,policy,next_,start_ori,goal)
    except ValueError:
        next_ = policy[start][0]
        x1,y1 = start
        x2,y2 = next_
        next_ori = (x2-x1,y2-y1)
        # change_ori = tuple(np.subtract(next_ori,start_ori))
        if np.dot(next_ori,start_ori)==1:
            shortest_path_controls.append(0) #Move Forward
        elif np.dot(next_ori,start_ori)==-1:
            shortest_path_controls.append(5) #Move Backward
        elif np.cross(next_ori,start_ori)==1:
            shortest_path_controls.append(1) #Move Left
        else:
            shortest_path_controls.append(2) #Move Right
        get_shortest_path(shortest_path,shortest_path_controls,policy,(x2,y2),next_ori,goal)
    
    return

def move_robot(shortest_path_controls,env):

    for i in range(len(shortest_path_controls)):

        if shortest_path_controls[i] == 0:

            cost, done = step(env, MF)
            plot_env(env)
        
        elif shortest_path_controls[i] == 1:

            cost, done = step(env, TL)
            cost, done = step(env, MF)
            plot_env(env)         
        
        elif shortest_path_controls[i] == 2:
            
            cost, done = step(env, TR)
            cost, done = step(env, MF)
            plot_env(env)

        elif shortest_path_controls[i] == 5:

            cost, done = step(env, TL)
            cost, done = step(env, TL)
            cost, done = step(env, MF)
            plot_env(env)

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

def Start_To_Goal_direct(env_grid,start,goal):

    policy_direct = {}
    cost_to_goal_direct = np.full(env_grid.shape,np.inf)
    cost_to_goal_direct[goal[1],goal[0]] = 0
    cost_to_goal_direct,policy_direct = BFS([goal].copy(),cost_to_goal_direct,env_grid,policy_direct)
    return cost_to_goal_direct[start[1],start[0]],policy_direct

def Start_To_Goal_viaDoor(env,start,start_ori,key,door,goal):

    policy_key = {}
    cost_to_key = np.full(env.shape,np.inf)
    cost_to_key[key[1],key[0]] = 0
    cost_to_key,policy_key = BFS([key].copy,cost_to_key,env_grid,policy_key) 

    policy_door = {}
    cost_to_door = np.full(env.shape,np.inf)
    cost_to_door[door[1],door[0]] = 0
    cost_to_door,policy_door = BFS([door].copy,cost_to_door,env_grid,policy_door) 

    policy_goal = {}
    cost_to_goal = np.full(env.shape,np.inf)
    cost_to_goal[goal[1],goal[0]] = 0
    cost_to_goal,policy_goal = BFS([goal].copy(),cost_to_goal,env_grid,policy_goal) 

    return cost_to_key+cost_to_door+cost_to_goal,policy_key,policy_door,policy_goal



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
    goal = [(2,6)]
    cost_to_goal = np.full((8,8),np.inf)
    cost_to_goal[6,2] = 0
    policy = {}
    print(goal)
    # cost_to_goal,policy = BFS(goal.copy(),cost_to_goal,env_grid,policy)
    print(goal)
    # cost_to_goal = np.full(env_grid.shape,np.inf)
    # policy = {}
    # cost_to_goal,policy = BFS(goal,cost_to_goal,env_grid,policy)
    # print(cost_to_goal)
    # print(policy)

    # print('------------------------')
    # print(goal)
    # shortest_path_controls=[]
    # shortest_path = []
    start = (2,4)
    # start_ori = (-1,0)
    # print(get_shortest_path(shortest_path,shortest_path_controls,policy,start,start_ori,goal[0]))
    # print(shortest_path)
    # plot_env(env)
    # #move_robot(shortest_path_controls,env)
    goal = (6,6)
    print([goal].copy())
    cost_to_goal_direct,policy_direct =  Start_To_Goal_direct(env_grid,start,goal)
    print(cost_to_goal_direct)
    print(policy_direct)


    

        
        
    
