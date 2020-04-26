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

def get_shortest_path(shortest_path,shortest_path_controls,policy,start,start_ori,goal,best_pickup_position=None):

    if start == goal:

        return
    shortest_path.append((start))
    if best_pickup_position in policy[start]:

        x1,y1 = start
        x2,y2 = best_pickup_position
        next_ori = (x2-x1,y2-y1)
        if np.dot(next_ori,start_ori)==1:
            shortest_path_controls.append(0) #Move Forward
        elif np.dot(next_ori,start_ori)==-1:
            shortest_path_controls.append(5) #Move Backward
        elif np.cross(next_ori,start_ori)==1:
            shortest_path_controls.append(1) #Move Left
        else:
            shortest_path_controls.append(2) #Move Right
        get_shortest_path(shortest_path,shortest_path_controls,policy,(x2,y2),next_ori,goal)
    
    else:
        
        next_ = tuple( np.add(start,start_ori) ) 
        try:
            policy[start].index(next_)
            shortest_path_controls.append(0)
            get_shortest_path(shortest_path,shortest_path_controls,policy,next_,start_ori,goal,best_pickup_position)
        except ValueError:
            next_ = policy[start][0]
            x1,y1 = start
            x2,y2 = next_
            # next_ori = (x2-x1,y1-y2)
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
            get_shortest_path(shortest_path,shortest_path_controls,policy,(x2,y2),next_ori,goal,best_pickup_position)
    
    return

def visualize_costs(cost_matrix,flag=None):

    fig, ax = plt.subplots()
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            text = ax.text(j, i, cost_matrix[i, j],ha="center", va="center", color="w") 

    if flag==1:
        ax.set_title('Cost to key for: '+ env_name)
        im = ax.imshow(cost_matrix)
        # fig.savefig('./costs/' + env_name + '_costKey.png')

    elif flag==2:
        ax.set_title('Cost to door for: '+ env_name)
        im = ax.imshow(cost_matrix)
        # fig.savefig('./costs/' + env_name + '_costDoor.png')

    elif flag==3:
        ax.set_title('Cost to goal (open door) for: '+ env_name)
        im = ax.imshow(cost_matrix)
        # fig.savefig('./costs/' + env_name + '_costGoal_open.png')
    
    elif flag==4:
        ax.set_title('Cost to goal (closed door) for: '+ env_name)
        im = ax.imshow(cost_matrix)
        # fig.savefig('./costs/' + env_name + '_costGoal_closed.png')

def visualize_policy(policy,env_grid,pickup_positions=None,door=None,flag=None,goal_positions_open=None,goal_positions=None):

    x_max = env_grid.shape[0]
    y_max = env_grid.shape[1]

    fig, ax = plt.subplots()
    
    for ids in policy:

        X = ids[0]
        Y = ids[1]
        next_list = policy[ids]

        for alpha in range(len(next_list)):

            next_id = next_list[alpha]
            U = (next_id[0] - X)*0.5
            V = (next_id[1] - Y)*(-0.5)

            q = ax.quiver(X+0.5, Y+0.5, U, V,units='xy' ,scale=1)

    plt.grid()

    ax.set_aspect('equal')
    plt.xlim(0,x_max)
    plt.ylim(y_max,0)

    if pickup_positions is not None:

        for pos in range(len(pickup_positions)):

            pos_x,pos_y = pickup_positions[pos]
            ax.fill([pos_x,pos_x,pos_x+1,pos_x+1],[pos_y,pos_y+1,pos_y+1,pos_y],"b")
            ax.set_title('Policy for key: '+ env_name)
            # fig.savefig('./policy/' + env_name + '_policyKey.png')

    if door is not None:

        door_x,door_y = door
        ax.fill([door_x-1,door_x-1,door_x,door_x],[door_y,door_y+1,door_y+1,door_y],'tab:purple')
        ax.fill([door_x,door_x,door_x+1,door_x+1],[door_y,door_y+1,door_y+1,door_y],"y")
        ax.set_title('Policy for door: '+ env_name)
        # fig.savefig('./policy/' + env_name + '_policyDoor.png')

    if goal_positions_open is not None:

        for pos in range(len(goal_positions_open)):

            pos_x,pos_y = goal_positions_open[pos]
            ax.fill([pos_x,pos_x,pos_x+1,pos_x+1],[pos_y,pos_y+1,pos_y+1,pos_y],"g")
            ax.set_title('Policy for goal (door open): '+ env_name)
        # fig.savefig('./policy/' + env_name + '_policyGoal_open.png')
    
    if goal_positions is not None:

        for pos in range(len(goal_positions)):

            pos_x,pos_y = goal_positions[pos]
            ax.fill([pos_x,pos_x,pos_x+1,pos_x+1],[pos_y,pos_y+1,pos_y+1,pos_y],"g")
            ax.set_title('Policy for goal (door closed): '+ env_name)
        # fig.savefig('./policy/' + env_name + '_policyGoal_closed.png')

def visualize_value_function(start,pickup_positions,door,goal_positions,shortest_path_key=None,shortest_path_door=None,shortest_path_goal=None,shortest_path_direct=None):

    plt_pickup = {}
    plt_pickup[0] =[]
    plt_pickup[1]=[]
    plt_pickup[2]=[]
    plt_pickup[3]=[]
    label_pickup = ['Pickup position 1','Pickup position 2','Pickup position 3','Pickup position 4']

    plt_door = []
    plt_goal = {}
    plt_goal[0] =[]
    plt_goal[1]=[]
    plt_goal[2]=[]
    plt_goal[3]=[]
    env_grid_value = gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T
    label_goal = ['Goal Approach 1','Goal Approach 2','Goal Approach 3','Goal Approach 4']
    
    plt.figure()
    if shortest_path_key is not None:
        
        for i in range(len(shortest_path_key)):

            next_pos = shortest_path_key[i]
            policy_next = {}
            cost_from_next = np.full(env_grid_value.shape,np.inf)
            cost_from_next[next_pos[1],next_pos[0]] = 0
            cost_from_next,policy_next = BFS([next_pos].copy(),cost_from_next,env_grid_value,policy_next)
            plt_door.append(cost_from_next[door[1],door[0]-1])
            
            for j in range(len(pickup_positions)):

                posi = pickup_positions[j]
                plt_pickup[j].append(cost_from_next[posi[1],posi[0]])
            
            for k in range(len(goal_positions)):

                posi = goal_positions[k]
                plt_goal[k].append(cost_from_next[posi[1],posi[0]])

        for i in range(1,len(shortest_path_door)):

            next_pos = shortest_path_door[i]
            policy_next = {}
            cost_from_next = np.full(env_grid_value.shape,np.inf)
            cost_from_next[next_pos[1],next_pos[0]] = 0
            cost_from_next,policy_next = BFS([next_pos].copy(),cost_from_next,env_grid_value,policy_next)
            plt_door.append(cost_from_next[door[1],door[0]-1])
            
            for j in range(len(pickup_positions)):

                posi = pickup_positions[j]
                plt_pickup[j].append(cost_from_next[posi[1],posi[0]])
            
            for k in range(len(goal_positions)):

                posi = goal_positions[k]
                plt_goal[k].append(cost_from_next[posi[1],posi[0]])     
        
        env_grid_value[door[1],door[0]] = 1  

        for i in range(1,len(shortest_path_goal)):

            next_pos = shortest_path_goal[i]
            policy_next = {}
            cost_from_next = np.full(env_grid_value.shape,np.inf)
            cost_from_next[next_pos[1],next_pos[0]] = 0
            cost_from_next,policy_next = BFS([next_pos].copy(),cost_from_next,env_grid_value,policy_next)
            plt_door.append(cost_from_next[door[1],door[0]-1])
            
            for j in range(len(pickup_positions)):

                posi = pickup_positions[j]
                plt_pickup[j].append(cost_from_next[posi[1],posi[0]])
            
            for k in range(len(goal_positions)):

                posi = goal_positions[k]
                plt_goal[k].append(cost_from_next[posi[1],posi[0]]) 
        
    elif shortest_path_direct is not None:
    
        for i in range(len(shortest_path_direct)):

            next_pos = shortest_path_direct[i]
            policy_next = {}
            cost_from_next = np.full(env_grid_value.shape,np.inf)
            cost_from_next[next_pos[1],next_pos[0]] = 0
            cost_from_next,policy_next = BFS([next_pos].copy(),cost_from_next,env_grid_value,policy_next)
            plt_door.append(cost_from_next[door[1],door[0]-1])
            
            for j in range(len(pickup_positions)):

                posi = pickup_positions[j]
                plt_pickup[j].append(cost_from_next[posi[1],posi[0]])
            
            for k in range(len(goal_positions)):

                posi = goal_positions[k]
                plt_goal[k].append(cost_from_next[posi[1],posi[0]])
            


    plt.plot(plt_door,label='Unlock door position')
    for j in range(len(pickup_positions)):
        plt.plot(plt_pickup[j],':',label = label_pickup[j])
    for k in range(1,len(goal_positions)):
        plt.plot(plt_goal[k],'-.',label=label_goal[k])
    plt.plot(plt_goal[0],'o',label=label_goal[0])
    plt.legend()
    plt.title('Value function: ' + env_name)
    # plt.xlim([0,4])
    # plt.savefig('./value/' + env_name + '_valueFunction.png')
            





def controls_to_seq(shortest_path_controls,flag):

    seq = []
    for i in range(len(shortest_path_controls)):

        if shortest_path_controls[i] == 0:

            seq.append(0)
        
        elif shortest_path_controls[i] == 1:

            seq.append(1)
            seq.append(0)

        elif shortest_path_controls[i] == 2:

            seq.append(2)
            seq.append(0)

        elif shortest_path_controls[i] == 5:

            seq.append(1)
            seq.append(1)
            seq.append(0)
    if flag:
        seq[-1] = flag
    return seq

def get_pickup_positions(policy_key,key):

    pickup_positions = []
    for dict_key in policy_key:

        if key in policy_key[dict_key]:
            pickup_positions.append(dict_key)

    return pickup_positions

def get_best_pickup_position(cost_to_key,cost_to_door,pickup_positions):

    costs = []
    for num in range(len(pickup_positions)):

        cost1 = cost_to_key[pickup_positions[num][1],pickup_positions[num][0]]
        cost2 = cost_to_door[pickup_positions[num][1],pickup_positions[num][0]]
        total_cost = cost1+cost2
        costs.append(total_cost)

    costs_copy = costs.copy()
    costs_copy.sort()
    min_cost = costs_copy[0]
    if costs.count(min_cost)==1:

        return pickup_positions[costs.index(min_cost)]
    
    else:

        return None



def doorkey_problem(env):
    
    env_grid = gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T

    start = tuple(info['init_agent_pos'])
    start_ori = tuple(info['init_agent_dir'])
    key = tuple(info['key_pos'])
    door = tuple(info['door_pos'])
    goal = tuple(info['goal_pos'])

    cost_to_goal_direct,policy_direct =  Start_To_Goal_direct(env_grid,start,goal)

    cost_to_goal_direct_val = cost_to_goal_direct[start[1],start[0]]

    cost_viaDoor , seq_viaDoor,pickup_positions = Start_To_Goal_viaDoor(env_grid,start,start_ori,key,door,goal)
    plot_env(env)

    if cost_to_goal_direct_val > cost_viaDoor:
    
        seq_best = seq_viaDoor
    
    else:
        env_grid = gym_minigrid.minigrid.Grid.encode(env.grid)[:,:,0].T    
        shortest_path_direct = []
        shortest_path_controls_direct = []
        get_shortest_path(shortest_path_direct,shortest_path_controls_direct,policy_direct,start,start_ori,goal)
        seq_direct = controls_to_seq(shortest_path_controls_direct,0)
        seq_best = seq_direct

        goal_positions_closed = get_pickup_positions(policy_direct,goal)
        visualize_policy(policy_direct,env_grid,goal_positions=goal_positions_closed)
        visualize_costs(cost_to_goal_direct,4)
        visualize_value_function(start,pickup_positions,door,goal_positions_closed,shortest_path_direct=shortest_path_direct)

    draw_gif_from_seq(seq_best,env,path='./gif/'+ env_name + '.gif')

    

    optim_act_seq = seq_best

    return optim_act_seq

def Start_To_Goal_direct(env_grid,start,goal):

    policy_direct = {}
    cost_to_goal_direct = np.full(env_grid.shape,np.inf)
    cost_to_goal_direct[goal[1],goal[0]] = 0
    cost_to_goal_direct,policy_direct = BFS([goal].copy(),cost_to_goal_direct,env_grid,policy_direct)
    return cost_to_goal_direct,policy_direct

def Start_To_Goal_viaDoor(env_grid,start,start_ori,key,door,goal):

    door_status = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    is_locked = door_status.is_locked

    if is_locked:

        if env.carrying is None:

            # Get the policy to pickup key
            policy_key = {}
            cost_to_key = np.full(env_grid.shape,np.inf)
            cost_to_key[key[1],key[0]] = 0
            cost_to_key,policy_key = BFS([key].copy(),cost_to_key,env_grid,policy_key) 
            cost_key = cost_to_key[start[1],start[0]]-1
            visualize_costs(cost_to_key,1)

            # Get the possible pickup positions using the policy
            pickup_positions = get_pickup_positions(policy_key,key)
            print(pickup_positions)
            visualize_policy(policy_key,env_grid,pickup_positions)

            # Get the cost to reach any position from the start position
            ## Used to determine the cost to reach pickup location
            policy_start = {}
            cost_from_start = np.full(env_grid.shape,np.inf)
            cost_from_start[start[1],start[0]] = 0
            cost_from_start,policy_start = BFS([start].copy(),cost_from_start,env_grid,policy_start)
            print(cost_from_start)

            # Since the key will be picked up, it'll be free space
            env_grid[key[1],key[0]] = 1

        # Get the policy to unlock the door
        policy_door = {}
        cost_to_door = np.full(env_grid.shape,np.inf)
        cost_to_door[door[1],door[0]] = 0
        cost_to_door,policy_door = BFS([door].copy(),cost_to_door,env_grid,policy_door)
        #Visualize
        visualize_policy(policy_door,env_grid,door=door)
        visualize_costs(cost_to_door,2)

        if env.carrying is None:
            # Get the best pickup position
            best_pickup_position = get_best_pickup_position(cost_from_start,cost_to_door,pickup_positions)
            print(best_pickup_position)

            # Get the shortest path using the best pickup position
            shortest_path_key = []
            shortest_path_controls_key = []
            get_shortest_path(shortest_path_key,shortest_path_controls_key,policy_key,start,start_ori,key,best_pickup_position)
            robot_key_pos = shortest_path_key[-1]
            key_ori = (key[0]-robot_key_pos[0],key[1]-robot_key_pos[1])
            
            seq_key = controls_to_seq(shortest_path_controls_key,3)
        else:
            # The robot is carrying key, treat current position as pickup position
            robot_key_pos = start
            key_ori = start_ori

        # Get the shortest path from best pickup position to Goal
        shortest_path_door = []
        shortest_path_controls_door = []
        get_shortest_path(shortest_path_door,shortest_path_controls_door,policy_door,robot_key_pos,key_ori,door)
        robot_door_pos = shortest_path_door[-1]
        door_ori = (door[0]-robot_door_pos[0],robot_door_pos[1]-door[1])
        env_grid[door[1],door[0]] = 1
        seq_door = controls_to_seq(shortest_path_controls_door,4)   
        cost_door = cost_to_door[robot_key_pos[1],robot_key_pos[0]]-1

    # Get the policy to reach Goal
    policy_goal = {}
    cost_to_goal = np.full(env_grid.shape,np.inf)
    cost_to_goal[goal[1],goal[0]] = 0
    cost_to_goal,policy_goal = BFS([goal].copy(),cost_to_goal,env_grid,policy_goal)
    cost_goal = cost_to_goal[robot_door_pos[1],robot_door_pos[0]]
    goal_positions_open = get_pickup_positions(policy_goal,goal)
    visualize_policy(policy_goal,env_grid,goal_positions_open=goal_positions_open)
    visualize_costs(cost_to_goal,3)

    # Get the shortest path to reach Goal after unlocking the door
    shortest_path_goal = []
    shortest_path_controls_goal = []
    get_shortest_path(shortest_path_goal,shortest_path_controls_goal,policy_goal,robot_door_pos,door_ori,goal)
    seq_goal = controls_to_seq(shortest_path_controls_goal,0)

    seq = seq_key + seq_door + seq_goal


    visualize_value_function(start,pickup_positions,door,goal_positions_open,shortest_path_key=shortest_path_key,shortest_path_door=shortest_path_door,shortest_path_goal=shortest_path_goal)

    return cost_key+cost_door+cost_goal,seq,pickup_positions

if __name__ == '__main__':

    env_name = 'doorkey-6x6-shortcut'
    env_path = './envs/'+ env_name +'.env'
    env, info = load_env(env_path) # load an environment
    doorkey_problem(env)
    
    
    
    

    

        
        
    
