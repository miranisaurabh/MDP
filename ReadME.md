## Funtions:

BFS: gets the shortest cost from all possible states to the given state. returns an array with cooresponding costs and a tree having child nodes for that shortest path.

get_shortest_path(): returns the optimal controls which passes throught the best pickup position

controls_to_sequence(): takes the optimal con trols and returns the optimal sequence. Note: Control is different from sequence here

get_pickup_positions(): returns all the possible pickup positions

get_best_pickup_position(): returns best pickup position that minimizes cost from start to key and key to door

Start_To_Goal_direct(): computes cost and policy for direct reach if possible

Start_To_Goal_viaDoor(): computes cost and policy for path via Door

## How to run the code?

Just change the variable env_name to the desired environment. It should be located in ./env folder and must have extension .env.