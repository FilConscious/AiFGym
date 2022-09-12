import os
import cv2
import gym
import time
import random
import numpy as np  

from pathlib import Path
from collections.abc import Sequence, Iterable
from datetime import datetime
from gym import Env, spaces


# Retrieving the path to THIS file
PATH_TO_FILE = Path(__file__)
# The granparent directory of THIS file, i.e. 
# in Windows: C:\Users\..\..\custom_gym
# in Linux: home/../../custom_gym
GRANPA_DIR = PATH_TO_FILE.parents[1]
# Now we can access the images folder in custom_gym
IMAGES_DIR = GRANPA_DIR.joinpath("images")


### DEBUGGING ###

# p = Path(__file__)
# print(f'The path p is: {p}')
# print(f'The abs path p is: {p.absolute()}')
# print(f'The abs parent of path p is: {p.parent.absolute()}')
# print(f'The parent of path p is: {p.parent}')
# print(f'The granparent of path p is: {p.parents[1]}')
# print(f'The absolute granparent of path p is: {p.parents[1].absolute()}')
# print(f'The cwd is: {Path().absolute()}')

### END OF DEBUGGING ###


class GridWorldV0(Env):

    # Length of the side of a square tile in a grid-world
    l = 125
    
    def __init__(self, params):
        super(GridWorldV0, self).__init__()
        
        # Define height and width of the grid
        self.grid_height = params['height']
        self.grid_width = params['width'] 
        # Whether env structure will be random or not
        self.rand_conf = params['rand_conf']
        # List of wall tiles coordinates grouped as two-value tuples
        self.walls = params['walls']

        # Computing the number of tiles/states in the grid world
        self.num_states = int(self.grid_height * self.grid_width)

        # Define a discrete observation space using gym.spaces                  
        self.observation_space = spaces.Discrete(self.num_states,)

        # Define a RGB representation of the grid world (a canvas) to render the environment images
        # N.B. a tensor of ones is rendered as a white canvas by pyplot.imshow()
        self.rgb_image_shape = (self.grid_height * self.l, self.grid_width * self.l, 3)                    
        self.canvas = np.ones(self.rgb_image_shape)

        # Max and min coordinates for the RGB image representation
        self.y_min = 0
        self.x_min = 0
        self.y_max = self.rgb_image_shape[0]
        self.x_max = self.rgb_image_shape[1]

        # Define a coarse-grained representation of the grid world
        # N.B. The states in self.observation_space are reshaped into a two dimensional array
        self.coarse_representation = np.arange(self.num_states).reshape((self.grid_height, self.grid_width)) 

        # Define an action space ranging from 0 to 3 (four actions in total)
        self.action_space = spaces.Discrete(4,)

        # Goal state for the agent
        self.goal_state = params['goal_state'] 
        # Whether the agent's position will be random or not
        self.rand_pos = params['rand_pos']
        # Starting position for the agent
        self.start_agent_pos = params['start_agent_pos']

        # List of canvas frames for recording a video of the environment
        self.canvas_frames = []
        # Define elements present inside the grid world (e.g. wall tiles)
        self.elements = {}


    # def set_rgb_image_shape(self, x_max=625, y_max=625):
    #     '''
    #     Function to set the size of the canvas on which to render the environment.

    #     '''

    #     self.rgb_image_x_max = x_max
    #     self.rgb_image_y_max = y_max

    #     return (self.rgb_image_y_max, self.rgb_image_x_max, 3)


    def coarse_to_rgb(self, m: int, n: int) -> Sequence[int]:
        ''' 
        Function that maps from coarse-grained/abstract states, i.e. the states in self.observation_space,
        to their RGB counterparts.

        The conversion can be understood via the Kronecker product between two matrices A and B, where

        - A belongs to R^(m x n), with n = self.grid_height and m = self.grid_width (i.e. A is self.coarse_representation)
        - B belongs to R^(p x q) with p = self.rgb_image_y_max // self.grid_height and q = self.x_max // self.grid_width

        If the Kronecker product is indicated by A x B, we have that A x B belongs to R^{pm x qn} and that 
        (A x B)_{i, j} = a_{i // p, j // q} b_{i % p, j % q}. 

        N.B. For what we are interested in, the indices (m, n) are mapped to the indices of a top-left vertice of a rectangular block 
        in A x B which can be obtained by adding multiplying p (q) to m + 1 (n + 1).
        
        '''

        assert n >= 0 and n <= self.coarse_representation.shape[1], print('Coarse grained x coordinate not within valid bounds.')
        assert m >= 0 and m <= self.coarse_representation.shape[0], print('Coarse grained y coordinate not within valid bounds.')

        p = self.y_max // self.grid_height
        q = self.x_max // self.grid_width

        i = m * p
        j = n * q

        return i, j


    def rgb_to_coarse(self, i: int, j: int) -> int:
        ''' 
        Function that maps from RGB values to coarse-grained/abstract states, i.e. the states in self.observation_space.

        The conversion can be understood via the Kronecker product between two matrices A and B, where

        - A belongs to R^(m x n), with n = self.grid_height and m = self.grid_width (i.e. A is self.coarse_representation)
        - B belongs to R^(p x q) with p = self.rgb_image_y_max // self.grid_height and q = self.x_max // self.grid_width

        If the Kronecker product is indicated by A x B, we have that A x B belongs to R^{pm x qn} and that 
        (A x B)_{i, j} = a_{i // p, j // q} b_{i % p, j % q}. This formula shows you exactly how to compute m and n, 
        
        '''

        assert j >= self.x_min and j <= self.x_max, print('RGB x coordinate not within valid bounds.')
        assert i >= self.y_min and i <= self.y_max, print('RGB y coordinate not within valid bounds.')

        p = self.y_max // self.grid_height
        q = self.x_max // self.grid_width

        m = i // p
        n = j // q

        return m, n


    def is_colliding(self, position: tuple[float], *arg) -> bool:
        '''
        Function to check if the position of one element is colliding with that of another or with those of several elements.

        '''

        #assert isinstance(a, Element), print('First argument is not an Element object.')

        x, y = position
        coarse_state = self.coarse_representation[self.rgb_to_coarse(y, x)]

        for e in arg:

            assert isinstance(e, Element), print(f'Item {e} in *arg is not an Element object.')

            x_e, y_e = e.get_position()
            coarse_state_e = self.coarse_representation[self.rgb_to_coarse(y_e, x_e)]

            if coarse_state_e == coarse_state:

                return True


    def configure_env(self, rand_config=False):
        '''
        Function to create and add all the structural elements (not the agent) to the environment, e.g. wall tiles

        '''

        if rand_config:

            num_tiles = random.choice(np.arange(0, self.num_states // 2))

            for t in range(num_tiles):

                positioning_tile = True

                while positioning_tile:

                    # Randomly picking some indices for the coarse-grained position of the tile
                    m = random.choice(np.arange(self.grid_height))
                    n = random.choice(np.arange(self.grid_width))
                    # Convert coarse-grained position of the wall tile
                    y, x = self.coarse_to_rgb(m, n)
                    # Create wall tile and set its position
                    wall_tile = Wall("wall", self.x_max, self.x_min, self.y_max, self.y_min)
                    wall_tile.set_position(x, y)

                    if not (self.is_colliding(wall_tile.get_position(), *self.elements.values())):
                        positioning_tile = False

                # Adding the tile to the environment's dictionary of elements
                self.elements[f'tile_{y}{x}'] = wall_tile

        else:

            # for t in range(4):

            #     # Convert coarse-grained position of the wall tile
            #     y, x = self.coarse_to_rgb(2, 1 + t)
            #     # Create wall tile and set its position
            #     wall_tile = Wall("wall", self.x_max, self.x_min, self.y_max, self.y_min)
            #     wall_tile.set_position(x, y)
            #     # Adding the tile to the environment's dictionary of elements
            #     self.elements[f'tile_{y}{x}'] = wall_tile

            for t in self.walls:

                # Convert coarse-grained position of the wall tile
                y, x = self.coarse_to_rgb(t[0], t[1])
                # Create wall tile and set its position
                wall_tile = Wall("wall", self.x_max, self.x_min, self.y_max, self.y_min)
                wall_tile.set_position(x, y)
                # Adding the tile to the environment's dictionary of elements
                self.elements[f'tile_{y}{x}'] = wall_tile


    def configure_agent(self, position=0, rand_pos=False, goal=None):
        '''
        Function to create representation of the agent in the environment.

        '''

        # Creating agent object
        squirrel = Squirrel("squirrel", self.x_max, self.x_min, self.y_max, self.y_min)
        # Set goal for the agent in RGB representation
        if goal != None:
            # Assert that the goal state is valid  
            assert self.observation_space.contains(goal), "Invalid goal state."

            # Retrieve indices of the goal in the coarse-grained representation
            m, n = np.where(self.coarse_representation==goal)
            # Conversion into RGB representation
            y_goal, x_goal = self.coarse_to_rgb(int(m), int(n))            
            # Instantiating Goal object
            agent_goal = Goal("goal", self.x_max, self.x_min, self.y_max, self.y_min)
            # Set goal position
            agent_goal.set_position(x_goal, y_goal)
            # Adding goal element to the environment structure
            self.elements["goal"] = agent_goal
            # Giving goal to the agent
            squirrel.set_goal(x_goal, y_goal)

        if rand_pos:
            # Set random agent position using RGB representation
            positioning_agent = True

            while positioning_agent:

                # Determine a place to intialise the agent in
                x = random.randrange(int(0), int(625), 125)
                y = random.randrange(int(0), int(625), 125)
                squirrel.set_position(x,y)

                if not (self.is_colliding(squirrel.get_position(), *self.elements.values())):
                    positioning_agent = False
        else:
            # Set agent position using state representation
            i, j = np.where(self.coarse_representation==position)
            y, x = self.coarse_to_rgb(int(i), int(j))
            squirrel.set_position(x, y)


        # Add agent to the dictionary of environment's elements 
        self.elements["agent"] = squirrel


    def get_action_meaning(self, action: int) -> str:
        '''
        Function that takes the integer representation of an action and returns its meaning.

        '''

        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid action."

        actions_meanings = {0: "up", 1: "right", 2: "down", 3: "left"}
        # print(f'Action is: {action}')
        # print(type(action))
        # print(actions_meanings[action])

        return actions_meanings[action]


    def draw_canvas(self):
        '''
        Function to draw the elements of the environment on the RGB canvas.

        '''

        # Init the canvas 
        self.canvas = np.ones(self.rgb_image_shape) * 1 

        # Drawing a grid by setting rows and columns to zero every l positions
        for i in [np.arange(0, self.y_max, self.l)]:
            self.canvas[i, :] = 0 

        for i in [np.arange(0, self.x_max, self.l)]:
            self.canvas[:, i] = 0 

        # Draw the element/agent on canvas
        for elem in self.elements.values():

            elem_shape = elem.icon.shape

            if isinstance(elem, Squirrel):
                x,y = elem.x + 15, elem.y + 15
            else:
                x,y = elem.x, elem.y

            #print(f'Element shape 1: {elem_shape[1]}')
            #print(f'y coord: {y}')
            #print(f'x coord: {x}')

            self.canvas[y: y + elem_shape[1], x: x + elem_shape[0], :] = elem.icon

        # Save canvas frame
        self.canvas_frames.append(np.copy(self.canvas))


    def draw_intermediate_canvas(self, intended_action: str):
        '''
        Function to draw the elements of the environment on the RGB canvas but with the crucial 
        difference that the agent is displaced toward the direction of movement (determined by
        the intended action).

        '''

        # Compute a delta in the direction of intended movement
        if intended_action == 'up':
            delta = (0, -10)

        elif intended_action == 'right':
            delta = (10, 0)

        elif intended_action == 'down':
            delta = (0, 10)

        else:
            delta = (-10, 0)

        # Init the canvas 
        self.canvas = np.ones(self.rgb_image_shape) * 1 

        # Drawing a grid by setting rows and columns to zero every l positions
        for i in [np.arange(0, self.y_max, self.l)]:
            self.canvas[i, :] = 0 

        for i in [np.arange(0, self.x_max, self.l)]:
            self.canvas[:, i] = 0 

        # Draw the element/agent on canvas
        for elem in self.elements.values():

            elem_shape = elem.icon.shape

            if isinstance(elem, Squirrel):
                x,y = elem.x + 15 + delta[0], elem.y + 15 + delta[1]
            else:
                x,y = elem.x, elem.y

            self.canvas[y: y + elem_shape[1], x: x + elem_shape[0], :] = elem.icon

        # Save canvas frame
        self.canvas_frames.append(np.copy(self.canvas))


    def action_effect(self, x_pos: int, y_pos: int, agent_move: str) -> tuple:
        '''
        Function that returns the action effect in the environment, i.e. a movement to new RGB coordinates.
        '''

        # Get the absolute value of the RGB displacement created by the possible actions (row-wise and column-wise)
        p = self.y_max // self.grid_height
        q = self.x_max // self.grid_width

        # Get the actual RGB displacement based on the action
        if agent_move == 'up':
            delta_pos = (0, -p)

        elif agent_move == 'right':
            delta_pos = (q, 0)

        elif agent_move == 'down':
            delta_pos = (0, p)

        else:
            delta_pos = (-q, 0)

        next_x = x_pos + delta_pos[0]
        next_y = y_pos + delta_pos[1]

        return next_x, next_y


    def step(self, action):
        '''
        Function that advance the environment one step based on the agent's action.

        '''

        # Assert that it is a valid action
        action = int(action) 
        assert self.action_space.contains(action), "Invalid action."
        # Get action meaning
        agent_move = self.get_action_meaning(action).lower()
        # List of possible actions
        possible_actions = [self.get_action_meaning(a) for a in range(self.action_space.n)]

        # Flag that marks the termination of an episode
        done = False

        # Get RGB position of the agent
        x_agent, y_agent = self.elements["agent"].get_position()
        # Converte RGB position into coarse-grained state position
        state = self.coarse_representation[self.rgb_to_coarse(y_agent, x_agent)]

        for a in possible_actions:

            if agent_move == a:
                # Draw intermediate frame to show the intended movement of the agent
                self.draw_intermediate_canvas(agent_move)
                # Computing the new position in RGB coordinates
                new_x, new_y = self.action_effect(x_agent, y_agent, agent_move)
                # Determine proposed new position that respects the x, y values bounds
                prop_x, prop_y = self.elements["agent"].proposed_position(new_x, new_y)
                # Checking if the proposed position is already occupied (collision) 
                if not (self.is_colliding((prop_x, prop_y), *self.elements.values())):
                    # Move agent to the proposed position
                    self.elements["agent"].move_to_prop_xy()
                    # Converte new RGB position into coarse-grained state position
                    next_state = self.coarse_representation[self.rgb_to_coarse(prop_y, prop_x)]
                    # Break out of the for loop
                    break

                else:
                    # The agent remains at its current position
                    next_state = state
                    # Break out of the for loop
                    break
 
        # Computing reward, checking for terminal/goal state
        agent_goal_xy = self.elements["agent"].get_goal() 
        if state == self.coarse_representation[self.rgb_to_coarse(agent_goal_xy[1], agent_goal_xy[0])]:
            reward = 1
            done = True
        else:
            reward = 0

        # Draw elements on the canvas
        self.draw_canvas()

        return next_state, reward, done, []


    def reset(self):

        # Configuring the agent (setting its position)
        self.configure_agent(self.start_agent_pos, self.rand_pos, self.goal_state)
        # Configuring the environment (e.g. adding wall tiles)
        self.configure_env(self.rand_conf)
        # Reset the canvas frames list
        self.canvas_frames = []
        # Reset the reward
        self.ep_return  = 0
        # Draw elements on the canvas
        self.draw_canvas()

        # Return the starting position (state) for the agent
        return self.start_agent_pos


    def make_video(self, save_dir):
        '''
        Function to create video using the arrays saved in self.canvas_frames.

        N.B. If cv2.VideoWriter is given a numpy array as frame, it seems the data type needs to be set to 'uint8', 
        otherwise the video created is either unsupported or striped. 

        '''

        # Datetime object containing current date and time
        now = datetime.now()
        # Converting data-time in an appropriate string: '_dd.mm.YYYY_H.M.S'
        # N.B. Used for video identification when saving to file
        dt_string = now.strftime('%d.%m.%Y_%H.%M.%S')

        size = self.canvas_frames[0].shape
        fps = 2 
        path_to_video = save_dir.joinpath(f'{dt_string}_video.mp4')
        out = cv2.VideoWriter(str(path_to_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), isColor=True)

        for frame in self.canvas_frames:
            #cv2.imshow('frame', frame)
            #cv2.waitKey(0)
            frame = (frame * 255).astype('uint8')
            #frame = frame.astype('float32')
            out.write(frame)

        out.release()


    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("GridWorldV0", self.canvas)
            cv2.waitKey(0)
        
        elif mode == "rgb_array":
            return self.canvas
        

    def close(self):
        cv2.destroyAllWindows()


class Element(object):

    def __init__(self, name, x_max, x_min, y_max, y_min):

        self.x = 0
        self.y = 0
        self.prop_x = 0
        self.prop_y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
    

    def set_position(self, x, y):

        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)


    def proposed_position(self, x, y):

        self.prop_x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.prop_y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
    
        return self.prop_x, self.prop_y

    def get_position(self):

        return (self.x, self.y)
    

    def move(self, del_x, del_y):

        self.x += del_x
        self.y += del_y
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)


    def move_to_prop_xy(self):

        self.x = self.prop_x
        self.y = self.prop_y


    def clamp(self, n, minn, maxn):

        return max(min(maxn, n), minn)


class Squirrel(Element):

    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Squirrel, self).__init__(name, x_max, x_min, y_max, y_min)

        path_to_image = IMAGES_DIR.joinpath("Skippy.jpg")
        self.icon = cv2.imread(str(path_to_image)) / 255.0
        #self.icon = self.icon.astype('float32')
        #self.icon = cv2.cvtColor(self.icon, cv2.COLOR_BGR2RGB)
        self.icon_w = 90
        self.icon_h = 90
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w)).astype('float32')

        self.goal_x = None
        self.goal_y = None


    def set_goal(self, x_goal, y_goal):
        '''
        Function to set the goal state for the agent.

        '''

        assert x_goal >= self.x_min and x_goal <= self.x_max, print('Goal RGB x coordinate not within valid bounds.')
        assert y_goal >= self.y_min and y_goal <= self.y_max, print('Goal RGB y coordinate not within valid bounds.')

        self.goal_x = x_goal
        self.goal_y = y_goal


    def get_goal(self):
        '''
        Function to get the goal state for the agent.

        '''

        return (self.goal_x, self.goal_y)
            

class Wall(Element):

    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Wall, self).__init__(name, x_max, x_min, y_max, y_min)

        path_to_image = IMAGES_DIR.joinpath("wall.jpg")
        self.icon = cv2.imread(str(path_to_image)) / 255.0
        #self.icon = self.icon.astype('float32')
        #self.icon = cv2.cvtColor(self.icon, cv2.COLOR_BGR2RGB)
        self.icon_w = 125
        self.icon_h = 125
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w)).astype('float32')


class Goal(Element):

    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Goal, self).__init__(name, x_max, x_min, y_max, y_min)

        path_to_image = IMAGES_DIR.joinpath("walnut_c.jpg")
        self.icon = cv2.imread(str(path_to_image)) / 255.0
        #self.icon = self.icon.astype('float32')
        #self.icon = cv2.cvtColor(self.icon, cv2.COLOR_BGR2RGB)
        self.icon_w = 125
        self.icon_h = 125
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w)).astype('float32')
        
        