# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        #util.raiseNotDefined()
        pass

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        #util.raiseNotDefined()
        pass

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        #util.raiseNotDefined()
        pass

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        #util.raiseNotDefined()
        pass


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]    
    
def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    pacman_path = [[]]                         # Collection of directions to move the pacman
    actual_movement = util.Stack()             # Next steps 
    visited_nodes = util.Queue()               # All visited nodes
    last_position = problem.getStartState()    # Last position before looping in while
    actual_movement.push(last_position)
    visited_nodes.push(last_position)

    while not problem.isGoalState(last_position):
        verifying_possible_moviments = actual_movement.pop()
        last_path = pacman_path.pop(-1)

        for neighbor in problem.getSuccessors(verifying_possible_moviments):
            new_direction = last_path.copy()
            new_direction.append(getDirection(neighbor))

            if getPosition(neighbor) not in visited_nodes.list:
                actual_movement.push(getPosition(neighbor))
                visited_nodes.push(getPosition(neighbor))
                
                pacman_path.append(new_direction)
                
        last_position = getLastPosition(actual_movement)

    # Returning the last tried movement, to go out of the while
    # the algorithm needs to find the goal
    return pacman_path[-1]

        

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    pacman_path = [[]]
    actual_movement = util.Queue()           # Collection of directions to move the pacman
    visited_nodes = util.Queue()             # Next steps
    last_position = problem.getStartState()  # All visited nodes
    actual_movement.push(last_position)      # Last position before looping in while
    visited_nodes.push(last_position)        # The actual node 

    while not problem.isGoalState(last_position):
        verifying_possible_moviments = actual_movement.pop()
        last_path = pacman_path.pop(0)

        for neighbor in problem.getSuccessors(verifying_possible_moviments):
            new_direction = last_path.copy()
            new_direction.append(getDirection(neighbor))

            if getPosition(neighbor) not in visited_nodes.list:
                visited_nodes.push(getPosition(neighbor))
                actual_movement.push(getPosition(neighbor))
                
                pacman_path.append(new_direction)
                
        last_position = getLastPosition(actual_movement)

    # Returning the best found movement
    return pacman_path[0]


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pacman_path = [[]]
    actual_movement = {}                            # Collection of directions to move the pacman and its cost
    visited_nodes = util.Queue()                    # Next steps
    last_position = problem.getStartState()         # All visited nodes
    movement_cost = 0                               # All paths and its cost
    actual_movement[last_position] = movement_cost  # Last position before looping in while

    while not problem.isGoalState(last_position):
        
        # To get the cheapest path is necessary to find the smallest number
        # in the actual_movement dict, finding it we need to remove the
        # element in the same position in the pacman_path list of lists
        cheapest_path = min(actual_movement, key=actual_movement.get)
        verifying_possible_moviments = cheapest_path
        pos_in_dict = list(actual_movement).index(cheapest_path)
        del actual_movement[cheapest_path]

        # Save the last_path and pop this element from the list
        # it saves some memory and remove paths that don't go
        # to the goal
        last_path = pacman_path[pos_in_dict]
        del pacman_path[pos_in_dict]

        for neighbor in problem.getSuccessors(verifying_possible_moviments):
            new_direction = last_path.copy()
            new_direction.append(getDirection(neighbor))
            actual_path_cost = problem.getCostOfActions(new_direction)

            if getPosition(neighbor) not in visited_nodes.list:
                visited_nodes.push(getPosition(neighbor))
                # Getting the cost from previous walked path, plus the actual one
                actual_movement[getPosition(neighbor)] = actual_path_cost + getMovementCost(neighbor)
                
                pacman_path.append(new_direction)
        
        # The last calculated position is the cheapest
        last_position = list(actual_movement.keys())[-1]

    # Returning the last tried movement, to go out of the while
    # the algorithm needs to find the goal
    return pacman_path[-1]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pacman_path = [[]]
    actual_movement = {}                            # Collection of directions to move the pacman and its cost
    visited_nodes = util.Queue()                    # Next steps
    last_position = problem.getStartState()         # All visited nodes
    movement_cost = 0                               # All paths and its cost
    actual_movement[last_position] = movement_cost  # Last position before looping in while

    while not problem.isGoalState(last_position):
        
        # To get the cheapest path is necessary to find the smallest number
        # in the actual_movement dict, finding it we need to remove the
        # element in the same position in the pacman_path list of lists
        cheapest_path = min(actual_movement, key=actual_movement.get)
        verifying_possible_moviments = cheapest_path
        pos_in_dict = list(actual_movement).index(cheapest_path)
        del actual_movement[cheapest_path]

        # Save the last_path and pop this element from the list
        # it saves some memory and remove paths that don't go
        # to the goal
        last_path = pacman_path[pos_in_dict]
        del pacman_path[pos_in_dict]

        for neighbor in problem.getSuccessors(verifying_possible_moviments):
            new_direction = last_path.copy()
            new_direction.append(getDirection(neighbor))
            actual_path_cost = problem.getCostOfActions(new_direction)

            if getPosition(neighbor) not in visited_nodes.list:
                visited_nodes.push(getPosition(neighbor))
                # Getting the cost from previous walked path, plus the actual one
                actual_movement[getPosition(neighbor)] = actual_path_cost + getMovementCost(neighbor) + heuristic(getPosition(neighbor), problem)
                
                pacman_path.append(new_direction)
        
        # The last calculated position is the cheapest
        last_position = list(actual_movement.keys())[-1]

    # Returning the last tried movement, to go out of the while
    # the algorithm needs to find the goal
    return pacman_path[-1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

def getPosition(t: tuple):
    return t[0]

def getDirection(t: tuple):
    return t[1]

def getMovementCost(t: tuple):
    return t[2]

def getLastPosition(q: util.Queue):
    return q.list[0]

def getLastPosition(s: util.Stack):
    return s.list[-1]