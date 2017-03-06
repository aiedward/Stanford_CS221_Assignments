from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    # This returns the maximized move for agent 0
    self.nodes_visited = 0
    def minmax_rec(state,agent,d):
        self.nodes_visited += 1
        if agent == state.getNumAgents():
            agent = 0
        if d == 0:
            return (self.evaluationFunction(state),Directions.NORTH)
        if len(state.getLegalActions(agent)) == 0:
            return (self.evaluationFunction(state),Directions.NORTH)
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state),Directions.NORTH)
        ret = []
        if agent == 0: # if agent is Pacman
            for action in state.getLegalActions(agent):
                if action == None or action == Directions.STOP:
                    continue
                nextState = state.generateSuccessor(agent,action)
                v_and_action = (minmax_rec(nextState,agent+1,d)[0],action)
                ret.append(v_and_action)
            #print ret
            return max(ret)
        
        for action in state.getLegalActions(agent): # if agent is a ghost
            if action == None or action == Directions.STOP:
                continue
            nextState = state.generateSuccessor(agent,action)
            if agent == gameState.getNumAgents() - 1: # if agent is the last ghost
                v_and_action = (minmax_rec(nextState,agent+1,d-1)[0],action)
            else:
                v_and_action = (minmax_rec(nextState,agent+1,d)[0],action)
            ret.append(v_and_action)
        return min(ret)
    res = minmax_rec(gameState,self.index,self.depth)
    print self.nodes_visited
    return res[1]
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    self.nodes_visited = 0
    def alphaBetaRec(alpha,beta,state,agent,d):
        #print("Alpha is "+str(alpha))
        #print("Beta is "+str(beta)) 
        self.nodes_visited += 1
        if agent == state.getNumAgents():
            agent = 0
        if d == 0:
            return (self.evaluationFunction(state), Directions.NORTH)
        if len(state.getLegalActions(agent)) == 0:
            return (self.evaluationFunction(state), Directions.NORTH)
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state), Directions.NORTH)
        ret = []
        if agent == 0: # if agent is Pacman
            for action in state.getLegalActions(agent):
                if action == None or action == Directions.STOP:
                    continue
                nextState = state.generateSuccessor(agent,action)
                v = alphaBetaRec(alpha,beta,nextState,agent+1,d)
                if v != None:
                    v_and_action = (v[0],action)
                    node_v = v[0]
                    if node_v >= beta:
                        return
                    else:
                        alpha = max(node_v,alpha)
                    ret.append(v_and_action)
            if len(ret) > 0:
                return max(ret)
            return
        else: # if agent is a ghost
            for action in state.getLegalActions(agent):
                if action == None or action == Directions.STOP:
                    continue
                nextState = state.generateSuccessor(agent,action)
                if agent == gameState.getNumAgents() - 1: # if agent is the last ghost
                    v = alphaBetaRec(alpha,beta,nextState,agent+1,d-1)
                else:
                    v = alphaBetaRec(alpha,beta,nextState,agent+1,d)
                if v != None:
                    v_and_action = (v[0],action)
                    node_v = v[0]
                    if node_v <= alpha:
                        return
                    else:
                        beta = min(node_v,beta)
                    ret.append(v_and_action)
            if len(ret) > 0:
                return min(ret)
            return
    ret = alphaBetaRec(-9223372036854775807,9223372036854775807,gameState,self.index,self.depth)
    print self.nodes_visited
    return ret[1]
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    def expected_minimax_rec(state,agent,d):
        if agent == state.getNumAgents():
            agent = 0
        if d == 0:
            return (self.evaluationFunction(state),Directions.NORTH)
        if len(state.getLegalActions(agent)) == 0:
            return (self.evaluationFunction(state),Directions.NORTH)
        if state.isWin() or state.isLose():
            return (self.evaluationFunction(state),Directions.NORTH)
        ret = []
        if agent == 0: # if agent is Pacman
            for action in state.getLegalActions(agent):
                if action == None or action == Directions.STOP:
                    continue
                nextState = state.generateSuccessor(agent,action)
                v_and_action = (expected_minimax_rec(nextState,agent+1,d)[0],action)
                ret.append(v_and_action)
            #print ret
            return max(ret)

        # if agent is a ghost
        if agent > 0:
            result = (0,Directions.STOP)
            for action in state.getLegalActions(agent): # if agent is a ghost
                prob_of_this_action = 1.0 / len(state.getLegalActions(agent))
                if action == None or action == Directions.STOP:
                    continue
                nextState = state.generateSuccessor(agent,action)
                if agent == gameState.getNumAgents() - 1: # if agent is the last ghost
                    result = (result[0]+prob_of_this_action*(expected_minimax_rec(nextState,agent+1,d-1)[0]),action)
                else:
                    result = (result[0]+prob_of_this_action*(expected_minimax_rec(nextState,agent+1,d)[0]),action)
            return result
    res = expected_minimax_rec(gameState,self.index,self.depth)
    return res[1]
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """

  # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
  #raise Exception("Not implemented yet")
  score = 4.5*currentGameState.getScore()
  if currentGameState.isWin():
      return 9223372036854775807
  if currentGameState.isLose():
      return -9223372036854775807
  pacman_loc = currentGameState.getPacmanPosition()
  ghost_pos = list(currentGameState.getGhostPositions())
  food = currentGameState.getFood().asList()
  closestfood = 9223372036854775807
  closestcapsule = 9223372036854775807
  for i in range(0,len(food)):
      #if (util.manhattanDistance(food[i],pacman_loc) < closestfood):
      closestfood = min(util.manhattanDistance(food[i],pacman_loc),closestfood)
  for i in range(0,len(currentGameState.getCapsules())):
      closestcapsule = min(util.manhattanDistance(currentGameState.getCapsules()[i],pacman_loc),closestcapsule)
  i = 1
  closestGhost = 9223372036854775807
  closestScaredGhost = 9223372036854775807
  #for i in range(1,currentGameState.getNumAgents()):
  #    ghost_state = currentGameState.getGhostState(i)
  #    print ghost_state
  num_scared_ghost = 0
  num_ghosts_in_prison = 0
  while i <= currentGameState.getNumAgents() - 1:
      ghostPos = currentGameState.getGhostPosition(i)
      if ghostPos[1] >= 5 and ghostPos[0] >= 7 and ghostPos[0] <= 12:
          num_ghosts_in_prison += 1
      ghost_state = currentGameState.getGhostState(i)
      if ghost_state.scaredTimer > 0:
          next_scared_ghost = util.manhattanDistance(pacman_loc,ghostPos)
          closestScaredGhost = min(closestScaredGhost,next_scared_ghost)
          num_scared_ghost += 1
      else:
          ghost_dist = util.manhattanDistance(pacman_loc,ghostPos)
          closestGhost = min(closestGhost,ghost_dist)
      i += 1

  if pacman_loc[1] >= 5 and pacman_loc[0] >= 7 and pacman_loc[0] <= 12:
      score -= 99999999
  
  score -= 2.5 * 1/float(closestGhost)
  score -= 1.5 * closestfood
  #score += 15 * 1./closestfood
  score += 10 * 1/float(closestScaredGhost)
  score += 3 * num_ghosts_in_prison
  #score -= 4.5 * closestcapsule
  score -= 4.5 * len(food)
  score -= 100 * len(currentGameState.getCapsules())
  return score
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
