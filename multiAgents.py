# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        currFood = currentGameState.getFood()
        list = currFood.asList()
        list.sort(lambda x,y: util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))
        food_dis = util.manhattanDistance(newPos, list[0])
        ghost = [g.getPosition() for g in newGhostStates]
        ghost.sort(lambda x,y: util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))
        ghosts = [util.manhattanDistance(newPos, g) for g in ghost]
        if len(ghosts) == 0:
            score = 0
        else:
            if ghosts[0] == 0:
                score = -9999
            else:
                score = ghosts[0]
        if food_dis != 0:
            return score/float(food_dis)
        return score*2
        
        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    def successors(self, gameState, agentInd):
        """
        Returns a list of game states one action away for agentInd.
        """
        actions = gameState.getLegalActions(agentInd)
        return [(gameState.generateSuccessor(agentInd, action), action) for action in actions]
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def gameEnd(gameState, depth, agent):
            if gameState.isWin():
                return True
            if gameState.isLose():
                return True
            if agent < gameState.getNumAgents():
                if len(gameState.getLegalActions(agent)) == 0:
                    return True
                return False
            if self.depth == depth:
                return True
            return False
            
        def node(state, depth, agent):
            if gameEnd(state, depth, agent):
                return self.evaluationFunction(state)
            if agent < state.getNumAgents():
                next_states = [node(state.generateSuccessor(agent, a), depth, agent + 1) for a in state.getLegalActions(agent)]
                if agent == 0:
                    return max(next_states)
                return min(next_states)
            else:
                return node(state, depth + 1, 0)
        list = [node(gameState.generateSuccessor(0, state), 1, 1) for state in gameState.getLegalActions(0)]
        moves = [state for state in gameState.getLegalActions(0)]
        return moves[list.index(max(list))]
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        val = None
        alpha = None
        beta = None
        best = None
        
        def endGame(state, depth):
            if state.isWin():
                return True
            if state.isLose():
                return True
            if depth > self.depth:
                return True
            return False
            
        def node(state, depth, agent, alpha, beta):
            if agent != 0:
                if agent != state.getNumAgents():
                    worst = None
                    actions = state.getLegalActions(agent)
                    if len(actions) == 0:
                        return self.evaluationFunction(state)
                    for action in actions:
                        successor = node(state.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)
                        if worst is None:
                            worst = successor
                        else:
                            worst = min(worst, successor)
                        if beta is None:
                            beta = worst
                        else:
                            beta = min(beta, worst)
                        if alpha is not None:
                            if alpha > worst:
                                return worst
                    return worst
                return node(state, depth + 1, 0, alpha, beta)
            else:
                if endGame(state, depth):
                    return self.evaluationFunction(state)
                best = None
                actions = state.getLegalActions(agent)
                if len(actions) == 0:
                    return self.evaluationFunction(state)
                for action in actions:
                    successor = node(state.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)
                    best = max(best, successor)
                    alpha = max(alpha, best)
                    if beta is not None:
                        if beta < best:
                            return best
                return best

        for action in gameState.getLegalActions(0):
            val = max(val, node(gameState.generateSuccessor(0, action), 1, 1, alpha, beta))
            if alpha is None:
                alpha = val
                best = action
            else:
                if val > alpha:
                    best = action
                alpha = max(val,alpha)
        return best
                
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def gameEnd(gameState, depth, agent):
            if gameState.isWin():
                return True
            if gameState.isLose():
                return True
            if agent < gameState.getNumAgents():
                if len(gameState.getLegalActions(agent)) == 0:
                    return True
                return False
            if self.depth == depth:
                return True
            return False
            
        def node(state, depth, agent):
            if gameEnd(state, depth, agent):
                return self.evaluationFunction(state)
            if agent < state.getNumAgents():
                next_states = [node(state.generateSuccessor(agent, a), depth, agent + 1) for a in state.getLegalActions(agent)]
                if agent == 0:
                    return max(next_states)
                return np.mean(next_states)
            else:
                return node(state, depth + 1, 0)
        list = [node(gameState.generateSuccessor(0, state), 1, 1) for state in gameState.getLegalActions(0)]
        moves = [state for state in gameState.getLegalActions(0)]
        return moves[list.index(max(list))]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 1000000*currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()  
    food = currentGameState.getFood()
    manDistanceToClosestPower = food.height+food.width
    for capsule in currentGameState.getCapsules():
        manDistanceToClosestPower = min(manDistanceToClosestPower, util.manhattanDistance(pos, capsule))
    disGhost = food.height+food.width
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer <= 0:
            disGhost = min(disGhost, util.manhattanDistance(pos, ghost.getPosition()))
    foodList = food.asList()
    foodDist = 0
    if foodList:
        foodList.sort(lambda x,y: util.manhattanDistance(pos, x)-util.manhattanDistance(pos, y))
        foodDist = util.manhattanDistance(pos, foodList[0])
    return disGhost + currentGameState.getScore() - 100*len(currentGameState.getCapsules()) - 20*len(currentGameState.getGhostStates()) - foodDist

    

# Abbreviation
better = betterEvaluationFunction

