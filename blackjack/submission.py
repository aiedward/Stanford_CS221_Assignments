import util, math, random, pdb
from collections import defaultdict

############################################################
class ValueIteration(util.MDPAlgorithm):
    '''
    Solve the MDP using value iteration.  Your solve() method must set
    - self.V to the dictionary mapping states to optimal values
    - self.pi to the dictionary mapping states to an optimal action
    Note: epsilon is the error tolerance: you should stop value iteration when
    all of the values change by less than epsilon.
    The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
    '''
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        def computeQ(mdp, V, state, action):
            # Return Q(state, action) based on V(state).
            return sum(prob * (reward + mdp.discount() * V[newState]) \
                            for newState, prob, reward in mdp.succAndProbReward(state, action))

        def computeOptimalPolicy(mdp, V):
            # Return the optimal policy given the values V.
            pi = {}
            for state in mdp.states:
                pi[state] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
            return pi

        V = defaultdict(float)  # state -> value of state
        numIters = 0
        while True:
            newV = {}
            for state in mdp.states:
                newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
            numIters += 1
            if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
                V = newV
                break
            V = newV

        # Compute the optimal policy now
        pi = computeOptimalPolicy(mdp, V)
        print "ValueIteration: %d iterations" % numIters
        self.pi = pi
        self.V = V

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        if state == 1:
            return ['quit']
        if state == 0:
            return ['stay', 'quit']
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        if state == 1:
            return []
        if state == 0:
            if action == 'stay':
                t0 = 0
                t1 = 1
                t0_noisy = 0.5*t0+0.25
                t1_noisy = 0.5*t1+0.25
                #return [(0,t0,5),(1,t1,10)]
                return [(0,t0_noisy,5),(1,t1_noisy,10)]
            if action == 'quit':
                return []
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return 1
        # END_YOUR_CODE

mdp = CounterexampleMDP()
test = ValueIteration()
test.solve(mdp)
print test.V

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        #print state
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        total,next_card,deckCardCounts = state
        if deckCardCounts == None:
            return []
        # can get any of the remaining cards in the deck at equal probability
        ret = []
        #prob_for_each_card_value = []
        #num_of_cards_in_deck = sum(deckCardCounts)
        #for count_per_value in deckCardCounts:
        #    prob_for_this_value = count_per_value/float(num_of_cards_in_deck)
        #    prob_for_each_card_value.append(prob_for_this_value)
        if action == 'Take':
            if next_card == None: # if player did not peek before
            # can take any of the possible card values
                for i_cardValue in range(0,len(self.cardValues)):
                    if deckCardCounts[i_cardValue] == 0: # probability will be 0
                        continue
                    next_state_total = total + self.cardValues[i_cardValue]
                    reward = 0
                    #prob = prob_for_each_card_value[i_cardValue]
                    prob = deckCardCounts[i_cardValue]/float(sum(deckCardCounts))
                    if next_state_total > self.threshold: # if bust happens
                        next_deckCardCounts = None # game ends
                    else:
                        next_deckCardCounts = list(deckCardCounts)
                        next_deckCardCounts[i_cardValue] -= 1
                        next_deckCardCounts = tuple(next_deckCardCounts)
                        if sum(next_deckCardCounts) == 0:
                            next_deckCardCounts = None
                            reward = next_state_total
                    next_state = (next_state_total,next_card,next_deckCardCounts) # next_card is kept unchanged
                    #pdb.set_trace()
                    ret.append((next_state,prob,reward))
            else: # if peeked previously
                prob = 1
                next_state_total = total + self.cardValues[next_card]
                reward = 0
                if next_state_total > self.threshold: # if bust happens
                    next_deckCardCounts = None # game ends
                else:
                    next_deckCardCounts = list(deckCardCounts)
                    next_deckCardCounts[next_card] -= 1
                    next_deckCardCounts = tuple(next_deckCardCounts)
                    if sum(next_deckCardCounts) == 0:
                        next_deckCardCounts = None
                        reward = next_state_total
                next_state = (next_state_total,None,next_deckCardCounts)
                ret.append((next_state,prob,reward))
        elif action == 'Peek':
            if next_card != None: # if peeked before
                return []
            else:
                # the top card peeked can be any of the available values
                for i_cardValue in range(0,len(self.cardValues)):
                    next_card = i_cardValue
                    #prob = prob_for_each_card_value[i_cardValue]
                    prob = deckCardCounts[i_cardValue]/float(sum(deckCardCounts))
                    if prob != 0:
                        #pdb.set_trace()
                        reward = -self.peekCost
                        next_state = (total,next_card,deckCardCounts)
                        ret.append((next_state,prob,reward))
        elif action == 'Quit':
            reward = total
            prob = 1
            next_state = (total,None,None)
            ret.append((next_state,prob,reward))
        #pdb.set_trace()
        #print ret
        return ret
        # END_YOUR_CODE
        
    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    ret = BlackjackMDP([2,4,6,17],2,20,1)
    return ret
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        if not newState:
            return
        # copied and modified from getAction() function
        actions = self.actions(newState)
        action_prime_opt = max((act,self.getQ(newState, act)) for act in actions)[0]
        #print("New a\' is: "+str(action_prime)+" Old a' is: "+str(action_prime_old_opt))
        current_Qopt = self.getQ(state,action)
        target = reward + self.discount*self.getQ(newState,action_prime_opt)
        # use SGD
        for x, y in self.featureExtractor(state, action):
            self.weights[x] = (1-self.getStepSize())*current_Qopt+self.getStepSize()*target
        # END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
largeMDP.computeStates()



############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    # state format: (total,next_card_index_if_peeked,(deckCardCounts))
    # action format: string 'Take','Peek', or 'Quit'
    # total and the action
    #print('state is: '+str(state))
    #print('action is: '+str(action))
    ret = []
    total_and_action = ((state[0],action),1)
    ret.append(total_and_action)
    # presence/absence of each card and the action
    if state[2] != None:
        presence = ['feature_presence'] #avoid confusion with the next feature
        for card_count in state[2]:
            if card_count != 0:
                presence.append(1)
            else:
                presence.append(0)
        presence.append(action)
        presence_feature = (tuple(presence),1)
        ret.append(presence_feature)
    # the number of cards for each card type and the action
        for i in range(0,len(state[2])):
            num_of_cards_action = (i, state[2][i], action)
            ret.append((num_of_cards_action,1))
    #print('ret is: '+str(ret))
    return ret
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

