import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return 0
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return state == len(self.query)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        results = []
        for i in range(state,1+len(self.query)):
            # return format: (action, newState, cost)
            results.append((self.query[state:i],i,self.unigramCost(self.query[state:i])))
        return results
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))
    
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return (wordsegUtil.SENTENCE_BEGIN,0)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return state[1] == len(self.queryWords)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        word_to_fill = self.queryWords[state[1]]
        next_word_to_fill_index = state[1]+1
        ret=[]
        edges = self.possibleFills(word_to_fill)
        if not edges: # if there's nothing to fill, just add the word itself
            edges_tmp = [word_to_fill]
            edges = set(edges_tmp)
        #print('Edges for current word ['+str(word_to_fill)+']: '+str(edges))
        for word in edges:
            cost = self.bigramCost(state[0],word)
            #print("Cost of bigram ("+state[0]+","+word+") is "+str(cost))
            next_state = (word,next_word_to_fill_index)
            ret.append((word,next_state,cost))

        return ret
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    #print("Test sequence: "+str(queryWords))
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords,bigramCost,possibleFills))
    #print("result: "+str(ucs.actions)+"\n")
    return " ".join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        self.subproblem = VowelInsertionProblem(self.query,self.bigramCost,self.possibleFills)
        starting_state = self.subproblem.startState()
        #print("Starting State is: "+str(starting_state))
        return starting_state
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return self.subproblem.isEnd(state)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 15 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        ret = []
        query_length = len(self.query)
        #print("Current State is  "+str(state))
        for i in range(state[1],query_length+1):
            segment = self.query[state[1]:i]
            edges = self.possibleFills(segment)
            for word in edges:
                next_state = (word,i)
                cost = self.bigramCost(state[0],word)
                ret.append((word,next_state,cost))
        return ret
        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    print("Query is "+str(query))
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query,bigramCost,possibleFills))
    print("Result is "+str(ucs.actions))
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
