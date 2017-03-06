import collections

############################################################
# Problem 3a

def computeMaxWordLength(text):
    """
    Given a string |text|, return the longest word in |text|.  If there are
    ties, choose the word that comes latest in the alphabet.
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    return max(sorted(text.split(),reverse=True), key=len)
    # END_YOUR_CODE

############################################################
# Problem 3b

def manhattanDistance(loc1, loc2):
    """
    Return the Manhattan distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    return abs(loc2[0]-loc1[0])+abs(loc2[1]-loc1[1]) 
    # END_YOUR_CODE

############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    High-level idea: generate sentences similar to a given sentence.
    Given a sentence (sequence of words), return a list of all possible
    alternative sentences of the same length, where each pair of adjacent words
    also occurs in the original sentence. (The words within each pair should appear 
    in the same order in the output sentence as they did in the orignal sentence.)
    Notes:
    - The order of the sentences you output doesn't matter.
    - You must not output duplicates.
    - Your generated sentence can use a word in the original sentence more than
      once.
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    result={}
    def recurse(array):
        if len(array)==len(sentence_list):
            result[tuple(array)]=True
            return
        else:
            last_item=array[len(array)-1]
            #print("Current Array is: "+str(array))
            #print('the last item is: '+str(last_item))
            new_array=[]
            for key in word_pairs:
                if last_item == key[0]:
                    next_item=key[1]
                    #print("Appending the next item [\'"+str(next_item)+"\'] from key: "+str(key))
                    tmp_array=list(array)
                    #print("array is "+str(array))
                    tmp_array.append(next_item)
                    #print("after appending, we got: "+str(tmp_array))
                    new_array.append(tmp_array)
            for new_arr in new_array:
                recurse(new_arr)
    sentence_list=sentence.split()
    #print("Sentence is: "+str(sentence_list))
    #generate word pairs
    word_pairs={}
    for i in range(0,len(sentence_list)-1):
        word_pairs[(sentence_list[i],sentence_list[i+1])]=True
    # END_YOUR_CODE
    #print word_pairs
    for word in sentence_list:
        array_param=[word]
        a=recurse(array_param)
    ret=[]
    for r in result:
        r=" ".join(r)
        ret.append(r)
    return ret
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    product=0
    for key in v1:
        product+=v1[key]*v2[key]
    return product
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    for key in v2:
        v1[key]=v1[key]+scale*v2[key]
    return v1
    # END_YOUR_CODE

############################################################
# Problem 3f

def computeMostFrequentWord(text):
    """
    Splits the string |text| by whitespace and returns two things as a pair: 
        the set of words that occur the maximum number of times, and
    their count, i.e.
    (set of words that occur the most number of times, that maximum number/count)
    You might find it useful to use collections.defaultdict(float).
    """
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    words=collections.defaultdict(float)
    for word in text.split():
        words[word]+=1
    max_val=words[max(words, key=lambda i: words[i])]
    ret=[]
    for key in words:
        if words[key]==max_val:
            ret.append(key)
    ret=(set(ret),max_val)
    return ret
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindrome(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    n = len(text)
    if n==0:
        return 0
    if n==1:
        return 1

    # Use a table to store subproblem results as an application of DP
    # the table stores the longest palindromic subsequence in the substring
    # text[i:j+1]
    P=list()
    for i in range(0,n):
        P_inner=list()
        for j in range(0,n):
            P_inner.append(0)
        P.append(P_inner)

    # Each letter (substring of length 1) is a palindrome of its own
    # so the diagonal is filled with 1
    for i in range(0,n):
        P[i][i] = 1

    for x in range(2, n+1): #start from the substring with only length 2, then expand the substring in every iteration
        for i in range(0,n-x+1):
            j = i+x-1
            if text[i] != text[j]:
                P[i][j] = max(P[i][j-1], P[i+1][j]);
            else:
                P[i][j] = P[i+1][j-1] + 2
    return P[0][n-1]

    # END_YOUR_CODE
