#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    phi=collections.defaultdict(float)
    x=x.split()
    for word in x:
       phi[word]+=1
    return phi
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    
    #take the gradient of the hinge loss
    #- from lecture we know this value = -1[margin<1]phi(x)y
    def dhingeLoss(features,weights,y):
        #compute margin = w dot phi(x) y
        margin = dotProduct(weights,features)*y
        if margin < 1:
            marginLessThan1=1
        else:
            marginLessThan1=0
        gradient={}
        for key,value in features.items():
            gradient[key] = -marginLessThan1*value*y
        return gradient
    
    for i in range(0,numIters):
        for x,y in trainExamples:
            increment(weights,-eta,dhingeLoss(featureExtractor(x),weights,y))
        '''
        trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        testError = evaluatePredictor(testExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        print("Iteration: "+str(i+1)+"------------------\n")
        print('Train Error: '+str(trainError)+"\n")
        print('Test Error: '+str(testError)+"\n")
        '''
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        phi={}
        len_phi = random.randint(1,len(weights))
        for i in range(0,len_phi):
                key = random.choice(weights.keys())
                phi[key] = random.random()-0.5
        if dotProduct(phi,weights)>0:
            y=1
        else:
            y=0
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        x="".join(x.split())
        ret=collections.defaultdict(float)
        for i in range(0,len(x)-n+1):
            ret[x[i:i+n]]+=1
        return ret
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    centroids = []
    assignments = []
    example_length = []
    centroid_length = []
    for i in range(0,len(examples)):
        example_length.append(dotProduct(examples[i],examples[i]))
    # initialize the centroids as random element of examples
    '''
    def L2(v1,v2): #compute the distance between two vectors
        diff_vector={}
        print('example: '+str(v1))
        print('centroid: '+str(v2))
        if v1==v2:
            return 0
        for key in v1:
            if key in v2:
                diff_vector[key] = abs(v1[key]-v2[key])
            else:
                diff_vector[key] = abs(v1[key])
        for key in v2:
            if key not in v1:
                diff_vector[key] = abs(v2[key])
        return dotProduct(diff_vector,diff_vector)
    '''
    def L2(v1,v2,v1_index,v2_index):
        v1v2 = dotProduct(v1,v2)
        res = -2*v1v2 + example_length[v1_index] + centroid_length[v2_index]
        return res
    for i in range(0,K):
        random.seed(42)
        rand_centroid = random.choice(examples)
        while rand_centroid in centroids:
            rand_centroid = random.choice(examples)
        centroids.append(rand_centroid)
        centroid_length.append(dotProduct(rand_centroid,rand_centroid))
    for i in range(0,len(examples)):
        assignments.append(0)
    # start the iteration
    for i in range(0,maxIters):
        #print('Iteration: '+str(i))
        prev_centroids = list(centroids)
        clusters=[] # a list of lists. Stores the examples assigned to each
                    # - centroid.
        for j in range(0,K):
            clusters.append([])
        L2norms=[] # a list of lists. L2norms[i] is the distances between
                   # - examples[i] and all centroids
        # compute the distance of each example to each centroid
        #for example in examples:
        for j in range(0,len(examples)):
            example_L2_norms=[]
            #for centroid in centroids:
            for k in range(0,len(centroids)):
                example_L2_norms.append(L2(examples[j],centroids[k],j,k))
            L2norms.append(example_L2_norms)
        # based on the distances, figure out which centroid should each example assign itself to
        for j in range(0,len(L2norms)):
            assigned_centroid = L2norms[j].index(min(L2norms[j]))
            assignments[j] = assigned_centroid
            clusters[assigned_centroid].append(examples[j])
        # based on the assignments, re-compute the centroids
        for j in range(0,len(centroids)):
            #print('j is: '+str(j))
            new_centroid = {}
            points = clusters[j]
            #print('Clusters: '+str(clusters))
            for pt in points:
                for key,value in pt.items():
                    if key in new_centroid:
                        new_centroid[key] += value
                    else:
                        new_centroid[key] = value
            for key in new_centroid: # take the average
                new_centroid[key] = new_centroid[key]/float(len(points))
            #print('New centroid: '+str(new_centroid))
            centroids[j] = new_centroid
            centroid_length[j] = dotProduct(new_centroid,new_centroid)
            #print('Centroids: '+str(centroids))
        if prev_centroids == centroids:
            break
            
    #end of for i in (0,maxIter)

    # compute final reconstruction loss
    kloss = 0
    for i in range(0,len(examples)):
        kloss += L2(examples[i],centroids[assignments[i]],i,assignments[i])

    print('Final Reconstruction Loss: '+str(kloss)+'\n')
    return centroids,assignments,kloss
    # END_YOUR_CODE
