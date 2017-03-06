import util,submission

#def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,
#             sort=False):

mdp = submission.smallMDP
rl = submission.QLearningAlgorithm(mdp.actions,mdp.discount(),submission.identityFeatureExtractor,explorationProb=0)
res = util.simulate(mdp,rl,numTrials=30000)

f=open('sim_res.txt','w')
f.write(str(res)+"\n")
f.close()

print len(res)

p={}
for state in mdp.states:
    pi_rl[state] = rl.getAction(state)
print "small test case"
print "pi of reinforcement learning is:"
print pi_rl

algo = submission.ValueIteration()
algo.solve(mdp)
print "pi of Value iteration is:"
print algo.pi

