import collections, sys, os
from logic import *

############################################################
# Problem 1: propositional logic
# Convert each of the following natural language sentences into a propositional
# logic formula.  See rainWet() in examples.py for a relevant example.

# Sentence: "If it's summer and we're in California, then it doesn't rain."
def formula1a():
    # Predicates to use:
    Summer = Atom('Summer')               # whether it's summer
    California = Atom('California')       # whether we're in California
    Rain = Atom('Rain')                   # whether it's raining
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    cali_and_summer = And(Summer, California)
    not_rain = Not(Rain)
    return Implies(cali_and_summer,not_rain)
    # END_YOUR_CODE

# Sentence: "It's wet if and only if it is raining or the sprinklers are on."
def formula1b():
    # Predicates to use:
    Rain = Atom('Rain')              # whether it is raining
    Wet = Atom('Wet')                # whether it it wet
    Sprinklers = Atom('Sprinklers')  # whether the sprinklers are on
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    raining_or_sprinklers = Or(Rain, Sprinklers)
    return Equiv(raining_or_sprinklers,Wet)
    # END_YOUR_CODE

# Sentence: "Either it's day or night (but not both)."
def formula1c():
    # Predicates to use:
    Day = Atom('Day')     # whether it's day
    Night = Atom('Night') # whether it's night
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    return Xor(Day,Night)
    # END_YOUR_CODE

############################################################
# Problem 2: first-order logic

# Sentence: "Every person has a mother."
def formula2a():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Mother(x, y): return Atom('Mother', x, y)  # whether x's mother is y

    # Note: You do NOT have to enforce that the mother is a "person"
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    return Forall('$name', Implies(Person('$name'), Exists('$potential_mother_name', Mother('$name', '$potential_mother_name'))))
    # END_YOUR_CODE

# Sentence: "At least one person has no children."
def formula2b():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Child(x, y): return Atom('Child', x, y)    # whether x has a child y

    # Note: You do NOT have to enforce that the child is a "person"
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    def NotChildOf(x,y): return Not(Exists('$y', Child('$x', '$y')))
    return Exists('$x', And(Person('$x'), NotChildOf('$x','$y')))
    # END_YOUR_CODE

# Return a formula which defines Daughter in terms of Female and Child.
# See parentChild() in examples.py for a relevant example.
def formula2c():
    # Predicates to use:
    def Female(x): return Atom('Female', x)            # whether x is female
    def Child(x, y): return Atom('Child', x, y)        # whether x has a child y
    def Daughter(x, y): return Atom('Daughter', x, y)  # whether x has a daughter y
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    FemaleChild =  And(Child('$x', '$y'), Female('$y'))
    DaughterIsFemaleChild = Equiv(Daughter('$x', '$y'), FemaleChild)
    return Forall('$x', Forall('$y', DaughterIsFemaleChild))
    # END_YOUR_CODE

# Return a formula which defines Grandmother in terms of Female and Parent.
# Note: It is ok for a person to be her own parent
def formula2d():
    # Predicates to use:
    def Female(x): return Atom('Female', x)                  # whether x is female
    def Parent(x, y): return Atom('Parent', x, y)            # whether x has a parent y
    def Grandmother(x, y): return Atom('Grandmother', x, y)  # whether x has a grandmother y
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    ExistsFemaleParentParent = Exists('$z', AndList([Female('$y'), Parent('$x', '$z'), Parent('$z', '$y')]))
    return Forall('$x', Forall('$y', Equiv(Grandmother('$x', '$y'), ExistsFemaleParentParent)))
    # END_YOUR_CODE

############################################################
# Problem 3: Liar puzzle

# Facts:
# 0. John: "It wasn't me!"
# 1. Susan: "It was Nicole!"
# 2. Mark: "No, it was Susan!"
# 3. Nicole: "Susan's a liar."
# 4. Exactly one person is telling the truth.
# 5. Exactly one person crashed the server.
# Query: Who did it?
# This function returns a list of 6 formulas corresponding to each of the
# above facts.
# Hint: You might want to use the Equals predicate, defined in logic.py.  This
# predicate is used to assert that two objects are the same.
# In particular, Equals(x,x) = True and Equals(x,y) = False iff x is not equal to y.
def liar():
    def TellTruth(x): return Atom('TellTruth', x)
    def CrashedServer(x): return Atom('CrashedServer', x)
    john = Constant('john')
    susan = Constant('susan')
    nicole = Constant('nicole')
    mark = Constant('mark')
    formulas = []
    # We provide the formula for fact 0 here.
    formulas.append(Equiv(TellTruth(john), Not(CrashedServer(john))))
    # You should add 5 formulas, one for each of facts 1-5.
    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    # Susan says: "It was Nicole!" -> if Susan is telling the truth, then Nicole crashed the server (vice versa)
    formulas.append(Equiv(TellTruth(susan), CrashedServer(nicole)))
    # Mark says: "No, it was Susan!" -> If Mark is telling the truth, then Susan crashed the server & Nicole didn't (vice versa)
    formulas.append(Equiv(TellTruth(mark), CrashedServer(susan)))
    #Nicole says: "Susan's a liar." -> If Nicole is telling the truth, then Susan crashed the server (vice versa)
    #formulas.append(Equiv(TellTruth(nicole), CrashedServer(susan)))
    formulas.append(Equiv(TellTruth(nicole), Not(TellTruth(susan))))
    # You know that exactly one person is telling the truth.
    def OthersNotTellingTruth(x,y): return Forall('$y', Equiv(Not(Equals('$x', '$y')),Not(TellTruth('$y'))))
    formulas.append(Exists('$x', And(TellTruth('$x'), OthersNotTellingTruth('$x','$y'))))
    # You also know exactly one person crashed the server.
    def OthersDidNotCrashServer(x,y): return Forall('$y', Equiv(Not(Equals('$x', '$y')),Not(CrashedServer('$y'))))
    formulas.append(Exists('$x', And(CrashedServer('$x'), OthersDidNotCrashServer('$x','$y'))))
    
    # END_YOUR_CODE
    query = CrashedServer('$x')
    return (formulas, query)

############################################################
# Problem 5: Odd and even integers

# Return the following 6 laws:
# 0. Each number $x$ has a unique successor, which is not equal to $x$.
# 1. Each number is either even or odd, but not both.
# 2. The successor number of an even number is odd.
# 3. The successor number of an odd number is even.
# 4. For every number $x$, the successor of $x$ is larger than $x$.
# 5. Larger is a transitive property: if $x$ is larger than $y$ and $y$ is
#    larger than $z$, then $x$ is larger than $z$.
# Query: For each number, there exists an even number larger than it.
def ints():
    def Even(x): return Atom('Even', x)                  # whether x is even
    def Odd(x): return Atom('Odd', x)                    # whether x is odd
    def Successor(x, y): return Atom('Successor', x, y)  # whether x's successor is y
    def Larger(x, y): return Atom('Larger', x, y)        # whether x is larger than y
    # Note: all objects are numbers, so we don't need to define Number as an
    # explicit predicate.
    # Note: pay attention to the order of arguments of Successor and Larger.
    # Populate |formulas| with the 6 laws above and set |query| to be the
    # query.
    # Hint: You might want to use the Equals predicate, defined in logic.py.  This
    # predicate is used to assert that two objects are the same.
    formulas = []
    query = None
    # BEGIN_YOUR_CODE (our solution is 30 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    #0
    rule1 = Equiv(And(Successor('$x', '$y'),Not(Equals('$x', '$y'))),Equals('$z', '$y'))
    AndList1 = [Successor('$x', '$z'),Not(Equals('$x', '$z')),Forall('$y',rule1)]
    formulas.append(Forall('$x', Exists('$z',AndList(AndList1))))
    #1
    IsEven = And(Even('$x'),Not(Odd('$x')))
    IsOdd = And(Odd('$x'),Not(Even('$x')))
    EitherEvenOrOdd = Forall('$x',Or(IsEven,IsOdd))
    formulas.append(EitherEvenOrOdd)
    #2
    IsEvenAndHasSuccessor = And(Even('$x'),Successor('$x','$y'))
    ImpliesOdd = Implies(IsEvenAndHasSuccessor,Odd('$y'))
    formulas.append(Forall('$x', Forall('$y', ImpliesOdd)))
    #3
    IsOddAndHasSuccessor = And(Odd('$x'),Successor('$x','$y'))
    ImpliesEven = Implies(IsOddAndHasSuccessor,Even('$y'))
    formulas.append(Forall('$x', Forall('$y', ImpliesEven)))
    #4
    HasSuccessor = Successor('$x','$y')
    formulas.append(Forall('$x', Forall('$y',Implies(HasSuccessor,Larger('$y', '$x')))))
    #5
    XlargerThanY = Larger('$x', '$y')
    YlargerThanZ = Larger('$y', '$z')
    formulas.append(Forall('$x', Forall('$y', Forall('$z', (Implies(And(XlargerThanY,YlargerThanZ),Larger('$x','$z')))))))
    # END_YOUR_CODE
    query = Forall('$x', Exists('$y', And(Even('$y'), Larger('$y', '$x'))))
    return (formulas, query)

############################################################
# Problem 6: semantic parsing (extra credit)
# Each of the following functions should return a GrammarRule.
# Look at createBaseEnglishGrammar() in nlparser.py to see what these rules should look like.
# For example, the rule for 'X is a Y' is:
#     GrammarRule('$Clause', ['$Name', 'is', 'a', '$Noun'],
#                 lambda args: Atom(args[1].title(), args[0].lower()))
# Note: args[0] corresponds to $Name and args[1] corresponds to $Noun.
# Note: by convention, .title() should be applied to all predicates (e.g., Cat).
# Note: by convention, .lower() should be applied to constant symbols (e.g., garfield).

from nlparser import GrammarRule
def RelatedNoun(x, y): return Atom('RelatedNoun', x, y)
def createRule1():
    # Return a GrammarRule for 'every $Noun $Verb some $Noun'
    # Note: universal quantification should be outside existential quantification.
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    return GrammarRule('$Clause', ['every', '$Noun', '$Verb', 'some', '$Noun'],
        #lambda args: print(args))
        lambda args: Forall('$x', Implies(Atom(args[0].title(), '$x'), Atom(args[1].title(), '$x'))))
    # END_YOUR_CODE

def createRule2():
    # Return a GrammarRule for 'there is some $Noun that every $Noun $Verb'
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    return GrammarRule('$Clause', ['there', 'is' ,'some', '$Noun', 'that', 'every', '$Noun', '$Verb'],
        lambda args: Forall('$x', Implies(Atom(args[0].title(), '$x'), Atom(args[1].title(), '$x'))))
    # END_YOUR_CODE

def createRule3():
    # Return a GrammarRule for 'if a $Noun $Verb a $Noun then the former $Verb the latter'
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    return GrammarRule('$Clause', ['if', 'a', '$Noun', '$Verb', 'a', '$Noun', 'then', 'the', 'former', '$Verb', 'the', 'latter'],
        lambda args: Forall('$x', Implies(Atom(args[0].title(), '$x'), Atom(args[1].title(), '$x'))))
    # END_YOUR_CODE
