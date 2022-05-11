# regex engine in Python
# parser and classes
# xiayun.sun@gmail.com
# 06-JUL-2013

import pdb
from functools import reduce
from typing import List

class Token:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return self.name + ":" + self.value


class Lexer:
    def __init__(self, pattern):
        self.source = pattern
        self.symbols = {'(': 'LEFT_PAREN', ')': 'RIGHT_PAREN', '*': 'STAR', '|': 'ALT', '\x08': 'CONCAT', '+': 'PLUS',
                        '?': 'QMARK', '<': 'LT',
                        '>': 'GT'}  # extended with <>: this way <blabla> is taken to be a single char token
        self.current = 0
        self.length = len(self.source)

    def get_token(self):
        if self.current < self.length:
            c = self.source[self.current]
            self.current += 1
            if c not in self.symbols.keys():  # CHAR
                token = Token('CHAR', c)
            elif c == '<':
                # get a string as one token
                c = self.source[self.current]
                self.current += 1
                newtoken = ""
                while (c != '>') and (
                        self.current < self.length):  # until we hit the end of pattern or else the end token symbol
                    newtoken += c
                    c = self.source[self.current]
                    self.current += 1
                    # TODO: check that token was well formed (ended with '>')
                token = Token('CHAR', newtoken)
            else:
                token = Token(self.symbols[c], c)
            return token
        else:
            return Token('NONE', '')


class ParseError(Exception): pass


'''
Grammar for regex:

regex = exp $

exp      = term [|] exp      {push '|'}
         | term
         |                   empty?

term     = factor term       chain {add \x08}
         | factor

factor   = primary [*]       star {push '*'}
         | primary [+]       plus {push '+'}
         | primary [?]       optional {push '?'}
         | primary

primary  = \( exp \)
         | char              literal {push char}
         | <<char*>>         literal {push <<chars>>}
'''


class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.tokens = []
        self.lookahead = self.lexer.get_token()

    def consume(self, name):
        if self.lookahead.name == name:
            self.lookahead = self.lexer.get_token()
        elif self.lookahead.name != name:
            raise ParseError

    def parse(self):
        self.exp()
        return self.tokens

    def exp(self):
        self.term()
        if self.lookahead.name == 'ALT':
            t = self.lookahead
            self.consume('ALT')
            self.exp()
            self.tokens.append(t)

    def term(self):
        self.factor()
        if self.lookahead.value not in ')|':
            self.term()
            self.tokens.append(Token('CONCAT', '\x08'))

    def factor(self):
        self.primary()
        if self.lookahead.name in ['STAR', 'PLUS', 'QMARK']:
            self.tokens.append(self.lookahead)
            self.consume(self.lookahead.name)

    def primary(self):
        if self.lookahead.name == 'LEFT_PAREN':
            self.consume('LEFT_PAREN')
            self.exp()
            self.consume('RIGHT_PAREN')
        elif self.lookahead.name == 'CHAR':
            self.tokens.append(self.lookahead)
            self.consume('CHAR')


class State:
    def __init__(self, name):
        self.epsilon = []  # epsilon-closure
        self.transitions = {}  # char : state
        self.name = name
        self.is_end = False

    def hasTransitionTo(self, state2):
        return (state2 in self.transitions.values()) or (state2 in self.epsilon)        

    def getSymbols(self): #set of symbols such that this state has outgoing transitions with these symbols
        return list(self.transitions.keys())

    def getKeysToState(self,state):
        return [k for k in self.transitions.keys() if (self.transitions[k]==state)] + (['EPS'] if (state in self.epsilon) else [])

    def getLoops(self):        
        return self.getKeysToState(self) 

    def hasLoops(self, state):
        return len(self.getLoops())>0

    def getEpsilonClosure(self):
        closure = set()
        queue = [self]
        while queue:
            vertex = queue.pop(0)
            if vertex not in closure:
                closure.add(vertex)
                queue.extend([s for s in vertex.epsilon if s not in closure])                    
        return closure #all visited states     

    def removeTransitions(self,listTokens):
        for t in listTokens:        
            self.removeTransition(t)

    def removeTransition(self,t):
        if (t!='EPS'):
            self.transitions.pop(t)

    def removeEpsilonTransitions(self, statelist):
        for s in statelist:
            if(s in self.epsilon):
                self.epsilon.remove(s)        
            else:
                print("info: ", s.name, "not found in", self.name,"epsilon transitions")

    
class NFA:
    def __init__(self, start, end=None):
        self.start = start
        if (end): #specifying an end state is optional
            self.end = end # start and end states
            end.is_end = True

    def toDFA(self, log=False):
        statesDict ={} # name to state in DFA
        toNFADict = {} # state to set of corresponding states in NFA
        def makeName(stateset): # a function to generate names for states
            return "_".join(sorted([s.name for s in stateset]))
        dfstartset = self.start.getEpsilonClosure()
        startname = makeName(dfstartset)
        dfaStart = State(startname)
        statesDict[startname] = dfaStart
        toNFADict[dfaStart] = dfstartset        
        queue = [startname] # queue is name of dfa state in case we hit the same stateset again
        visited = set()
        while(queue):
            dfaStateName = queue.pop(0)
            if dfaStateName in visited:
                continue
            visited.add(dfaStateName)
            # get DFA state object, and corresponding set of NFA State objects
            dfaState = statesDict[dfaStateName]
            nfaStates = toNFADict[dfaState]
            # get list of symbols for outgoing transitions (from NFA)
            symbols = list(reduce(lambda a, b,: a + b, [s.getSymbols() for s in nfaStates]))
            for symbol in symbols:
                NFAendStates = set()
                for s in nfaStates:
                    if symbol in s.transitions:
                        NFAendStates.update(s.transitions[symbol].getEpsilonClosure())
                endStateName = makeName(NFAendStates)
                if ((endStateName not in queue) and (endStateName not in visited)):
                    # create new "set" state for DFA
                    if(log):
                        print("==new state:", endStateName)
                    newstate = State(endStateName)
                    statesDict[endStateName] = newstate                            
                    # check if it's an end state
                    for nfs in NFAendStates:
                        if nfs.is_end:
                            statesDict[endStateName].is_end = True
                            break
                    toNFADict[newstate] = NFAendStates
                    queue.append(endStateName)
                endState = statesDict[endStateName]
                # add transition from current state
                dfaState.transitions[symbol] = endState
        return NFA(dfaStart)

    def deepCopy(self):
        oldstateDict = {}
        newstateDict = {}
        for s in self.allReachableStates():
            oldstateDict[s.name] = s    # to be able to get a state by name
            newstateDict[s.name] = State(s.name) # new states with the same names
        for newstate in newstateDict.values():
            #print("copying transitions for state", newstate.name)
            newstate.epsilon = list(map(lambda s: newstateDict[s.name], oldstateDict[newstate.name].epsilon)) #copy over the epsilon transitions
            newstate.transitions = { ks[0]: newstateDict[ks[1].name] for ks in oldstateDict[newstate.name].transitions.items()} #copy over the epsilon transitions
        newstart = newstateDict[self.start.name]
        #newend = newstateDict[self.end.name]
        return NFA(newstart)

    def renameStates(self):
        statelist=self.allReachableStates()
        counter=0
        for s in statelist:
            s.name = "S_"+str(counter)
            counter +=1

    
    def addstate(self, state, state_set): # add state + recursively add epsilon transitions
        if state in state_set:
            return
        state_set.add(state)
        for eps in state.epsilon:
            self.addstate(eps, state_set)
    
    def getPredecessors(self, state):
        return list(filter(lambda s:s.hasTransitionTo(state), self.allReachableStates()))

    def getSuccessors(self, state):
        return list(state.transitions.values()) + state.epsilon

    def getInEdges(self, state):
        alledges = []
        for pred in self.getPredecessors(state):
            if (pred == state):
                continue  # skip self loops
            trans = pred.transitions
            edges = [(symbol, pred) for symbol in trans if (trans[symbol] == state)]
            if (state in pred.epsilon):
                edges.append(('EPS', pred))
            alledges.extend(edges)
        return alledges

    def getOutEdges(self, state):        
        edges1 = [(k, state.transitions[k]) for k in state.transitions if state.transitions[k]!= state] 
        return edges1 + [('EPS', s) for s in state.epsilon]

    def indegree(self, state):
        return len(self.getPredecessors(state))

    def outdegree(self, state):
        return len(self.getSuccessors(state))

    def decomposePaths(self, log=False):   #TODO: we assume there's only one end state
        statelist = self.allReachableStates()
        decomp = {}
        counter = 0
        for s1 in statelist:
            for s2 in statelist:
                if(log):
                    print ("----------------", s1.name, "to", s2.name)
                nfaCopy = self.deepCopy()                
                for s in nfaCopy.allReachableStates():
                    s.is_end = False #existing end states are no longer end states
                    # get copies of states s1 and s2
                    if (s.name == s1.name):
                        s1c = s
                    if (s.name == s2.name):
                        s2c = s
                if (nfaCopy.indegree(s1c)==0): # make s1 the new start state
                    nfaCopy.start = s1c
                else : # if s1 has incoming edges we need to add a new start state with an epsilon-transition
                    newstart = State("fakeStart")
                    newstart.epsilon.append(s1c)
                    nfaCopy.start = newstart
        
                if (nfaCopy.outdegree(s2c)==0): # make s2 the new end state
                    nfaCopy.end = s2c
                    s2c.is_end = True
                else : # if s2 has outgoing edges we need to add a new separate end state with an epsilon-transition
                    newend = State("fakeEnd")
                    newend.is_end = True
                    nfaCopy.end = newend
                    s2c.epsilon.append(newend)
                if(log):
                    nfaCopy.uglyprint() # ------------- what it looks like
                
                # merge parallel edges (and get list of states to remove while we're at it)
                
                #for s in nfaCopy.allReachableStates():
                #    for succ in nfaCopy.getSuccessors(s):
                #        # merge all symbols as one regex with "or" symbols
                #        allsymbols = s.getKeysToState(succ)
                #        s.removeTransitions(allsymbols)
                #        s.transitions[toregex(allsymbols)] = succ #TODO: if done with an NFA there is a risk of replacing an epsilon transition with a transition saying 'EPS'
                #    if not(nfaCopy.start==s or s.is_end):
                #        toremove.append(s)

                toremove = []
                for s in nfaCopy.allReachableStates():
                    tklist = list(s.transitions.keys())
                    for tk in tklist:
                        s.transitions[NFATreeNode(Token('CHAR', tk))] = s.transitions[tk]
                        s.transitions.pop(tk)

                    if (len(s.epsilon) > 1):
                        print("WARNING: more than 1 epsilon transition in ", s.name)
                    elif len(s.epsilon) == 1:
                        s.transitions[NFATreeNode(Token('EPS', 'eps'))] = s.epsilon[0]
                        s.epsilon = []
                    if not (nfaCopy.start == s or s.is_end):
                        toremove.append(s)
                # now do state removal
                if (log):
                    print("to remove:", [s.name for s in toremove])
                while (toremove):
                    r = toremove.pop(0)
                    # replace every A->R->B with A->B and concat , possibly loop with *
                    if (log):
                        print("removing:", r.name)
                    inedges = nfaCopy.getInEdges(r)
                    outedges = nfaCopy.getOutEdges(r)
                    loops = r.getLoops()
                    # print ("predecessors:", [p.name for p in preds], "succs:", [p.name for p in sucs])
                    for syma, a in inedges:
                        a.removeTransition(syma)
                        for symb, b in outedges:
                            nodelist = [syma]
                            if (loops):
                                loopsnode = NFATreeNode.makeBinaryAltTree(loops)
                                loopsnode = NFATreeNode.makeUnaryNode(Token('STAR', '*'), loopsnode)
                                nodelist.append(loopsnode)
                            nodelist.append(symb)
                            newtoken = NFATreeNode.makeBinaryConcatTree(nodelist)
                            a.transitions[newtoken] = b
                    # merge all parallel edges
                    for s in nfaCopy.allReachableStates():
                        for succ in nfaCopy.getSuccessors(s):
                            # merge all symbols as one regex with "or" symbols
                            allsymbols = s.getKeysToState(succ)
                            s.removeTransitions(allsymbols)
                            newlabel = NFATreeNode.makeBinaryAltTree(allsymbols)
                            s.transitions[newlabel] = succ

                            # print("new transition labeled", newtoken, "from", a.name, "to", b.name, "................")
                    if (log):
                        nfaCopy.uglyprint()
                counter += 1
                regexes = list(nfaCopy.start.transitions.keys())
                if len(regexes) > 0:
                    the_regex = regexes[0]
                    if len(regexes) > 1 and log:
                        print("Error? regexes >1 !!!:", regexes)
                    if (log):
                        print("regex:", the_regex)
                    if (the_regex.label.name != 'EPS'):
                        decomp[s1.name+"_"+s2.name] = (s1, s2, the_regex)
        return decomp
        # nfaCopy.uglyprint()

        # Manual decomposition of the regex using states from the original regex NFA
        '''
        re_expanded = {"r1": (states[0], states[1], "<a><b>*"),
                       "r2": (states[0], states[2], "<a><b>*<c>"),
                       "r3": (states[0], states[3], "<a><b>*<c><d>"),
                       "r4": (states[1], states[1], "<b>+"),
                       "r5": (states[1], states[2], "<b>*<c>"),
                       "r6": (states[1], states[3], "<b>*<c><d>"),
                       "r7": (states[2], states[3], "<d>")}
        '''

    def uglyprint(self):
        visited, queue = set(), [self.start]
        while queue:
            vertex = queue.pop(0)
            print ("v: "+ vertex.name + (" [start]" if vertex == self.start else "") + (" [end]" if vertex.is_end else ""))
            if vertex not in visited:
                visited.add(vertex)
                transitions = set()
                for k in vertex.transitions.keys():
                    trans = vertex.transitions[k]
                    print("-" + str(k) + '->' + trans.name)
                    if trans not in visited:
                        transitions.add(trans)

                transitions.update(vertex.epsilon)
                for eps in vertex.epsilon:
                    print("-eps->" + eps.name)

                queue.extend([s for s in transitions if (s not in visited) and (s not in queue)])

        # return len(visited) #set of graph nodes in terminal nodes of product automaton; list of visited nodes

    def pretty_print(self):
        '''
        print using Graphviz
        '''
        pass

    def match(self, s):
        current_states = set()
        self.addstate(self.start, current_states)

        for c in s:
            next_states = set()
            for state in current_states:
                if c in state.transitions.keys():
                    trans_state = state.transitions[c]
                    self.addstate(trans_state, next_states)

            current_states = next_states

        for s in current_states:
            if s.is_end:
                return True
        return False

    def size(self):
        return len(self.allReachableStates())

    def allReachableStates(self):
        visited, queue = set(), [self.start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                transitions = list(vertex.transitions.values())+vertex.epsilon
                #transitions.extend(vertex.epsilon)
                queue.extend([s for s in transitions if s not in visited])
                    
        return visited #all visited states        
        
        
class Handler:
    def __init__(self):
        self.handlers = {'CHAR': self.handle_char, 'CONCAT': self.handle_concat,
                         'ALT': self.handle_alt, 'STAR': self.handle_rep,
                         'PLUS': self.handle_rep, 'QMARK': self.handle_qmark}
        self.state_count = 0

    def create_state(self):
        self.state_count += 1
        return State('s' + str(self.state_count))

    def handle_char(self, t, nfa_stack):
        s0 = self.create_state()
        s1 = self.create_state()
        s0.transitions[t.value] = s1
        nfa = NFA(s0, s1)
        nfa_stack.append(nfa)

    def handle_concat(self, t, nfa_stack):
        n2 = nfa_stack.pop()
        n1 = nfa_stack.pop()

        n1.end.is_end = False
        n1.end.epsilon.append(n2.start)
        nfa = NFA(n1.start, n2.end)
        nfa_stack.append(nfa)

    def handle_alt(self, t, nfa_stack):
        n2 = nfa_stack.pop()
        n1 = nfa_stack.pop()
        ################################ debug
        # print "alternating! n1: len=", n1.size()
        # n1.uglyprint()
        # print "-------------- n2: len=", n2.size()
        # n2.uglyprint()
        ############################3
        # special case where it's just (a|b) or (a?|b?) [nfa sizes ==2]
        if (n1.size() == 2 and n2.size() == 2 and len(
                n2.end.transitions) == 0):  # just a possibly overkill check that n2 end state has no outgoing transitions
            # copy transitions of n2 to n1
            for sym in n2.start.transitions.keys():
                n1.start.transitions[sym] = n1.end  # all transitions necessarily go to end state
            if n2.end in n2.start.epsilon:
                n1.start.epsilon.append(n1.end)
            nfa_stack.append(n1)
        else:
            s0 = self.create_state()
            s0.epsilon = [n1.start, n2.start]
            s3 = self.create_state()
            n1.end.epsilon.append(s3)
            n2.end.epsilon.append(s3)
            n1.end.is_end = False
            n2.end.is_end = False
            nfa = NFA(s0, s3)
            nfa_stack.append(nfa)

    def handle_rep(self, t, nfa_stack):
        n1 = nfa_stack.pop()
        s0 = self.create_state()
        s1 = self.create_state()
        s0.epsilon = [n1.start]
        if t.name == 'STAR':
            s0.epsilon.append(s1)
        n1.end.epsilon.extend([s1, n1.start])
        n1.end.is_end = False
        nfa = NFA(s0, s1)
        nfa_stack.append(nfa)

    def handle_qmark(self, t, nfa_stack):
        n1 = nfa_stack.pop()
        n1.start.epsilon.append(n1.end)
        nfa_stack.append(n1)


class NFATreeNode:
    def __init__(self, token:Token):
        self.label = token
        self.children: List[NFATreeNode] = []

    @staticmethod
    def makeBinaryAltTree(nodes:List['NFATreeNode'])->'NFATreeNode':
        return reduce(lambda left, right: NFATreeNode.makeBinaryNode(Token('ALT', '|'), left, right), nodes)

    @staticmethod
    def makeBinaryConcatTree(nodes:List['NFATreeNode'])->'NFATreeNode':
        return reduce(lambda left, right: NFATreeNode.makeBinaryConcatNode(left, right), nodes)

    @staticmethod
    def makeBinaryConcatNode(left:'NFATreeNode', right:'NFATreeNode'):
        if (left.label.name == 'EPS'):
            return right
        if (right.label.name == 'EPS'):
            return left
        n = NFATreeNode(Token('CONCAT', '.'))
        n.addChild(left)
        n.addChild(right)
        return n

    @staticmethod
    def makeBinaryNode(token:Token, left:'NFATreeNode', right:'NFATreeNode')->'NFATreeNode':
        n = NFATreeNode(token)
        n.addChild(left)
        n.addChild(right)
        return n

    @staticmethod
    def makeUnaryNode(token:Token, child)->'NFATreeNode':
        n = NFATreeNode(token)
        n.addChild(child)
        return n

    @staticmethod
    def makeLeafNode(token:Token)->'NFATreeNode':
        n = NFATreeNode(token)
        #n.value = value
        return n

    @staticmethod
    def paren(s): #helper function to parenthesize an expression
            return "("+ s + ")"

    def addChild(self, node:'NFATreeNode'):
        self.children.append(node)

    def isLeaf(self):
        return len(self.children)==0

    def print(self):
        self.printIndented(0)

    def printIndented(self, depth):
        print(" " * depth * 2 + self.label.name)
        if (self.isLeaf()):
            print(" " * depth * 2 + self.label.value)
        else:
            for c in self.children:
                c.printIndented(depth + 1)

    def __str__(self):
        if (self.isLeaf()):
            return "<" + self.label.value + ">"
        else:
            if (self.label.name == 'ALT'):
                return "|".join([str(c) for c in self.children])
            elif (self.label.name == 'CONCAT'):
                return "".join([str(c) if (c.isLeaf() or c.label.name=="STAR" or c.label.name=="CONCAT") else NFATreeNode.paren(str(c)) for c in self.children])
            elif (self.label.name == 'STAR'):
                c = self.children[0]
                return (str(c) if (c.isLeaf()) else NFATreeNode.paren(str(c))) + '*'
            else:
                return "NFATREENODE [" + self.label.name + "](" + ", ".join([str(c) for c in self.children]) + ")"

    def sparql_str(self):
        if (self.isLeaf()):
            return self.label.value
        else:
            if (self.label.name == 'ALT'):
                return "|".join([c.sparql_str() for c in self.children])
            elif (self.label.name == 'CONCAT'):
                return "/".join([c.sparql_str() if (c.isLeaf() or c.label.name=="STAR"or c.label.name=="CONCAT") else NFATreeNode.paren(c.sparql_str()) for c in self.children])
            elif (self.label.name == 'STAR'):
                c = self.children[0]
                return (c.sparql_str() if (c.isLeaf()) else NFATreeNode.paren(c.sparql_str())) + '*'
            else:
                return "NFATREENODE [" + self.label.name + "](" + ", ".join([sparql_str(c) for c in self.children]) + ")"

# from postfix notation, build tree using stack
class HandlerTree:
    def __init__(self):
        self.handlers = {'CHAR': self.handle_char, 'CONCAT': self.handle_concat,
                         'ALT': self.handle_alt, 'STAR': self.handle_rep,
                         'PLUS': self.handle_rep, 'QMARK': self.handle_qmark}
        self.state_count = 0

    def handle_char(self, t, nfa_stack):
        nfa_stack.append(NFATreeNode.makeLeafNode(t))

    def handle_binary(self, t, nfa_stack):
        right = nfa_stack.pop()
        left = nfa_stack.pop()
        factor = NFATreeNode.makeBinaryNode(t, left, right)
        nfa_stack.append(factor)

    def handle_unary(self, t, nfa_stack):
        n1 = nfa_stack.pop()
        factor = NFATreeNode.makeUnaryNode(t, n1)
        nfa_stack.append(factor)

    def handle_concat(self, t, nfa_stack):
        self.handle_binary(t, nfa_stack)

    def handle_alt(self, t, nfa_stack):
        self.handle_binary(t, nfa_stack)

    def handle_rep(self, t, nfa_stack):
        self.handle_unary(t, nfa_stack)
        # in this context it works the same as for alt (binary op)

    def handle_qmark(self, t, nfa_stack):  # unary op
        self.handle_unary(self, t, nfa_stack)
