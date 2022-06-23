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
    
    def __hash__(self):
        return self.name.__hash__()

    #def __cmp__(self, other):
    #    return object.__cmp__(self, other)

    def __eq__(self, rhs):
        return self.name==rhs.name

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
        # print("create NFA with start state", start.name)
        self.start = start
        if (end): #specifying an end state is optional
            self.end = end # start and end states
            end.is_end = True

    def to_DFA(self, log=False):
        statesDict ={} # name to state in DFA
        toNFADict = {} # state to set of corresponding states in NFA
        def makeName(stateset): # a function to generate names for states
            return "_".join(sorted([s.name for s in stateset]))
        def make_end_if(state):
            state.is_end = any(map(lambda s:s.is_end, toNFADict[state]))  #this state is an end state if any of the corresponding NFA states are end states
        dfstartset = self.start.getEpsilonClosure()
        startname = makeName(dfstartset)
        dfaStart = State(startname)
        statesDict[startname] = dfaStart
        toNFADict[dfaStart] = dfstartset        
        make_end_if(dfaStart)
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
                    toNFADict[newstate] = NFAendStates
                    # check if it's an end state
                    make_end_if(newstate)
                    # for nfs in NFAendStates:
                    #     if nfs.is_end:
                    #         statesDict[endStateName].is_end = True
                    #         break
                    queue.append(endStateName)
                endState = statesDict[endStateName]
                # add transition from current state
                dfaState.transitions[symbol] = endState
        return NFA(dfaStart)


    
    def has_border_condition(self, state, from_position, out_cond)->bool:
        '''
        note: function specific to distributed rpq evaluation
        state: which state we're askign about
        from_position: true is the query related to this node in "from_node" position, false if it's for the "to_node" position
        out_cond: true to check if we need to filter "outgoing" nodes, false if we're checking for "incoming" nodes
        returns true or false
        '''
        
        if (from_position==out_cond): #from, outnodes / to, innodes
            return self.hasInverseTransitionFrom(state)
        else: #from, innodes / to, outnodes
            return self.hasForwardsTransitionTo(state)


    def restrict_start(self, is_forwards):
        '''
        changes the automaton so that it accepts only sequences starting with "forwards" symbols (if forwards=True), or "inverse" symbols if forwards=False
        '''
        def keep(symbol):
            return not (is_forwards==symbol.startswith("^")) 
        new_start = State(self.start.name+"_r")
        if (self.start.epsilon):
            raise ValueError("make automaton deterministic before running restrict_start")
        new_start.transitions= {sym:state for sym,state in self.start.transitions.items() if keep(sym)}
        self.start=new_start

    def restrict_end(self, is_forwards):
        '''
        changes the automaton so that it accepts only sequences ending with "forwards" symbols (if forwards=True), or "inverse" symbols if forwards=False
        '''
        def keep(symbol):
            return not (is_forwards==symbol.startswith("^")) 
        endstates = list(filter(lambda s: s.is_end, self.allReachableStates()))
        new_endstates = {s:State(s.name+'_r') for s in endstates}  #dist of oldstate to duplicates
        for s in self.allReachableStates():
            newtrans = {}
            for sym,state in s.transitions.items(): #replace transitions to end_states
                if(state in endstates):
                    newtrans[sym] = (new_endstates[state] if keep(sym) else state)
                else:
                    newtrans[sym] = state
            s.transitions = newtrans
            if(s.epsilon):
                raise ValueError("make automaton deterministic before running restrict_start")
        for old,dup in new_endstates.items():
            dup.is_end=True
            old.is_end=False
            dup.transitions = old.transitions
            #if(self.start==old):
            #    self.start=dup

    def deepCopy(self):
        oldstateDict = {}
        newstateDict = {}
        for s in self.allReachableStates():
            oldstateDict[s.name] = s    # to be able to get a state by name
            newstateDict[s.name] = State(s.name) # new states with the same names
            newstateDict[s.name].is_end = s.is_end
        for newstate in newstateDict.values():
            #print("copying transitions for state", newstate.name)
            newstate.epsilon = list(map(lambda s: newstateDict[s.name], oldstateDict[newstate.name].epsilon)) #copy over the epsilon transitions
            newstate.transitions = { ks[0]: newstateDict[ks[1].name] for ks in oldstateDict[newstate.name].transitions.items()} #copy over the epsilon transitions
        newstart = newstateDict[self.start.name]
        #newend = newstateDict[self.end.name]
        return NFA(newstart)

    def renameStates(self):
        statelist=sorted(self.allReachableStates(), key=lambda q:q.name)
        counter=0
        for s in statelist:
            s.name = "S_"+str(counter)
            counter +=1

    #beware this does not add a state to the automaton!!
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

    def hasInverseTransitionFrom(self, state):
        return any(map(lambda k:k.startswith('^'),state.transitions.keys()))        

    def hasForwardsTransitionFrom(self, state):
        return any(map(lambda k:not k.startswith('^'),state.transitions.keys()))        

    def hasForwardsTransitionTo(self, state):
        return any(map(lambda k: not k.startswith('^'),[sym for sym, st in self.getInEdges(state)]))

    def hasInverseTransitionTo(self, state):
        return any(map(lambda k: not k.startswith('^'),[sym for sym, st in self.getInEdges(state)]))

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

    def reversed(self):
        """
        given an automaton that recognizes language L, produces a new automaton that recognizes 
        language LR, where a word is in LR if reverse(word) is in L
        """
        # print("reversing automaton:")
        # self.uglyprint()
        # print("----")
        # create new automaton with inverted transitions
        
        copies = dict()
        visited, queue = set(), [self.start]
        copies[self.start.name] = State(self.start.name+"R")
        copies[self.start.name].is_end = True #start state is end state of new automaton
        # print(copies[self.start.name].name, "becomes end state")
        startstates = set()
        while queue:
            state = queue.pop(0)
            if state.name not in visited:
                visited.add(state.name)
                statecopy = copies[state.name]
                if (state.is_end):
                    startstates.add(statecopy) #the end states become start states
                for sym, to_state in state.transitions.items():
                    to_copy = copies.setdefault(to_state.name, State(to_state.name+"R")) # new state
                    reverse_sym = sym[1:] if sym.startswith("^")  else "^"+sym
                    if(sym in to_copy.transitions.keys()):
                        raise ValueError("Automaton cannot be reversed: state "+ state.name+" has two incoming transitions with symbol "+ sym)
                    to_copy.transitions[reverse_sym] = statecopy #"to_state" transitions to "from state" in reversed automaton
                    if(to_state.name not in visited):
                        queue.append(to_state)
                for to_state in state.epsilon:
                    to_copy = copies.setdefault(to_state.name, State(state.name+"R")) # new state
                    to_copy.epsilon.append(statecopy) #"to_state" epsilon-transitions to "from state" in reversed automaton
                    if(to_state not in visited):
                        queue.append(to_state)

        #handle start state(s)
        if(len(startstates)==1): #just one endstate
            nfaCopy = NFA(startstates.pop()) #end state is now the start state
        else: #multiple end states
            #create new state, esilon-transitions to the start states
            newstart= State("newStart")
            newstart.epsilon = list(startstates)
            nfaCopy = NFA(newstart)
        return nfaCopy

    def to_regex(self): #returns a regex (AST) equivalent to this automaton or None if there isn't one (end states not reachable from start)
        nfaCopy = self.deepCopy()
        #print("nfa to regex begin transformation -------")
        #nfaCopy.uglyprint()
        #print("-------")
        s1 = nfaCopy.start
        if (nfaCopy.indegree(s1)>0 or s1.is_end): # if s1 has incoming edges we need to add a new start state with an epsilon-transition
            newstart = State("fakeStart")
            newstart.epsilon.append(s1)
            nfaCopy.start = newstart
        endstates = list(filter(lambda s: s.is_end, nfaCopy.allReachableStates()))
        if(len(endstates)>1 or nfaCopy.outdegree(endstates[0])>0):
            newend = State("fakeEnd")
            newend.is_end = True
            for oldend in endstates:
                oldend.is_end = False
                oldend.epsilon.append(newend)
        
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
        while (toremove):
            r = toremove.pop(0)
            #print('removing', r.name)
            # replace every A->R->B with A->B and concat , possibly loop with *
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
                    #print ('merging parallel edges from ', s.name, 'to', succ.name)
                    # merge all symbols as one regex with "or" symbols
                    allsymbols = s.getKeysToState(succ)
                    s.removeTransitions(allsymbols)
                    newlabel = NFATreeNode.makeBinaryAltTree(allsymbols)
                    s.transitions[newlabel] = succ

                            # print("new transition labeled", newtoken, "from", a.name, "to", b.name, "................")
        #print("nfa to regex end transformation :")
        #nfaCopy.uglyprint()
        regexes = list(nfaCopy.start.transitions.keys())
        if len(regexes)==1:
            return regexes[0]
        else:
            print("nfa to regex: NO REGEXES")
            for t in regexes:
                print(t)
            return None

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
                if(s2c not in nfaCopy.allReachableStates()):
                    if (log):
                        print("========== no path from ",s1.name, "to", s2.name, "skipping !!! ==")
                    continue
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
                
                the_regex = nfaCopy.to_regex() # new function!
                if (the_regex):
                    if (log):
                        print("regex:", the_regex)
                    if (the_regex.label.name != 'EPS'):
                        edge_id = s1.name+"_"+s2.name
                        if(edge_id in decomp):
                            print("error! twice the same edge in query decomposition!", edge_id)
                        decomp[edge_id] = (s1, s2, the_regex)
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

class Handler2: #used to convert a regex syntax tree to an automaton
    def __init__(self):
        self.handlers = {'CHAR': self.handle_char, 'CONCAT': self.handle_concat,
                         'ALT': self.handle_alt, 'STAR': self.handle_rep,
                         'PLUS': self.handle_rep, 'QMARK': self.handle_qmark}
        self.state_count = 0

    def handle_tree(self, node):
        if(len(node.children)==0):  #leaf
            return self.handle_char(node.label)
        elif (len(node.children)==1): #unary node
            nfa = self.handle_tree(node.children[0])
            return self.handlers[node.label.name](node.label, nfa)
        else: #binary node
            nfa0 = self.handle_tree(node.children[0])
            nfa1 = self.handle_tree(node.children[1])
            return self.handlers[node.label.name](node.label, nfa0, nfa1)

    def create_state(self):
        self.state_count += 1
        return State('s' + str(self.state_count))

    def handle_char(self, t):
        s0 = self.create_state()
        s1 = self.create_state()
        s0.transitions[t.value] = s1
        nfa = NFA(s0, s1)
        #nfa_stack.append(nfa)
        return nfa

    def handle_concat(self, t, nleft, nright):
        nleft.end.is_end = False
        nleft.end.epsilon.append(nright.start)
        nfa = NFA(nleft.start, nright.end)
        return nfa

    def handle_alt(self, t, nleft, nright):
        n1= nleft #just renanming because of reused code
        n2 = nright
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
            return n1
        else:
            s0 = self.create_state()
            s0.epsilon = [n1.start, n2.start]
            s3 = self.create_state()
            n1.end.epsilon.append(s3)
            n2.end.epsilon.append(s3)
            n1.end.is_end = False
            n2.end.is_end = False
            nfa = NFA(s0, s3)
            return nfa

    def handle_rep(self, t, n1):
        s0 = self.create_state()
        s1 = self.create_state()
        s0.epsilon = [n1.start]
        if t.name == 'STAR':
            s0.epsilon.append(s1)
        n1.end.epsilon.extend([s1, n1.start])
        n1.end.is_end = False
        nfa = NFA(s0, s1)
        return nfa

    def handle_qmark(self, t, n1):
        n1.start.epsilon.append(n1.end)
        return n1


class NFATreeNode:
    

    def __init__(self, token:Token):
        self.label = token
        self.children: List[NFATreeNode] = []

    @staticmethod
    def makeBinaryAltTree(nodes:List['NFATreeNode'])->'NFATreeNode':
        if(all(map(lambda n: n.isEpsilon(), nodes))):
            return nodes[0]
        # if(any(map(lambda n: n.isEpsilon(), nodes))):
        #     nonepsilon = filter(lambda n: not n.isEpsilon(), nodes)
        #     child = reduce(lambda left, right: NFATreeNode.makeBinaryAltNode(Token('ALT', '|'), left, right), nonepsilon)
        #     return NFATreeNode.makeUnaryNode(Token('QMARK', '?'), child)
        # else:
        return reduce(lambda left, right: NFATreeNode.makeBinaryAltNode(left, right), nodes)

    @staticmethod
    def makeBinaryConcatTree(nodes:List['NFATreeNode'])->'NFATreeNode':
        return reduce(lambda left, right: NFATreeNode.makeBinaryConcatNode(left, right), nodes)

    @staticmethod
    def makeBinaryConcatNode(left:'NFATreeNode', right:'NFATreeNode'):
        if (left.label.name == 'EPS'):
            return right
        if (right.label.name == 'EPS'):
            return left
        if(right.isStar() and right.children[0].sameAs(left)): #simplify aa* to a+
            return NFATreeNode.makeUnaryNode(Token('PLUS', '+'), left)
        n = NFATreeNode.makeBinaryNode(Token('CONCAT', '.'), left, right)
        return n


    @staticmethod
    def makeBinaryAltNode(left:'NFATreeNode', right:'NFATreeNode')->'NFATreeNode':
        if (left.label.name == 'EPS'):
            if(right.isNullable()):
                return right
            return NFATreeNode.makeQmarkNode(right)
        if (right.label.name == 'EPS'):
            if(left.isNullable()):
                return left
            return NFATreeNode.makeQmarkNode(left)
        #simplify pattern b|a+b => a*b
        if(right.isConcat() and right.children[0].isPlus() and left.sameAs(right.children[1])):
            right.children[0].label= Token('STAR', '*')
            return right
        if(left.isConcat() and left.children[0].isPlus() and right.sameAs(left.children[1])):
            left.children[0].label= Token('STAR', '*')
            return left
        n = NFATreeNode.makeBinaryNode(Token('ALT', '|'), left, right)
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
    def makeQmarkNode(child)->'NFATreeNode':
        if(child.isNullable()):
            return child
        elif(child.isPlus()):
            child.label.name='STAR'
            child.label.value='*'
            return child
        n = NFATreeNode(Token('QMARK', '?'))
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

    def sameAs(self,other):
        return (self.label.name==other.label.name and 
                self.label.value==other.label.value and 
                len(self.children)==len(other.children) and 
                all(map(lambda z:z[0].sameAs(z[1]), zip(self.children, other.children))))

    def addChild(self, node:'NFATreeNode'):
        self.children.append(node)

    def isLeaf(self):
        return len(self.children)==0

    def isEpsilon(self):
        return self.label.name=='EPS'

    def isForwardsLeaf(self):
        return (self.isLeaf() and not self.isInverseLeaf())

    def isInverseLeaf(self):
        return (self.isLeaf() and self.label.value.startswith("^"))

    def isAlt(self):
        return (self.label.name=='ALT')

    def isStar(self):
        return (self.label.name=='STAR')

    def isConcat(self):
        return (self.label.name=='CONCAT')

    def isQmark(self):
        return (self.label.name=='QMARK')

    def isPlus(self):
        return (self.label.name=='PLUS')
    
    def isNullable(self):
        if (self.isStar() or self.isQmark() or self.isEpsilon()):
            return True
        elif self.isAlt(): #if any of the children are nullable
            return any(map(lambda c: c.isNullable(), self.children))
        elif self.isConcat(): #if all the children are nullable
            return all(map(lambda c: c.isNullable(), self.children))
        else: #plus nodes, leaf nodes
            return False                

    ####### functions to check whether a regular expression matches strings beginning or ending with ^symbols (inverse edges in RPQ) ########
    def possibleExtr(self, start, forwards):
        '''
        true if a string matching the regex may start/end with a "forwards"/"inverse" token
        '''
        if (self.isStar() or self.isQmark() or self.isPlus()):
            return self.children[0].possibleExtr(start, forwards)
        elif self.isAlt(): #if any of the children can end forwards
            return any(map(lambda c: c.possibleExtr(start, forwards), self.children))
        elif self.isConcat(): #if the right child can end F or is nullable and left can end F
            endside = 0 if start else 1
            oppositeside = 1 if start else 0
            return self.children[endside].possibleExtr(start, forwards) or (self.children[endside].isNullable() and self.children[oppositeside].possibleExtr(start, forwards))
        elif self.isForwardsLeaf():
            return (forwards)      # true if we're trying to figure out if the word can start/end with forwards token
        elif self.isInverseLeaf():
            return (not forwards) # true if we're trying to figure out if the word can start/end with inverse token

        raise ValueError("unknown node type:"+ t.__str__)

    def canEndForwards(self):
        '''
        true if a string matching the regex may end with a "forwards" token
        '''
        return self.possibleExtr(False, True)

    def canEndInverse(self):
        '''
        true if a string matching the regex may end with an "inverse" token
        '''
        return self.possibleExtr(False, False)

    def canBeginForwards(self):
        '''
        true if a string matching the regex may begin with an "forwards" token
        '''
        return self.possibleExtr(True, True)

    def canBeginInverse(self):
        '''
        true if a string matching the regex may begin with an "inverse" token
        '''
        return self.possibleExtr(True, False)
    ####################################################

    def to_NFA(self):
        handler = Handler2()
        return handler.handle_tree(self)

    def to_DFA(self):
        dfa = self.to_NFA().to_DFA()
        dfa.renameStates()
        return dfa

    def splitLanguage(self, atstart): #rewrite this regular expression as the alternation of two expressions such that one can start/end only with forwards edges, the other with inverse edges.
        #easier to reason on the automaton
        r1 = self.deepCopy().to_DFA() 
        r2 = self.deepCopy().to_DFA()
        if(atstart):
            r1.restrict_start(True)
            r2.restrict_start(False)
        else:
            r1.restrict_end(True)
            r2.restrict_end(False)
        #back to regex AST
        return r1.to_regex(), r2.to_regex() #first one starts/ends only with forwards edges, second one starts/ends only with inverse


    def copy(self):
        node = NFATreeNode(self.label)
        node.children = self.children
        return node

    def deepCopy(self):
        node = NFATreeNode(self.label)
        node.children = [c.deepCopy() for c in self.children]
        return node

    #used to make a (sub-)expression non nullable. i.e. make epsilon no longer part of the language
    def star_to_plus(self): 
        if self.isStar(): #star to plus
            self.label=Token('PLUS', '+')
        if self.isQmark(): #remove question mark node altogether
            childlabel = self.children[0].label
            grandchildren = self.children[0].children
            self.label=childlabel
            self.children = grandchildren
        elif self.isAlt(): 
            for tc in self.children:
                tc.star_to_plus()
        elif self.isConcat():
            if(self.children[0].isNullable() and self.children[1].isNullable()):
                c10 = self.children[0]
                c20 = self.children[1]
                c1_nonnul = self.children[0].deepCopy()
                c1_nonnul.star_to_plus() #mutator
                c2_nonnul = self.children[1].deepCopy()
                c2_nonnul.star_to_plus()
                left = NFATreeNode.makeBinaryConcatNode(c10, c2_nonnul)
                right = NFATreeNode.makeBinaryConcatNode(c1_nonnul, c20)
                self.label = Token('ALT', '|')
                self.children = [left, right]

        #else = leaf nodes, plus nodes: nothing to be done        

    def non_eps(self): #same as above, except returns new object
        copy = self.deepCopy()
        copy.star_to_plus()
        return copy
        

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
                return "".join([str(c) if (c.isLeaf() or c.isStar() or c.isPlus() or c.isConcat()) else NFATreeNode.paren(str(c)) for c in self.children])
            elif (self.label.name == 'STAR'):
                c = self.children[0]
                return (str(c) if (c.isLeaf()) else NFATreeNode.paren(str(c))) + '*'
            elif (self.label.name == 'PLUS'):
                c = self.children[0]
                return (str(c) if (c.isLeaf()) else NFATreeNode.paren(str(c))) + '+'
            elif (self.label.name == 'QMARK'):
                c = self.children[0]
                return (str(c) if (c.isLeaf()) else NFATreeNode.paren(str(c))) + '?'

            else:
                return "NFATREENODE [" + self.label.name + "](" + ", ".join([str(c) for c in self.children]) + ")"

    def sparql_str(self):
        if (self.isLeaf()):
            return '<'+ self.label.value +'>' if self.label.value.startswith('http') else ('^<'+ self.label.value[1:] +'>' if self.label.value.startswith('^http') else self.label.value)
        else:
            if (self.label.name == 'ALT'):
                return "|".join([c.sparql_str() for c in self.children])
            elif (self.label.name == 'CONCAT'):
                return "/".join([c.sparql_str() if (c.isLeaf() or c.label.name=="STAR" or c.label.name=="PLUS" or c.label.name=="CONCAT") else NFATreeNode.paren(c.sparql_str()) for c in self.children])
            elif (self.label.name == 'STAR'):
                c = self.children[0]
                return (c.sparql_str() if (c.isLeaf()) else NFATreeNode.paren(c.sparql_str())) + '*'
            elif (self.label.name == 'PLUS'):
                c = self.children[0]
                return (c.sparql_str() if (c.isLeaf()) else NFATreeNode.paren(c.sparql_str())) + '+'
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
