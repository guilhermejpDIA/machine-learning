import pandas as pd
import numpy as np

class MarkovChain():
    def __init__(self):
        self.one_step_matrix = None
        self.state_space = None
        self.proba_range = None
        self.dictionary = None

    def encode(self, x):
        res = []
        un = np.unique(x)
        for i in x:
            for j in range(len(un)):
                if i==un[j]:
                    res.append(j)
        self.dictionary = un
        return res

    def decode(self, x):
        res = []
        for i in x:
            for j in range(len(self.dictionary)):
                if i==j:
                    res.append(self.dictionary[j])
        return res

    def get_proba_range(self, A):
        B={}
        for i in range(len(A)):
            B[i] = {}
            s = 0.0
            for j in range(len(A[0])):
                B[i][j] = [s, s+A[i][j]]
                s = s+A[i][j]
        self.dictionary = list(range(len(A)))
        return B

    def set_one_step_matrix(self, A):
        self.one_step_matrix = A
        self.state_space = list(range(len(A)))
        self.proba_range = self.get_proba_range(A)

    def get_one_step_matrix(self, seq):
        x = self.encode(seq)
        size = len(np.unique(x))

        d = {}
        n = {}
        final = np.zeros([size,size])
        for i in range(len(x)-1):
            state = (x[i], x[i+1])
            if state in d:
                d[state]+=1
            else:
                d[state]=1
            if x[i] in n:
                n[x[i]]+=1
            else:
                n[x[i]]=1


        for i, j in d.items():
            (w,p) = i
            if w in n:
                final[w][p]= j/n[w]

        self.one_step_matrix = final
        self.state_space = list(range(size))
        self.proba_range = self.get_proba_range(final)

        return final

    def get_n_step_matrix(self, n, seq = None):
        if self.one_step_matrix is None:
            if seq is not None:
                self.one_step_matrix = self.get_one_step_matrix((seq))
            else:
                print('Fit the one step matrix first! Or pass the sequence as argument seq.')
                return
        return np.linalg.matrix_power(self.one_step_matrix,n)

    def generate_sequence(self, n, first_state = None):
        if self.one_step_matrix is None:
            print('Fit the model first!\n')
            return
        else:
            if first_state is None:
                res = [np.random.choice(self.state_space)]
            else:
                first_state = np.where(m.dictionary==first_state)[0][0]
                res = [first_state]
            for i in range(n-1):
                proba = np.random.uniform()
                for state in self.state_space:
                    if self.proba_range[res[i]][state][0]<=proba<self.proba_range[res[i]][state][1]:
                        res.append(state)
                        break
        return self.decode(res)

    def fit(self, seq):
        self.get_one_step_matrix(seq)
