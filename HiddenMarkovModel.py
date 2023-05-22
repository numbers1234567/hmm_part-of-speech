class HiddenMarkovModel:
    def __init__(self, hmm_file):
        with open(hmm_file, "r", encoding="utf-8") as f:
            # Directly get the HMM data
            self.numObservations = int(f.readline())
            self.numStates = int(f.readline())
            self.obs2idx = {f.readline().strip():i for i in range(self.numObservations)}
            self.state2idx = {f.readline().strip():i for i in range(self.numStates)}
            
            self.a = []
            self.b = []
            for i in range(self.numStates):
                self.a.append([float(j) for j in f.readline().strip().split(" ")])
            for i in range(self.numStates):
                self.b.append([float(j) for j in f.readline().strip().split(" ")])
            self.pi = [float(i) for i in f.readline().strip().split(" ")]
        # Reverse operation
        self.idx2state = [0]*self.numStates
        self.idx2obs = [0]*self.numObservations
        for key,val in self.state2idx.items():
            self.idx2state[val] = key
        for key,val in self.obs2idx.items():
            self.idx2obs[val] = key
    
    def emissionProbI(self, obsI, staI):
        return self.b[staI][obsI]
    
    def emissionProb(self, observation, state):
        return self.emissionProbI([self.obs2idx[i] for i in observation], self.state2idx[state])
    
    def viterbi(self,O=None,S=None):
        assert O or S
        if not O: O = [None]*len(S)
        if not S: S = [None]*len(O)
        OI = [self.obs2idx[i] if i else -1 for i in O]
        SI = [self.state2idx[i] if i else -1 for i in S]
        
        # Base case
        deltaT = [self.pi[i]*self.emissionProbI(OI[0], i) for i in range(self.numStates)]
        mppT = [[i] for i in range(self.numStates)]
        if SI[0]>=0:
            deltaT = [0]*self.numStates
            deltaT[SI[0]]=1
        
        for T in range(2, len(O)+1): # len-1 inductive steps
            mppTNew = []
            deltaTNew = []
            for i in range(self.numStates):
                maxVal = 0 # value at argmax
                maxPrev = 0 # compute argmax
                # Unknown state at time T
                if not (SI[T-1] >= 0 and SI[T-1] != i):
                    emissionProb = self.emissionProbI(OI[T-1],i) if OI[T-1]>=0 else 1
                    for prev in range(self.numStates):
                        val = self.a[prev][i]*emissionProb*deltaT[prev]
                        if val > maxVal: 
                            maxVal = val
                            maxPrev = prev
                # Add delta_T(i), mpp_T(i)
                mppTNew.append(mppT[maxPrev].copy() + [i])
                deltaTNew.append(maxVal)
            mppT = mppTNew
            deltaT = deltaTNew.copy()
        # Final step, argmax deltaT[x]
        mppInts = max([(deltaT[i],mppT[i]) for i in range(self.numStates)])[-1]
        return [self.idx2state[i] for i in mppInts]
    
    def forwardBackward(self,O=None,S=None):
        assert O or S
        assert O or S
        if not O: O = [None]*len(S)
        if not S: S = [None]*len(O)
        OI = [self.obs2idx[i] if i else -1 for i in O]
        SI = [self.state2idx[i] if i else -1 for i in S]
        
        # Base case
        alphaT = [self.pi[i]*self.emissionProbI(OI[0], i) for i in range(self.numStates)]
        if SI[0]>=0:
            alphaT = [0]*self.numStates
            alphaT[SI[0]]=1
            
        for T in range(2, len(O)+1): # len-1 inductive steps
            alphaTNew = []
            for i in range(self.numStates):
                sumVal = 0 # alpha_T(i)
                # Unknown state at time T
                if not (SI[T-1] >= 0 and SI[T-1] != i):
                    emissionProb = self.emissionProbI(OI[T-1],i) if OI[T-1]>=0 else 1
                    for prev in range(self.numStates):
                        sumVal += self.a[prev][i]*emissionProb*alphaT[prev]
                alphaTNew.append(sumVal)
        return max(alphaTNew)