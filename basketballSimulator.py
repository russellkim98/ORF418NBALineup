import numpy as np
import pandas as pd
import itertools

class Simulator():


    def __init__(self):
        # top 100 values(monte carlo)
        self.truth_100 = np.asarray([0 for i in range(100)])
        self.lineup_index_100 = 0
        # covariance matrix
        self.covariance = np.asarray([[0 for x in range(100)] for y in range(100)])
        # lineups
        self.lineups = 0
        #theta values
        self.truth1 = np.asarray([0 for i in range(15)])
        self.truth2 = np.asarray([[0]*15 for i in range(15)])
        self.truth3 = np.asarray([[[0]*15 for i in range(15)] for j in range(15)])
        self.ind1 = np.asarray([0 for i in range(15)])
        self.truth = np.asarray([0 for i in range(15)])
        self.belief1 = np.asarray([0 for i in range(15)])
        self.belief2 = np.asarray([[0]*15 for i in range(15)])
        self.belief3 = np.asarray([[[0]*15 for i in range(15)] for j in range(15)])
        
    def OldpopulateTruth(self, seed):
        np.random.seed(seed)
        firstLineup = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
        
        
        # Generate possible lineups
        #gen = itertools.permutations(firstLineup,len(firstLineup))
        #lineups = set()
        #for i in gen:
        #    if i not in lineups:
        #        lineups.add(i)
        #lineups = list(lineups)
        
        lineups = [firstLineup]*3003
        # Populate theta values
        for i in range(15):
            self.truth1[i] = np.random.randint(-20,20)
            print(self.truth1[i])
            for j in range(15):
                self.truth2[i][j] = np.random.randint(-5,5)
                for k in range(15):
                    self.truth3[i][j][k] = np.random.uniform(-2,2)
        # Create truth values
        truth = np.asarray([0 for i in range(3003)])
        for i in range(3003):
            tempX = np.asarray(lineups[i])
            #this one is vertical
            tempVertX = tempX.reshape(len(tempX),1)
            #this one is horizontal
            transposeX = tempX.reshape(1,len(tempX))
            x2 = np.matmul(tempVertX,transposeX)
            #populate the x3 indicator matrix
            x3 = np.asarray([[[0]*15 for i in range(15)] for j in range(15)])
            
            transposeX = [y for x in transposeX for y in x]
            
            for j in range(len(transposeX)):
                
                if transposeX[j] == 1:
                    for k in range(15):
                        for l in range(15):
                            x3[j][k][l] = 1
                            x3[k][j][l] = 1
                            x3[k][l][j] = 1
                
                    
            #sum 1
            sum1 = np.dot(transposeX,self.truth1)
            #sum 2
            sum2 = sum(sum(np.multiply(self.truth2,x2)))
            #sum 3
            sum3 = sum(sum(sum(np.multiply(self.truth3,x3))))
            truth[i] =  sum1+sum2+sum3
        self.truth = truth
        return(truth)
    
    def populateTruth(self,seed):
         # Populate theta values
        for i in range(15):
            self.truth1[i] = np.random.randint(-20,21)
            for j in range(15):
                self.truth2[i][j] = np.random.randint(-3,3)
                for k in range(15):
                    self.truth3[i][j][k] = np.random.randint(-2,3)
        truth = [0]*3003
        iter = 0
        self.lineups = list(itertools.combinations(range(15),10))
        for y in self.lineups:
            for i in range(10):
                truth[iter] += self.truth1[y[i]]
            for x in itertools.combinations(y, 2):
                truth[iter] += self.truth2[x[0]][x[1]]
            for x in itertools.combinations(y, 3):
                truth[iter] += self.truth3[x[0]][x[1]][x[2]]
                
            iter += 1
        return(np.asarray(truth))
    

    def monteCarlo(self,seed):
        truths = list()
        for i in range(5):
            truths.append(self.populateTruth(i))
        truthAverage = [sum(x)/5 for x in zip(truths[0],truths[1],truths[2],truths[3],truths[4])]
        
        indices = np.random.choice(3003,100)
        self.lineup_index_100 = indices
        self.truth_100 = [truthAverage[i] for i in self.lineup_index_100]
        
    def fillCov(self):
        def intersection(lst1, lst2):
            lst3 = [value for value in lst1 if value in lst2]
            return(lst3)
        var_player = 22.5
        for i in range(100):
            lineup_i = self.lineups[self.lineup_index_100[i]]
            for j in range(100):
                lineup_j = self.lineups[self.lineup_index_100[j]]
                self.covariance[i][j] = len(intersection(lineup_i,lineup_j))*var_player
     
    def simulate(self, seed):
        np.random.seed(seed)
        tempTruth = [np.random.normal(self.truth_100[i], scale = 8) for i in self.truth_100]
        rewards= [0,0,0,0]
        #EG policy initializea
        
        #Boltzmann policy initialize

        #Pure exploitation initialize

        #KG policy initialize

        
        def drawObservations(self, lineupChoice):
            return(tempTruth[lineupChoice]+ np.random.normal(0,scale = 10))
        
        for i in range(82):
            #get choices for all the policies and put in a list
            choices = []
            results = [drawObservations(choice) for choice in choices]
            # make policies learn 
            rewards = [rewards[i]+results[i] for i in rewards]
            
        return(rewards)
                
            
    

        
        

test = Simulator()
test.monteCarlo(5)
test.fillCov()








