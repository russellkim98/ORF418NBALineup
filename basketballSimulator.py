import numpy as np
import os
os.chdir('/Users/therealrussellkim/ORF418/ORF418NBALineup')
import itertools
import epsilonGreedy as eg
import Boltzmann as b
import UCB as UCB
import KGCB as KGCB
import matplotlib.pyplot as plt

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
        
        self.debugMean = 0
      
    # Matrix Multiplication method, not used anymore
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
        np.random.seed(seed)
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
    

    def monteCarlo(self):
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
     
                
              
    def simulate(self, ep, boltz,ucb):
        #tempTruth = [np.random.normal(self.truth_100[i], scale = 8) for i in range(100)]
        rewards = [[0 for j in range(82)] for k in range(4)]
        # Create prior mean
        pMean = [np.random.uniform(3, 18) for i in range(100)]
        priorMeans = [pMean for _ in range(4)]
        # Create prior covariance matrix 
        self.fillCov()
        priorCov = [self.covariance.copy() for i in range(4)]

        #generate the truth from prior
        tempTruth = np.random.multivariate_normal(pMean, self.covariance)
        
        #EG policy initializes
        epsilon = ep
        #Boltzmann policy initialize
        theta_b = boltz
        #UCB initializes
        theta_u = ucb
        #KG policy initialize
        precision = [1/22.5 for i in range(100)]
        num_selected = [1 for i in range(100)]

        
        def drawObservations(self, lineupChoice):
            return(tempTruth[lineupChoice]+ np.random.normal(0,scale = np.sqrt(22.5)))
        
        for i in range(0,82):
            choices = [0 for j in range(4)]
            #get choices for all the policies and put in a list
            choices[0] = eg.EpsilonGreedy(priorMeans[0],epsilon)
            choices[1] = b.Boltzmann(priorMeans[1],theta_b,i)
            choices[2],numselected = UCB.UCB(priorMeans[2],theta_u,i,num_selected)
            choices[3] = KGCB.kgcb(priorMeans[3],precision,priorCov[3],i)
            #print('EGreedy choice {}, Boltzmann choice{}, UCB {}, KG {}'.format(choices[0], choices[1], choices[2], choices[3]))
            
            results = [drawObservations(self,j) for j in choices]
            
            for j in range(4):
                
                rewards[j][i] = results[j]
            
            
             ## THIS STUFF IS FOR UPDATING EQUATIONS

            # max_value is the best estimated value of the KG
            # x is the argument that produces max_value

            # observe the outcome of the decision
            # w_k=mu_k+Z*SigmaW_k where SigmaW is standard deviation of the
            # error for each observation
            for j in range(4):
                w_k = results[j]
                cov_m = np.asarray(priorCov[j])
                x = choices[j]
                # updating equations for Normal-Normal model with covariance
                addscalar = (w_k - priorMeans[j][x])/(1/precision[x] + cov_m[x][x])
                # cov_m_x is the x-th column of the covariance matrix cov_m
                cov_m_x = np.array([row[x] for row in cov_m])
                priorMeans[j]= np.add(priorMeans[j], np.multiply(addscalar, cov_m_x))
                cov_m = np.subtract(cov_m, np.divide(np.outer(cov_m_x, cov_m_x), 1/precision[x] + cov_m[x][x]))
                priorCov[j] = cov_m

        return(rewards)
                
         
    

def cumulative(lst):
    rewards = [0 for i in range(82)]
    for i in range(82):
        if i == 1:
            rewards[i] = lst[i]
        else:
            rewards[i] = lst[i] + rewards[i-1]
    return(rewards)
            
        
if __name__ == '__main__':
    # All Policies 
    test = Simulator()
    test.monteCarlo()
    test.fillCov()
    results = test.simulate(0.4,3,5)
    x = np.arange(1, 83, 1)
    cumul = [cumulative(results[j]) for j in range(4)]
    colors = ['-b','-r','g+','p-']
    labels = ['EGreedy choice', 'Boltzmann choice', 'UCB', 'KG']
    for i in range(4):
        plt.plot(x,cumul[i],colors[i],label = labels[i] )
    plt.xlabel('Cumulative performance of policies(points)')
    plt.ylabel('Number of games simulated')
    plt.figure(0)
    plt.legend(loc = 'upper left')
    plt.title("All policies")
    
    
    # Test different values of the epsilon value 
    test1 = Simulator()
    test1.monteCarlo()
    test1.fillCov()
    eps = np.arange(0,1,0.1)
    epsResults = [test1.simulate(eps[i],3,5) for i in range(len(eps))]
    yEps = [sum(epsResults[i][0]) for i in range(len(epsResults))]
    for i in range(len(yEps)):
        plt.scatter(eps[i],yEps[i],label = eps[i] )
    plt.legend(loc = 'best')
    plt.figure(1)
    plt.title("Varying values of epsilon in the EG policy")
    
    # Test different values of the Boltzmann Theta constant
    test2 = Simulator()
    test2.monteCarlo()
    test2.fillCov()
    boltzmannC = np.arange(0,10,0.5)
    boltzResults = [test1.simulate(0.3,boltzmannC[i],5) for i in range(len(boltzmannC))]
    yBoltz = [sum(boltzResults[i][0]) for i in range(len(boltzResults))]
    plt.figure(2)
    plt.title("Varying values of theta in the Boltzmann policy") 
    for i in range(len(boltzmannC)):
        plt.scatter(boltzmannC[i],yBoltz[i],label = boltzmannC[i] )
    plt.legend(loc = 'best')
    
    # Test different values of the UCB theta constant 
    test3 = Simulator()
    test3.monteCarlo()
    test3.fillCov()
    UCB_theta = np.arange(0,10,0.5)
    UCB_results = [test1.simulate(0.3,boltzmannC[i],5) for i in range(len(boltzmannC))]
    yBoltz = [sum(boltzResults[i][0]) for i in range(len(boltzResults))]
    plt.figure(2)
    plt.title("Varying values of theta in the Boltzmann policy") 
    for i in range(len(boltzmannC)):
        plt.scatter(boltzmannC[i],yBoltz[i],label = boltzmannC[i] )
    plt.legend(loc = 'best')    
       
    
    
    
    
    
    
    
    
    
    