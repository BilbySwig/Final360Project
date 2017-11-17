'''
Created on Nov 1, 2017

@author: Bilbo
'''
import time
from test.test_dis import outer
from random import randint
import random
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import igraph
from igraph import Graph

start=time.time()

c=1
b=4

population=[]
populationSize=15
numberOfTimeSteps=4000
currTimeStep=0
mutationRate=.001
iterations=50

strategies=["ALL_C", "ALL_D", "Tit4Tat", "CautiousTit", "Alternate", "Random"]
stratsSize=6

Pb=1      #this doesn't really matter rn but it's important to have bc i might end up testing lowered Pb's
Pn=.85
Pr=.017
relationships=np.zeros((populationSize,populationSize)).astype(int)

#unimportant for the replication but it made it easier to copy over some other code
class Individual:
    def __init__(self, strat, pay):
        if(strat=="rand"):
            self.strategy=strategies[randint(0, stratsSize-1)]
        else:    
            self.strategy=strat
        self.payoff=pay
        self.moves=[]
        self.alwaysC=False
        self.alwaysD=False
    def computeMove(self, opponent, iter):
        if(self.alwaysD==True):
            return "D"
        if(self.alwaysC==True):
            return "C"
        
        out="C"
        
        if(self.strategy=="ALL_D"):
            self.alwaysD=True
            out="D"
            
        if(self.strategy=="ALL_C"):
            self.alwaysC=True 
              
        if(self.strategy=="Tit4Tat"):  #starting c 
            if(iter>0):
                out=opponent.moves[iter-1] 
            
        if(self.strategy=="CautiousTit"):  #starting d
            out="D" 
            if(iter>0):
                out=opponent.moves[iter-1]  
                
        if(self.strategy=="Alternate"):
            if(iter>0):
                if(self.moves[iter-1]=="C"):
                    out="D" 
                
        if(self.strategy=="Random"):
            if(random.random()>.5):
                out="D"                      
        return out
    def addToPayoff(self, gamePay):
        self.payoff+=gamePay
    def mutate(self):
        self.strategy=strategies[randint(0, stratsSize-1)]
    def findFitness(self):
        global iterations
        return self.payoff    
    def clearMoves(self):
        if(len(self.moves)==iterations):
            for i in range(iterations):
                self.moves[i]="None"
        else:
            for i in range(iterations):
                self.moves.append("None")    
                
def initSim():
    relationships=np.zeros((populationSize,populationSize))
    for i in range(populationSize):
        guy=Individual("rand", 0)
        guy.clearMoves()
        population.append(guy)        

    for i in range(populationSize):
        for j in range(i, populationSize):
            if(j!=i):
                if(random.random()<= Pr):
                    relationships[i][j]=1
                    relationships[j][i]=1   
                        
                        
def generationRun():
    death= randint(0, populationSize-1) 
    mother=randint(0, populationSize-1)
    while(mother==death):
        mother=randint(0, populationSize-1)
    offspring=Individual("rand", 0)   #instead of rand, in the main program this will have a high probability of being the mother's strat
    relationships[:, death]=0
    relationships[death,:]=0
    
    
    for i in range(populationSize):
        relationships[death][i]=0
        if(relationships[mother][i]==1 and i!=death and random.random()<Pn):
            relationships[death][i]=1
            relationships[i][death]=1
        if( i!=death and relationships[mother][i]==0 and random.random()<Pr):
            relationships[death][i]=1
            relationships[i][death]=1
        if(i==mother and random.random()<Pb):
            relationships[death][i]=1  
            relationships[i][death]=1  
    
    population[death]=offspring


simRuns=500
degreeDistributionData=np.zeros((500, populationSize))
clusteringCoefData=np.zeros((500, populationSize+1))

for j in range(simRuns):
    initSim()
    for i in range(numberOfTimeSteps):
        generationRun()
    #this is a list of the %s of the pop with a degree over the degree specified by the index (made it 30 just to be safe)
    degrees=np.zeros(populationSize)
    clusco=np.zeros(populationSize+1)
    
    g=igraph.Graph().Adjacency(relationships.tolist(), mode=1)
    
    '''
    if(random.random()<.05):
        igraph.plot(g)
    '''
    
    deg=g.degree(np.arange(populationSize), mode=3, loops=True)#raw degree of each individual
    localCoefs=g.transitivity_local_undirected(np.arange(populationSize), mode="zero")#raw local clustering coefficient for each individual
    
    for i in range(populationSize):
        degrees[deg[i]]+=1#index= given degree, element= number with that degree
        clusco[int(localCoefs[i]*populationSize)]+=1
    popLeft=populationSize        
    for i in range(len(degreeDistributionData[0])):
        popLeft-=degrees[i]
        degrees[i]=popLeft/populationSize   
    popLeft=populationSize
    for i in range(len(clusteringCoefData[0])):
        popLeft-= clusco[i]
        clusco[i]=popLeft/populationSize               
    degreeDistributionData[j]=degrees   
    clusteringCoefData[j]=clusco

xaxisDD=np.arange(0, len(degreeDistributionData[0]), 1)
xaxisCC=np.arange(0, 1, 1/(populationSize+1))

fig, ax =plt.subplots(nrows=2, squeeze=False)    
avgDegDist=np.mean(degreeDistributionData, axis=0)
medDegDist=np.median(degreeDistributionData, axis=0)
DDhigh475=np.percentile(degreeDistributionData, 97.5, axis=0)
DDlow475=np.percentile(degreeDistributionData, 2.5, axis=0)

avgCC=np.mean(clusteringCoefData, axis=0)
CChigh475=np.percentile(clusteringCoefData,97.5, axis=0)
CClow475=np.percentile(clusteringCoefData,2.5, axis=0)

ax[0,0].set_xlim([0,32])     
ax[0,0].plot(xaxisDD, avgDegDist, color='black')
ax[0,0].plot(xaxisDD, medDegDist, color="green")
ax[0,0].fill_between(x=xaxisDD, y1=avgDegDist, y2=DDhigh475, color='blue')
ax[0,0].fill_between(x=xaxisDD, y1=avgDegDist, y2=DDlow475, color='blue')

ax[1,0].set_ylim([0,1])
ax[1,0].plot(xaxisCC, avgCC, color='red')
ax[1,0].fill_between(x=xaxisCC, y1=avgCC, y2=CChigh475, color='#FF6666')
ax[1,0].fill_between(x=xaxisCC, y1=avgCC, y2=CClow475, color='#FF6666')

print(time.time()-start)

plt.show()
        
        
        
        