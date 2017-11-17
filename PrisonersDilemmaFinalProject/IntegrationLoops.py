'''
Created on Nov 1, 2017

@author: Bilbo
'''
import time
from random import randint
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import igraph
from igraph import Graph
from scipy import stats

c=1
b=4

startTime=time.time()

population=[]
populationSize=100
numberOfTimeSteps=2000
currTimeStep=0
mutationRate=.001
iterations=50


strategies=["ALL_C", "ALL_D", "Tit4Tat", "CautiousTit", "Alternate", "Random"]
stratsSize=6

data=np.zeros((stratsSize,numberOfTimeSteps))

simulationRuns=5
degreeDistributionData=np.zeros((simulationRuns, populationSize))
clusteringCoefData=np.zeros((500, populationSize+1))
howManyDie=10

Pb=1      #this doesn't really matter rn but it's important to have bc i might end up testing lowered Pb's
Pn=.85
Pr=.017
relationships=[[0 for i in range(populationSize)]for j in range(populationSize)] 

class Individual:
    def __init__(self, strat, pay, ind):
        if(strat=="rand"):
            self.strategy=strategies[randint(0, stratsSize-1)]
        else:    
            self.strategy=strat
        self.payoff=pay
        self.moves=[]
        self.alwaysC=False
        self.alwaysD=False
        self.index=ind
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
    def getPopIndex(self):
        return self.index             
                             
def initSim():
    global population
    population=[]
    relationships=[[0 for i in range(populationSize)]for j in range(populationSize)]
    for i in range(populationSize):
        guy=Individual("rand", 0, i)
        guy.clearMoves()
        population.append(guy)        
        
    for i in range(populationSize):
        for j in range(i, populationSize):
            if(j!=i):
                if(random.random()<= Pr):
                    relationships[i][j]=1
                    relationships[j][i]=1   
                        
def runSim():
    initSim()
    for i in range(numberOfTimeSteps):
        runTimeStep()
    return     

def getOther(individual, selfIndex):
    ppl=[]
    for i in range(populationSize):
        if(relationships[selfIndex][i]==1):
            ppl.append(i)
    if(random.random()<.1 or len(ppl)==0):
        otherIndividual=population[randint(0, populationSize-1)]
        while(otherIndividual==individual):
            randIndex=randint(0, populationSize-1)    
            otherIndividual=population[randIndex]
        return otherIndividual
    else:   
        other=randint(0, len(ppl)-1)
        while(relationships[selfIndex][ppl[other]]==0):
            other=randint(0, len(ppl)-1)
        return population[ppl[other]]        

def playGame(ind1, ind2):
    if(currTimeStep!=0):
        ind1.clearMoves()
        ind2.clearMoves()
    global iterations
    payoff2=0 #this is to keep track of the 2nd person's payoff for relationship purposes bc it isn't being saved here
    for i in range(iterations):
        move1=ind1.computeMove(ind2, i)
        move2=ind2.computeMove(ind1, i)
        ind1.moves[i]=move1
        ind2.moves[i]=move2
        if(move1=="C"):
            if(move2=="C"):
                ind1.addToPayoff(b-c)
                payoff2+=b-c
            else:
                ind1.addToPayoff(-c)
                payoff2+=b
        else:
            if(move2=="C"):
                ind1.addToPayoff(b)
                payoff2-=c
            else:
                ind1.addToPayoff(-c)
                payoff2-=c
    '''
    temporarily disabled to obtain a baseline of whether the math is good on its own            
    if(ind1.payoff==-50 and payoff2==-50):
        relationships[ind1.getPopIndex()][ind2.getPopIndex()]=0
        relationships[ind2.getPopIndex()][ind1.getPopIndex()]=0
    '''    
def whoDies():
    Deaths=[]
    min=1000
    for i in range(populationSize):
        if(population[i].payoff<min):
            min=population[i].payoff    
    while(len(Deaths)<howManyDie):
        index=randint(0, populationSize-1)
        person=population[index]    
        already=False
        for i in range(len(Deaths)):
            if(Deaths[i]==person):
                already=True
        difference=(person.payoff-min)        
        if(already!=True and difference<=20):
            Deaths.append(index)  
    
    return Deaths  
      
   
      
def whoReproduces():      
    moms=[]
    max=-50
    for i in range(populationSize):
        if(population[i].payoff>max):
            max=population[i].payoff
    while(len(moms)<howManyDie):
        index=randint(0, populationSize-1)
        person=population[index]    
        already=False
        for i in range(len(moms)):
            if(moms[i]==person):
                already=True
        difference=(max-person.payoff)        
        if(already!=True and difference<=20):
            moms.append(index)    
    return moms

def games():
    for i in range(populationSize):
        print(i)
        currIndividual=population[i]
        otherIndividual=getOther(currIndividual, i)
        playGame(currIndividual, otherIndividual)


def selection():
    deadPpl=whoDies()
    mothers=whoReproduces()
    for i in range(len(deadPpl)):
        population[deadPpl[i]]=Individual(population[mothers[i]].strategy, 0, deadPpl[i])
        for j in range(populationSize):
            relationships[deadPpl[i]][j]=0
            relationships[j][deadPpl[i]]=0
        if(deadPpl[i]!=mothers[i] and random.random()<=Pb): 
            relationships[deadPpl[i]][mothers[i]]=1   #bc im replacing the dead ppl with the offspring, this, 
            relationships[mothers[i]][deadPpl[i]]=1   
        for j in range(populationSize):
            if(deadPpl[i]!=j and relationships[mothers[i]][j]==1 and random.random()<=Pn):
                relationships[deadPpl[i]][j]=1   #bc im replacing the dead ppl with the offspring, this, 
                relationships[j][deadPpl[i]]=1   #however odd it seems, works.   
            if(deadPpl[i]!=j and random.random()<=Pr):
                relationships[deadPpl[i]][j]=1   
                relationships[j][deadPpl[i]]=1    
        
    for i in range(populationSize):            
        population[i].payoff=0   
        population[i].alwaysD=False
        population[i].alwaysC=False    

def mutation():
    for i in range(populationSize):
        if(random.random()<mutationRate):
            currIndividual=population[i]
            currIndividual.mutate()
            

def getNumWithStrat(strat):
    count=0
    for i in range(populationSize):
        if(population[i].strategy==strat):
            count=count+1
    return count                   
            

def runTimeStep():
    global currTimeStep
    games()
    selection()
    mutation()
    for i in range(stratsSize):
        data[i][currTimeStep]=getNumWithStrat(strategies[i])
    currTimeStep=currTimeStep+1 
           
           
for k in range(simulationRuns):
    strt=time.time()
    currTimeStep=0
    runSim()  
    
    degreeDistribution=[0 for i in range(populationSize)]
    clusco=np.zeros(populationSize+1)
    
    g=igraph.Graph().Adjacency(relationships, mode=1)
    
    howManyConnections=g.degree(list(range(populationSize)), mode=3, loops=True)
    localCoefs=g.transitivity_local_undirected(list(range(populationSize)), mode="zero")#raw local clustering coefficient for each individual
    
    for i in range(populationSize):
        degreeDistribution[howManyConnections[i]]+=1 #counting how many ppl have a certain number of connections
        clusco[int(localCoefs[i]*67)]+=1
    popLeft=populationSize
    for i in range(populationSize):
        popLeft-=degreeDistribution[i]
        degreeDistribution[i]=popLeft/populationSize
    popLeft=populationSize
    for i in range(len(clusteringCoefData[0])):
        popLeft-= clusco[i]
        clusco[i]=popLeft/populationSize    
        
    degreeDistributionData[k]=degreeDistribution
    clusteringCoefData[j]=clusco
    
    print(time.time()-strt)
    
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

plt.show()

plt.show() 
print(time.time()-startTime)