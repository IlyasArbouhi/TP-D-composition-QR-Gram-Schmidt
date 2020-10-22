# -*- coding: utf-8 -*-
"""
ARBOUHI KAMAESWARAN
Created on Tue Mar 24 16:54:43 2020

@author: arbou
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import math as m



def DecompositionGS (A):
	n,m=A.shape

	Q,R=np.zeros((n,m)),np.zeros((n,m))

	R[0,0]=np.linalg.norm(A[:,0])
	Q[:,0]=A[:,0]/R[0,0]

	for i in range(1,n):
		for j in range(i):
			R[j,i]=np.dot(A[:,i],Q[:,j])

		somme =0
		for t in range(i):
			somme += R[t,i]*Q[:,t]
		W=A[:,i]- somme

		R[i,i]=np.linalg.norm(W)

		Q[:,i] = W/R[i,i]

	return Q,R,np.dot(Q,R)

def ReductionGauss(Aaug):
    lignes,colonnes =np.shape(Aaug)
    for k in range (0,lignes-1):
        for i in range (k+1,lignes):
            gik=Aaug[i,k]/Aaug[k,k]
            Aaug[i,:]=Aaug[i,:]-gik*Aaug[k,:]
    return Aaug



def ResolutionSysTriSup(l):
    lignes,colonnes=np.shape(l)
    X=np.zeros(lignes)
    X = np.array(X, dtype=float)
    
    for i in range(lignes-1,-1,-1):
        somme=0
        for k in range (i+1,lignes):
            somme=somme+l[i,k]*X[k]
        X[i]=(l[i,lignes]-somme)/l[i,i]
    return X



 

def ResolutionSysTriInf(A): 
    lignes,colonnes=np.shape(A)
    X=np.zeros(lignes)
    X = np.array(X, dtype=float)
    for i in range(0,lignes):
        somme=0
        for k in range(0,i):
            somme=somme+A[i,k]*X[k]
        X[i]=(A[i,lignes]-somme)/A[i,i]
    return X

def Gauss (A,B):
   
   Aaug = np.concatenate((A, B.T), axis = 1)
   Taug = ReductionGauss(Aaug)
   S = ResolutionSysTriSup(Taug)
   return S


def Resosup(A,b):
	n,m=A.shape
	x=np.zeros(n)
	for i in range(n-1, -1,-1):
		S = np.dot(A[i,i+1:],(x[i+1:]))
		x[i]=(1/A[i,i])*(b[i]-S)
	return x

def ResolGS(A,b):
    Q,R,P=DecompositionGS(A)
    C = np.linalg.cond(Q)
    Y = np.dot(Q.T,b)
    X=Resosup(R,Y)
    X = np.asarray(X).reshape(n,1)
    return X,C



def DecompositionLU(A):
    lignes,colonnes = np.shape(A)
    U=np.zeros((lignes,colonnes))
    U = np.array(U, dtype=float)
    L=np.eye(lignes)
    L = np.array(L, dtype=float)
    for i in range(0,lignes):
        for k in range(i+1,lignes):
            pivot = A[i][i]
            pivot = A[k][i]/pivot
            L[k][i]=pivot
            for j in range(i,lignes):
                A[k][j]=A[k][j]-(pivot*A[i][j])
    U=A
    return L,U


def ResolutionLU(A,B):
    lignes,colonnes = np.shape(A)
    X=np.zeros(lignes)
    X = np.array(X, dtype=float)
    L,U=DecompositionLU(A)
    
    Y = np.concatenate((L,B.T),axis =1)
    
    Y=np.array(ResolutionSysTriInf(Y))
    Y=np.asarray(Y).reshape(lignes,1)
    X=ResolutionSysTriSup(np.concatenate((U,Y),axis =1))
    
    return (X)

def ReductionGaussPartiel(A) :
    Aaug = A.copy()
    Aaug = np.array(Aaug, dtype=float)
    lignes,colonnes=Aaug.shape
    
    
    for k in range(colonnes-1):
        for i in range(k,lignes): #balayage ligne
            if abs(Aaug[k,k]) < abs(Aaug[i,k]):
                K= np.copy(Aaug)
                Aaug[k,:]=Aaug[i,:]
                Aaug[i,:]=K[k,:]
                
        for i in range (k+1,lignes):
            gik=Aaug[i,k]/Aaug[k,k]
            Aaug[i,:]=Aaug[i,:]-gik*Aaug[k,:]

    return Aaug



def GaussChoixPivotPartiel(A,B):
    
    Aaug = np.concatenate((A, B.T), axis = 1)
    
    Taug=ReductionGaussPartiel(Aaug)
    X= ResolutionSysTriSup(Taug)
    
    return(X)


def ReductionGaussTotal(A) :
    Aaug = A.copy()
    Aaug = np.array(Aaug, dtype=float)
    
    lignes,colonnes=Aaug.shape
    X = np.array(['x','y','z'])
    
    for k in range(colonnes-1):
        
        Anonaug = Aaug[k:colonnes-1,k:lignes]
        
        l,c= np.where(Aaug == np.amax(Anonaug))
        l = int(l)
        c = int(c)  
        
        if (k,k) != (l,c):
            K= np.copy(Aaug)
            Aaug[k,:]=Aaug[l,:] #On change les lignes 
            Aaug[l,:]=K[k,:]
            
            K= np.copy(Aaug)
            Aaug[:,k]=Aaug[:,c] #On change les colonnes 
            Aaug[:,c]=K[:,k]
            
            K= X[c]
            X[c]=X[k] #On regarde les changements sur l'ordre des solutions
            X[k]= K
                
        for i in range (k+1,lignes):
            
            gik=Aaug[i,k]/Aaug[k,k]
            Aaug[i,:]=Aaug[i,:]-gik*Aaug[k,:]
              
    return Aaug,X



def GaussChoixPivotTotal(A,B):
    
    Aaug = np.concatenate((A, B.T), axis = 1)
    
    Taug,Xformat=ReductionGaussTotal(Aaug)
    X= ResolutionSysTriSup(Taug)
    
    return X,Xformat

def Cholesky(A):
    n,m = np.shape(A) 
    L = np.zeros((n,m))

    for k in range(m):
        S = 0
        for j in range(k):
            S = S+L[k,j]**2

        else :
            L[k][k] = (A[k][k]-S)**(1/2)

        for i in range(k+1,n):

            S2 = 0 
            for j in range(k):
                S2 = S2+L[i][j]*L[k][j]

            L[i][k] = (A[i][k]-S2)/L[k][k]


    return L




def ResolCholesky(A,B):
    B = np.reshape(B, (n,1))
    L=Cholesky(A)
    Taug= np.hstack((L,B))          ###Hstack --> empile les matrices horizontalement (en colonnes)
    Y=ResolutionSysTriInf(Taug)
    Y=Y[:,np.newaxis]               ###newaxis --> augmente d'une dimension, la dimension de la matrice
    LT=np.transpose(L)              ###transpose --> fait la transposée de L
    Baug= np.hstack((LT,Y))         ###Hstack --> empile les matrices horizontalement (en colonnes)
    X=ResolutionSysTriSup(Baug)
    return X

def ResolCholeskypy(A,B):
    B = np.reshape(B, (n,1))
    L = np.linalg.cholesky(A)
    Taug= np.hstack((L,B))          ###Hstack --> empile les matrices horizontalement (en colonnes)
    Y=ResolutionSysTriInf(Taug)
    Y=Y[:,np.newaxis]               ###newaxis --> augmente d'une dimension, la dimension de la matrice
    LT=np.transpose(L)              ###transpose --> fait la transposée de L
    Baug= np.hstack((LT,Y))         ###Hstack --> empile les matrices horizontalement (en colonnes)
    X=ResolutionSysTriSup(Baug)
    return X


"""
X,formatX=GaussChoixPivotTotal(A,B)
print("X=",X,"\nDans l'ordre :",formatX)


C = np.random.randint(low= 1, high=5,size=(n,n))
while np.linalg(C)==0:
    C = np.random.randint(low= 1, high=5,size=(n,n))
    C = np.array(C, dtype=float)
A = C.dot(C.T)
""""""

Temps = []
indices = []

for n in range(10,500,50):
    try:
        A= np.random.randint(low=1,high=n,size=(n,n))
        A = np.array(A, dtype=float)
        B= np.random.randint(low=1,high=n,size=(n,1))
        B = np.array(B, dtype=float)
        
        
        
        X,C = ResolGS(A,B)
        print(C)
        Temps.append(C) 
        indices.append(n)
        
    except:
        print('')

x1 = indices
y1 = Temps
plt.plot(x1,y1)

plt.title("Conditionnement")
plt.ylabel('')
plt.xlabel('n')

plt.show()


  


Temps = []
indices = []

for n in range(10,500,50):
    try:
        A= np.random.randint(low=1,high=n,size=(n,n))
        A = np.array(A, dtype=float)
        B= np.random.randint(low=1,high=n,size=(1,n))
        B = np.array(B, dtype=float)
        
        
        
        X = GaussChoixPivotPartiel(A,B)
        S = np.linalg.norm(A.dot(X)-B)
        
        Temps.append(S) 
        indices.append(n)
        
    except:
        print('')

x1 = indices
y1 = Temps
plt.plot(x1,y1)

plt.title("Pivot de Gauss")
plt.ylabel('erreur')
plt.xlabel('n')

plt.show()


Temps = []
indices = []

for n in range(10,500,50):
    try:
        
        A= np.random.randint(low=1,high=n,size=(n,n))
        A = np.array(A, dtype=float)
        B = np.random.randint(low=-100,high=100,size=(n,1))
        
        B = np.array(B, dtype = float)
        
        
        X = ResolGS(A,B)
        W = A.dot(X)-B
        S = np.linalg.norm(W)
        
        Temps.append(S) 
        indices.append(n)
        
    except:
        print('')

x1 = indices
y1 = Temps
plt.plot(x1,y1)

plt.title("Gram-Schmidt")
plt.ylabel('erreur')
plt.xlabel('n')

plt.show()

Temps = []
indices = []

for n in range(10,500,50):
    try:
        
        B= np.random.randint(low=1,high=n,size=(1,n))
        B = np.array(B, dtype=float)
        
        C = np.random.randint(low= 1, high=10,size=(n,n))
        C = np.array(C, dtype=float)
        while np.linalg.det(C)==0:
            C = np.random.randint(low= 1, high=10,size=(n,n))
            C = np.array(C, dtype=float)
        A = C.dot(C.T)
        
        X = ResolCholesky(A,B)
        S = np.linalg.norm(A.dot(X)-B)
        
        Temps.append(S) 
        indices.append(n)
        
    except:
        print('')

x1 = indices
y1 = Temps
plt.plot(x1,y1)

plt.title("Résolution Cholesky")
plt.ylabel('erreur')
plt.xlabel('n')

plt.show()

Temps = []
indices = []

for n in range(10,500,50):
    try:
        A= np.random.randint(low=1,high=n,size=(n,n))
        A = np.array(A, dtype=float)
        B= np.random.randint(low=1,high=n,size=(1,n))
        B = np.array(B, dtype=float)
        
        Acop = A.copy()
        X = ResolutionLU(A,B)
        S = np.linalg.norm(Acop.dot(X)-B)
        
        Temps.append(S) 
        indices.append(n)
        
    except:
        print('')

x1 = indices
y1 = Temps
plt.plot(x1,y1)

plt.title("LU")
plt.ylabel('erreur')
plt.xlabel('n')

plt.show()

Temps = []
indices = []

for n in range(10,500,50):
    try:
        #A= np.random.randint(low=1,high=n,size=(n,n))
        #A = np.array(A, dtype=float)
        B= np.random.randint(low=1,high=n,size=(1,n))
        B = np.array(B, dtype=float)
        
        C = np.random.randint(low= 1, high=4,size=(n,n))
        C = np.array(C, dtype=float)
        while np.linalg.det(C)==0:
            C = np.random.randint(low= 1, high=4,size=(n,n))
            C = np.array(C, dtype=float)
        A = C.dot(C.T)
        
        X = ResolCholeskypy(A,B)
        S = np.linalg.norm(A.dot(X)-B)
        
        Temps.append(S) 
        indices.append(n)
        
    except:
        print('')

x1 = indices
y1 = Temps
plt.plot(x1,y1)

plt.title("Résolution Cholesky python")
plt.ylabel('erreur')
plt.xlabel('n')

plt.show()


Temps = []
indices = []

for n in range(10,600,20):
    try:
        #A= np.random.randint(low=1,high=n,size=(n,n))
        #A = np.array(A, dtype=float)

        B= np.random.randint(low=1,high=n,size=(1,n))
        B = np.array(B, dtype=float)
        
        C = np.random.randint(low= 1, high=5,size=(n,n))
        C = np.array(C, dtype=float)
        while np.linalg.det(C)==0:
            C = np.random.randint(low= 1, high=5,size=(n,n))
            C = np.array(C, dtype=float)
        A = C.dot(C.T)
        
        
        t1 = time.time()
        Gauss(A,B)
        t2 = time.time()
        t = t2 - t1
        
        Temps.append(t)
        indices.append(n)
    except:
        print('1')

x1 = indices
y1 = Temps

Temps = []
indices = []

for n in range(10,600,20):
    try:
        #A= np.random.randint(low=1,high=n,size=(n,n))
        #A = np.array(A, dtype=float)

        B= np.random.randint(low=1,high=n,size=(n,1))
        B = np.array(B, dtype=float)
        
        C = np.random.randint(low= 1, high=5,size=(n,n))
        C = np.array(C, dtype=float)
        while np.linalg.det(C)==0:
            C = np.random.randint(low= 1, high=5,size=(n,n))
            C = np.array(C, dtype=float)
        A = C.dot(C.T)
        
        
        t1 = time.time()
        ResolGS(A,B)
        t2 = time.time()
        t = t2 - t1
        
        Temps.append(t)
        indices.append(n)
    except:
        print('2')
y2 = Temps
x2 = indices


Temps = []
indices = []

for n in range(10,400,20):
    try:
        #A= np.random.randint(low=1,high=n,size=(n,n))
        #A = np.array(A, dtype=float)

        B= np.random.randint(low=1,high=n,size=(1,n))
        B = np.array(B, dtype=float)
        
        C = np.random.randint(low= 1, high=5,size=(n,n))
        C = np.array(C, dtype=float)
        while np.linalg.det(C)==0:
            C = np.random.randint(low= 1, high=5,size=(n,n))
            C = np.array(C, dtype=float)
        A = C.dot(C.T)
        
        
        t1 = time.time()
        ResolCholesky(A,B)
        t2 = time.time()
        t = t2 - t1
        
        Temps.append(t)
        indices.append(n)
    except:
        print('3')
        
y3 = Temps
x3 = indices

Temps = []
indices = []

for n in range(10,400,20):
    try:
        #A= np.random.randint(low=1,high=n,size=(n,n))
        #A = np.array(A, dtype=float)

        B= np.random.randint(low=1,high=n,size=(1,n))
        B = np.array(B, dtype=float)
        
        C = np.random.randint(low= 1, high=5,size=(n,n))
        C = np.array(C, dtype=float)
        while np.linalg.det(C)==0:
            C = np.random.randint(low= 1, high=5,size=(n,n))
            C = np.array(C, dtype=float)
        A = C.dot(C.T)
        
        t1 = time.time()
        ResolutionLU(A,B)
        t2 = time.time()
        t = t2 - t1
        
        Temps.append(t)
        indices.append(n)
    except:
        print('4')
y4 = Temps
x4 = indices

Temps = []
indices = []

for n in range(10,600,20):
    try:
        #A= np.random.randint(low=1,high=n,size=(n,n))
        #A = np.array(A, dtype=float)

        B= np.random.randint(low=1,high=n,size=(1,n))
        B = np.array(B, dtype=float)
        
        C = np.random.randint(low= 1, high=5,size=(n,n))
        C = np.array(C, dtype=float)
        while np.linalg.det(C)==0:
            C = np.random.randint(low= 1, high=5,size=(n,n))
            C = np.array(C, dtype=float)
        A = C.dot(C.T)
        
        
        t1 = time.time()
        ResolCholeskypy(A,B)
        t2 = time.time()
        t = t2 - t1
        
        Temps.append(t)
        indices.append(n)
    except:
        print('5')
y5 = Temps
x5 = indices

Temps = []
indices = []

for n in range(10,600,20):
    try:
        #A= np.random.randint(low=1,high=n,size=(n,n))
        #A = np.array(A, dtype=float)

        B= np.random.randint(low=1,high=n,size=(1,n))
        B = np.array(B, dtype=float)
        
        C = np.random.randint(low= 1, high=5,size=(n,n))
        C = np.array(C, dtype=float)
        while np.linalg.det(C)==0:
            C = np.random.randint(low= 1, high=5,size=(n,n))
            C = np.array(C, dtype=float)
        A = C.dot(C.T)
        
        
        t1 = time.time()
        GaussChoixPivotPartiel(A,B)
        t2 = time.time()
        t = t2 - t1
        
        Temps.append(t)
        indices.append(n)
    except:
        print('5')
y6 = Temps
x6 = indices

plt.plot(x1,y1,c='b',label	='Gauss')
plt.plot(x2,y2,c='g',label	='GS')
#plt.plot(x3,y3,c='y',label	='Cholesky')
#plt.plot(x4,y4,c='c',label	='LU')
plt.plot(x5,y5,c='m',label	='Choleskypy')
plt.plot(x6,y6,c='r',label	='Gauss Partiel')


plt.xlabel('n')
plt.title("Temps en fonction de n")
plt.ylabel('Temps')

plt.legend()
plt.show()
"""
"""

A=np.array([[1,4,1],[2,3,2],[-2,2,1]])
DecompositionGS(A)

A = np.array([[6,6,16],[-3,-9,-2],[6,-6,-8]],dtype=np.dtype(float))
Q,R = DecompositionGS(A)
print(Q)
print("")
print(R)
#DecompositionGS(A)
A = np.array([[1,1,1],[-1,0,1],[1,1,2]],dtype=np.dtype(float))
print(A)

Q,R = DecompositionGS(A)
print(Q)
print("")
print(R)
"""