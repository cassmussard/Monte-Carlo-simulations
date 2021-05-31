import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from numpy import genfromtxt
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
import time



#Define the Phi_alpha function in generalized convex spaces
class Pca_alpha(object):
    
    def __init__(self, alpha):
        
        self.alpha = alpha

    def phi_alpha(self, a):
        if a < 0:
            return -np.abs(a)**(1/self.alpha)
        elif a == 0:
            return 0
        else:
            return a**(1/self.alpha)
        
    def phi_alpha_matrix(self, x, a):
        n, k = x.shape
        B = np.zeros_like(x)
        for j in range(k):
            for i in range(n):
                if x[i,j] < 0:
                    B[i,j] = -np.abs(x[i,j])**(1/a)
                elif x[i,j] == 0:
                    B[i,j] = 0
                else:
                    B[i,j] = x[i,j]**(1/a)
        return B

    def project(self, x, a):
        n, k = x.shape
        Z = preprocessing.scale(x)
        R = (1/n) * (Z.T @ Z)
        _, vecp = linalg.eig(R)
        F = np.real(Z @ vecp)
        F_alpha = self.phi_alpha_matrix(F, a)
        return F_alpha

    def eigenvalues(self, x):
        n, k = x.shape
        Z = preprocessing.scale(x)
        R = (1/n) * (Z.T @ Z)
        valp, _ = linalg.eig(R)
        A = np.zeros((k,3))
        A[:,0] = np.real(valp.T)
        A = self.phi_alpha_matrix(A, self.alpha)
        A[:,2] = (np.cumsum(A[:,0]) / np.sum(A[:,0]))*100
        A[:,1] = (A[:,0] / A[:,0].sum())*100
        return A[:,0]

    def cta_var(self, x):
        n, k = x.shape
        Z = preprocessing.scale(x)
        R = (1/n) * (Z.T @ Z)
        valp, P = linalg.eig(R)
        cta_var = np.diag(np.real(valp)**(0.5)) @ P.T
        cta_var = self.phi_alpha_matrix(cta_var, self.alpha)
        cta_var1 = ((cta_var.T)**2 / np.sum((cta_var.T)**2, axis = 0))*100
        return np.round(cta_var1, 1)
    
    def cta(self, x):
        n, k = x.shape
        F = self.project(x, self.alpha)
        cta = (F**2 / np.sum(F**2, axis = 0))*100
        return np.round(cta, 1)

    def cta_var2(self, x):
        n, k = x.shape
        Z = preprocessing.scale(x)
        F = self.project(x, self.alpha)
        Z = self.phi_alpha_matrix(Z, self.alpha)
        P = np.zeros((k,k))
        for i in range(k):
            for j in range (k):
                P[i,j] = ss.pearsonr(F[:,j], Z[:,i])[0]                
        cta_var = (P**2 / np.sum(P**2, axis = 0))*100
        return np.round(cta_var, 1)
    
    def ctr(self, x):
        n, k = x.shape
        F = self.project(x, self.alpha)
        ctr = ((F.T)**2 / np.sum((F.T)**2, axis = 0))*100
        return np.round(ctr.T, 1)
    
    def hotelling(self, x, prob):
        "optimal alpha"
        n, k = x.shape
        list0 = []
        list_0 = []
        for a in np.arange(0.1, 10, 0.1):
            F = self.project(x, a)
            list1 = []
            list2 = []
            Hotelling1 = (n**2)*(n-1)/((n**2-1)*(n-1)) * (F[:,0])**2 / np.var(F[:,0])
            Hotelling2 = (n**2)*(n-2)/((n**2-1)*(n-1)) * ((F[:,0])**2 / np.var(F[:,0]) + (F[:,1])**2 / np.var(F[:,1]))
            for i in range(n):
                if Hotelling1[i] >= ss.f.ppf(1-prob,dfn=1,dfd=n-1):
                    list1.append(i)
                if Hotelling2[i] >= ss.f.ppf(1-prob,dfn=2,dfd=n-2):
                    list2.append(i)
            list0.append(len(list1))
            list_0.append(len(list2))
        return np.argmin(np.asarray(list0))/10 + 0.1, np.argmin(np.asarray(list_0))/10 + 0.1 

    def plot3D(self,x, y):
        n, k = x.shape
        F = self.project(x, self.alpha)
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(F[:, 0], F[:, 1], F[:, 2],c = y, cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title("Phi_alpha PCA")
        ax.set_xlabel("1st component")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd component")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd component")
        ax.w_zaxis.set_ticklabels([])
        return plt.show()

    def plot3D2(self,x):
        n, k = x.shape
        F = self.project(x, self.alpha)
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title("Phi_alpha PCA")
        ax.set_xlabel("1st component")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd component")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd component")
        ax.w_zaxis.set_ticklabels([])
        return plt.show()


#Start the Monte-Carlo simulations
debut = time.time() #Measure the time calculation
result = np.zeros((7,1))
for alpha in np.arange(1, 8, 1):
    mean = [10, 10, 10]
    cov = [[1, 0.9, 0.1], [0.9, 1, 0], [0.1, 0, 1]]
    model = Pca_alpha(alpha)
    stock = np.zeros((1000,3))
    #Browse the contaminated parameter
    for contamin_param in np.arange(1, 1001, 1):
        x = np.random.multivariate_normal(mean, cov, 500)#Generate a normal distribution with 3 dimensions
        np.random.seed(123) #Fix the normal distribution
        eig = model.eigenvalues(x) 
        a = int(np.random.randint(0,500)) #Chose a random line to contaminate
        #Contaminate
        x[a,:] = contamin_param*x[a,:]
        stock[contamin_param-1,:] = eig-model.eigenvalues(x) #Eigenvalues contaminated
    result[int(alpha-1),:] = (1/1000)*np.sum(stock[contamin_param-1,:]**2, axis = 0)
print(result)
print(time.time() - debut) 

#Print the graphic 
plt.plot(np.arange(1, 8), result[:,0], 'r--', label='MSE')#[np.argmin(result[:,0])].set_color('r')
plt.ylabel('MSE')
plt.xlabel('Alpha values')
plt.title('MSE Curve')
plt.legend(loc='best')
plt.show()