# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Magesh v
RegisterNumber: 212222040092
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="cadetblue")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted",color="plum")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot),color="cadetblue")
plt.show()

def costFunction(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h)))/x.shape[0]
  return j


def gradient(theta,x,y):

  h=sigmoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad


x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="mediumpurple")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted",color="pink")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=sigmoid(np.dot(x_train,theta))
  return(prob>=0.5).astype(int)

np.mean(predict(res.x,x)==y)
 
*/
```


## Output:
![ex 5 1](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/c2f189f6-4491-46e8-8218-233b5d57c2b6)
![ex 5 2](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/70e5f1b6-ce13-46ad-8687-b5210bfeafb9)
![ex 5 3](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/deff6b9f-8b69-488d-b370-3f70faadbad0)
![ex 5 4](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/4b4faefe-f30a-4d07-a114-c892e05ef3a5)
![ex 5 5](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/34b7249f-8478-459d-b4a7-2f0b694e49c3)
![ex 5 6](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/0a879df2-9187-4dab-9921-c9718cec526a)
![ex 5 7](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/3bcec7f0-6d22-4cb2-8735-24a09615c310)
![ex 57](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/f2a0f3f4-e755-44f5-b1ba-bc4f9693ab09)
![ex 5 8](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/cb63226b-fbfc-48f2-9e82-be89eba2a12f)
![ex 5 9](https://github.com/magesh534/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135577936/8002fcd1-1de8-43e2-b267-ae291554d8a1)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

