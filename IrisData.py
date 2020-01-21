from mpl_toolkits.mplot3d import Axes3D 

import csv
import math
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import random

class IrisClassifier:
    def __init__(self):
        self.w = Vector4(random.randint(1,7),random.randint(2,10),-1 * random.randint(3,18), -1)
        self.Data = []
        self.num = []
        self.a = 0.05
        self.ErrorToStop = 0.025
        self.plotVectors = []
        self.LearnCurve = []
        self.batch = self.w
        self.BatchData = []

    def parse(self, file):
        with open(file) as csvData:
            csvRead = csv.reader(csvData)
            for row in csvRead:
                if(row[0] != 'sepal_length'):
                    #if(row[4] == 'setosa'):
                    if(row[4] == 'versicolor'):
                        self.Data.append(Vector4(float(row[2]), float(row[3]), 1, 0))
                    if(row[4] == 'virginica'):
                        self.Data.append(Vector4(float(row[2]), float(row[3]), 1, 1))

    def train(self):
        self.plotVectors.append(self.w)
        k = random.randint(0,len(self.Data))
        i = 0
        while(self.MeanSquaredError() >= self.ErrorToStop and i < 100000):
            if(self.MeanSquaredError() > 0.49):
                self.w = Vector4(random.randint(1,6),random.randint(3,8),-1 * random.randint(3,18), -1)
            if(k == len(self.Data)):
                k = 0
            if(i % 50 == 0):
                self.plotVectors.append(self.w)
            self.num.append(len(self.num))
            self.UpdateWeights(self.Data[k], i)
            k += 1
            i += 1
        self.plotVectors.append(self.w)

    def BatchTrain(self):
        k = 0
        i = 0
        while(self.MeanSquaredError() >= self.ErrorToStop or i < 100000):
            mean = Vector4(0,0,1,-1)
            j = 0
            while(j < 20):
                mean = mean + self.BatchUpdateWeights(self.Data[(k+j)%len(self.Data)])
                j += 1
            self.batch = self.batch + mean * self.a
            self.BatchData.append(self.batch)
            k += 20
            i += 1


    def hW(self, x):
        return (1 / (1 + math.exp(-1 * self.w.Dot(x))))

    def UpdateWeights(self, x, i):
        self.a = 250 / (250 + i) 
        self.w = self.w + ((x * (x.val - self.hW(x)) * self.hW(x) * (1 - self.hW(x))) * self.a)
        self.LearnCurve.append(self.LearningCurve())

    def BatchUpdateWeights(self,x):
        return (x * (x.val - self.hWb(x))

    def hWb(self, x):
        return (1 / (1 + math.exp(-1 * self.batch.Dot(x))))

    def MeanSquaredError(self):
        i = 0
        mean = 0
        while(i < len(self.Data)):
            mean += (self.hW(self.Data[i]) - self.Data[i].val) * (self.hW(self.Data[i]) - self.Data[i].val)
            i += 1
        return (mean/(i+1))

    def LearningCurve(self):
        i = 0
        correct = 0
        while(i < len(self.Data)):
            if(abs(self.hW(self.Data[i]) - self.Data[i].val) <= 0.2):
                correct += 1
            i += 1
        return (correct / i)

    
class Vector4:
    def __init__(self, x, y, b, val):
        self.x = x
        self.y = y
        self.b = b
        self.val = val

    def Dot(self, b):
        return ((self.x * b.x) + (self.y * b.y) + (self.b * b.b))

    def GetSlope(self):
        return (-1 * (self.b / self.y ) / (self.b / self.x))

    def GetIntercept(self):
        return (-1 * self.b / self.y)

    def __add__(self,b):
        return Vector4(self.x + b.x, self.y + b.y, self.b + b.b, -1)

    #def __add__(self, b):
    #    return Vector4(self.x + b, self.y + b, self.b + b, -1)

    def __sub__(self, b):
        return Vector4(self.x - b.x, self.y - b.y, self.b + b.b, -1)

    def __mul__(self, b):
        return Vector4(self.x * b, self.y * b, self.b * b, -1)
    
    def __div__(self,b):
        return Vector4(self.x / b, self.y / b, self.b/b, -1)




fig = plt.figure()
data = IrisClassifier()
data.parse('irisdata.csv')
data.train()
#data.BatchTrain()


manualData = IrisClassifier()
manualData.w = Vector4(0.04675,0.0295,1, 0)
manualData.w.b = -0.28

mseTest1 = IrisClassifier()
mseTest1.w = Vector4(0.9,5.1,1,0)
mseTest1.w.b = -13

mseTest2 = IrisClassifier()
mseTest2.w = Vector4(2.2,3.0,1,0)
mseTest2.w.b = -8.2

mseTest1.Data = mseTest2.Data = manualData.Data = data.Data

#Print MSEs and weight vectors
print("Data MSE: ", round(data.MeanSquaredError() + 0.0001,3), "W0: ", round(data.w.x + 0.0001,3), "W1: ", round(data.w.y + 0.0001,3), "b: ", round(data.w.b + 0.0001,3))
print("Manual Data MSE: ", round(manualData.MeanSquaredError() + 0.0001,3), "W0: ", round(manualData.w.x + 0.0001,3), "W1: ", round(manualData.w.y + 0.0001,3), "b: ", round(manualData.w.b + 0.0001,3))
print("Small MSE: ", round(mseTest1.MeanSquaredError() + 0.0001,3), "W0 : ", mseTest1.w.x, "W1: ", mseTest1.w.y, "B: ", mseTest1.w.b)
print("Large MSE: ", round(mseTest2.MeanSquaredError() + 0.0001,3), "W0 : ", mseTest2.w.x, "W1: ", mseTest2.w.y, "B: ", mseTest2.w.b)

#Print Example Points
print("Example Data Points: ")
i = 0
while(i < len(data.Data)):
    print('Classifier Value: ',round(data.hW(data.Data[i]) + 0.0001,3),'True Value: ', data.Data[i].val, 'Petal Length: ',data.Data[i].x, 'Petal Width: ', data.Data[i].y)
    i += 4
i = 0
while(i < len(data.Data)):
    if(i % 4 != 0):
        print('Classifier Value: ',round(data.hW(data.Data[i]) + 0.0001,3),'True Value: ', data.Data[i].val, 'Petal Length: ',data.Data[i].x, 'Petal Width: ', data.Data[i].y)
    i += 5
print("\n")

#Plot the data
i = 0
pVersW = []
pVersL = []
pVirgW = []
pVirgL = []
while(i < len(data.Data)):
    if(data.Data[i].val == 1):
        pVirgW.append(data.Data[i].y)
        pVirgL.append(data.Data[i].x)
    else:
        pVersW.append(data.Data[i].y)
        pVersL.append(data.Data[i].x)
    i += 1

#Data Plot
ax1 = fig.add_subplot(321)

z = np.linspace(4,6,2)
x = np.linspace(0, data.num, 1)

ax1.plot(pVirgL, pVirgW, '.', label='Virginica')
ax1.plot(pVersL, pVersW, '.', label='Versicolor')

ax1.plot(z, manualData.w.GetSlope() * z + manualData.w.GetIntercept(), linestyle='--', label='Manual Dec. Bound')
ax1.plot(z, data.w.GetSlope() * z + data.w.GetIntercept(), linestyle='-', label='Computed Dec. Boundary')
ax1.plot(z, mseTest1.w.GetSlope() * z + mseTest1.w.GetIntercept(), linestyle='--', label='Small MSE')
ax1.plot(z, mseTest2.w.GetSlope() * z + mseTest2.w.GetIntercept(), linestyle='--', label='Large MSE')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)

#ax1.legend()
#ax1.xlabel('Petal Length (cm)')
#ax1.ylabel('Petal Width (cm)')
#ax1.title("Iris Data")

#Surface Plot
ax2 = fig.add_subplot(322, projection = '3d')
X = np.arange(0,10,0.25)
Y = np.arange(0,5,0.25)
X,Y = np.meshgrid(X,Y)
w0 = data.w.x
w1 = data.w.y
w2 = data.w.b

Z = 1 / (1 + np.exp(-1 * (X * w0+ Y * w1 + 1 * w2)))
ax2.plot_surface(X,Y,Z)

#Gradient Step Plot
ax3 = fig.add_subplot(323)
ax3.plot(pVirgL, pVirgW, '.', label='Virginica')
ax3.plot(pVersL, pVersW, '.', label='Versicolor')

i = 0
for n in range(0, len(data.BatchData)):
    if(data.BatchData[i].x != 0 and data.BatchData[i].y != 0):
        ax3.plot(z, data.BatchData[i].GetSlope() * z + data.BatchData[i].GetIntercept(), linestyle='--'); 
    i += 1



#Initial, Mid, and Final Boundary Plot
ax4 = fig.add_subplot(324)

ax4.plot(pVirgL, pVirgW, '.', label='Virginica')
ax4.plot(pVersL, pVersW, '.', label='Versicolor')
i = len(data.plotVectors) - 1
ax4.plot(z, data.plotVectors[i].GetSlope() * z + data.plotVectors[i].GetIntercept(), linestyle='dotted'); 
ax4.plot(z, data.plotVectors[0].GetSlope() * z + data.plotVectors[0].GetIntercept(), linestyle='dotted');
ax4.plot(z, data.plotVectors[(int)(i/2)].GetSlope() * z + data.plotVectors[(int)(i/2)].GetIntercept(), linestyle='dotted'); 

#Learning Curve Plot
ax5 = fig.add_subplot(326)

ax5.plot(data.num, data.LearnCurve, '-')

#Boundary over Time Plot
ax6 = fig.add_subplot(325)

ax6.plot(pVirgL, pVirgW, '.', label='Virginica')
ax6.plot(pVersL, pVersW, '.', label='Versicolor')

i = 0
for n in range(0, len(data.plotVectors)):
    if(data.plotVectors[i].x != 0 and data.plotVectors[i].y != 0):
        ax6.plot(z, data.plotVectors[i].GetSlope() * z + data.plotVectors[i].GetIntercept(), linestyle='dotted'); 
    i += 1


plt.show()
