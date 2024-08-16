import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from numpy.linalg import norm
import sys

class KNN:
    def __init__(self, k, distance_metric, encoder_type):
        self.k = k
        self.distance_metric = distance_metric
        self.encoder_type = encoder_type
        self.train_x = None
        self.train_y = None
    
    def train(self,data1):
        if(self.encoder_type == 'RESNET'):
            self.train_x = data1[:,1]
            self.train_y = data1[:,3]
        if(self.encoder_type == 'VIT'):
            self.train_x = data1[:,2]
            self.train_y = data1[:,3]

    def predict(self, test_x):
        predictions=[]
        for x in test_x:
            distance = []
            for x1 in self.train_x:
                distance.append(self.get_distance(np.array(x[0]),np.array(x1[0])))
            
            distance = np.array(distance) 
            dist = np.argsort(distance)[:self.k] 
            labels = self.train_y[dist]
            predictions.append(max(list(labels), key = list(labels).count))
        return predictions

    def get_distance(self,x,x1):
        if(self.distance_metric=="euclidean"):
            dist = np.sqrt(np.sum((x-x1)**2))
        if(self.distance_metric=="manhattan"):
            dist = np.sum(np.abs(x-x1))
        if(self.distance_metric=="cosine"):
            dist = 1-(np.dot(x,x1)/(norm(x)*norm(x1)))
        return dist
    
    def validate(self,data1):
        if(self.encoder_type == 'RESNET'):
            val_x = data1[:,1]
            val_y = data1[:,3]
        if(self.encoder_type == 'VIT'):
            val_x = data1[:,2]
            val_y = data1[:,3]
        predictions = self.predict(val_x)
        f1 = f1_score(val_y, predictions, average='macro')
        accuracy = accuracy_score(val_y, predictions)
        precision = precision_score(val_y, predictions, average='macro',zero_division=np.nan)
        recall = recall_score(val_y, predictions, average='macro',zero_division=np.nan)
        return f1, accuracy, precision,recall

train = np.load("data.npy",allow_pickle=True)

file_path = sys.argv[1]   # Get file path from command line argument
val = np.load(file_path,allow_pickle=True)

knn = KNN(encoder_type= 'VIT', k=11, distance_metric="manhattan")
knn.train(train)
f,a,p,r=knn.validate(val)
print("f1-score : ",f)
print("accuracy : ",a)
print("precision : ",p)
print("recall : ",r)