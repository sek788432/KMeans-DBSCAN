#!/usr/bin/env python
# coding: utf-8

import random
from random import randrange
import sys
from scipy.spatial import distance
import numpy as np

def my_kmeans(k, data, iteration = 30, switch = "++"):
    
    centroids=[]
    label =[0] * len(data)
    #The last column is label
    features = len(data.columns) - 1                  
    
    #Initialization: kmeans or kmeans++
    if switch == "++":
        plus_initialize(data, centroids, features, k)
    else:
        initialize(k, centroids)
    
    for i in range(iteration):
        last_centroids = centroids.copy()
        clust = update_cluster(data, k, label, centroids, features)       
        centroids = update_centroid(clust, features)
        if last_centroids == centroids:
            print("Centroids:\n", centroids)
            print("\nConverge at ", i, " iterations\n")
            break       
    print("My label:\n", np.array(label))
    return label

def initialize(c, centroids):
    #randomly generate centroid
    for i in range(c):
        #for IRIS data
        centroids.append([random.uniform(5,7), random.uniform(2,4), random.uniform(1,5), random.uniform(0,2)]) 
        
def plus_initialize(data,centroids,features,k):
    #pick a number from 0 to numbers of data-1
    ind = randrange(len(data))        
    centroid1 = []
    for j in range(features):
        centroid1.append(data.iloc[ind, j])
    centroids.append(centroid1)
    
    while(len(centroids) < k):
        max_dx = 0
        max_ind = 0
        tmp = []
        for i in range(len(data)): 
            point = []
            for j in range(features):
                point.append(data.iloc[i, j])
            dx = sys.maxsize        
            for j in range(len(centroids)):
                dist = distance.euclidean(centroids[j], point)
                if(dist < dx):
                    dx = dist                    
            if(dx > max_dx):
                max_dx = dx
                max_ind = i
                tmp = point.copy()
        centroids.append(tmp)     
      
def update_cluster(data, k, label, centroids, features):
    clusters=[]
    for i in range(k):
        clusters.append([]) 
    for i in range(len(data)):
        point = []
        for j in range(features):
            point.append(data.iloc[i, j])
        min_dist = sys.maxsize
        min_ind = 0
        for ind in range(k):
            dist =  distance.euclidean(point, centroids[ind])
            if dist < min_dist:
                min_dist = dist
                min_ind = ind
        clusters[min_ind].append(point)
        label[i] = min_ind
    return clusters

def update_centroid(cluster,features):   
    sum = []
    for i in range(len(cluster)):                 
        tmp = [0] * features 
        for j in range(len(cluster[i])):
            for k in range(features):            
                tmp[k] += cluster[i][j][k]
        for l in range(features):
            tmp[l] /= len(cluster[i])
        sum.append(tmp)
    return sum

def accuracy(label):
    error = 0
    start = 0
    for j in range(3):
        count = [0] * 3
        for i in range(start, 50 * (j+1)):
            count[label[i]] += 1
        error += 50 - max(count)
        start += 50   
    return (len(label) - error) / len(label)
