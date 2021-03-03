#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.spatial import distance
import pandas as pd


def my_dbscan(Data, eps, Minpts):
    label = [-2] * len(Data)                    #label -2 means not visited
    cluster = 0;
    for point in range(len(Data)):
        if(label[point] == -2):
            neighbor_pts = get_neighbor(Data, point, eps)
            if len(neighbor_pts) < Minpts:
                label[point] = -1             #label -1 means noise
            else:
                expandCluster(point, neighbor_pts, cluster, eps, Minpts, label,Data)
                cluster += 1
    return label

def get_neighbor(Data, point, eps):
    neighbor = []
    for i in range(len(Data)):
        if distance.euclidean(Data[i], Data[point]) <= eps:
            neighbor.append(i)
    return neighbor

def expandCluster( point, neighbor_pts, cluster, eps, Minpts, label, Data):
    label[point] = cluster
    ind = 0
    while ind < len(neighbor_pts):
        curr_point = neighbor_pts[ind];
        neigh = get_neighbor(Data, curr_point, eps)
        if len(neigh) >= Minpts:      
            for i in range(len(neigh)):
                if neigh[i] not in neighbor_pts:
                    neighbor_pts.append(neigh[i])
        label[curr_point] = cluster
        ind+=1   
