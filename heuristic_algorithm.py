from read_data import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from enum import Enum
def get_neighbors(i, j,m,n):
    """Get the list of neighbors for a given (i, j) index in an m x n grid."""
    neighbors = []
    
    # Up
    if i > 0:
        neighbors.append((i - 1, j))
    # Down
    if i < m - 1:
        neighbors.append((i + 1, j))
    # Left
    if j > 0:
        neighbors.append((i, j - 1))
    # Right
    if j < n - 1:
        neighbors.append((i, j + 1))

    return neighbors

def main():
    grid_size = (64, 96)
    y, x = grid_size
    adj = neighbour_adj_matrix(y,x)
    cars = read_data()
    L2,DC = read_data_EV()
    L2_cap = 3
    DC_cap = 12
    L2_cost = 2000
    DC_cost = 7500
    budget = 9164000
    unserved_cars = {}
    L2_built = {}
    DC_built = {}
    #First process preexisting L2 chargers
    for i in range(y):
        for j in range(x):
            L2_built[(i, j)] = 0
            DC_built[(i, j)]=0
            unserved_cars[(i,j)] = cars[(i,j)]-L2[(i,j)]*L2_cap
    #Then process preexisting DC chargers depending on which neighboring block/own block has the most unserved cars
    #-DC[(i,j)]*DC_cap
    for i in range(y):
        for j in range(x):
            if DC[(i,j)] > 0:
                capacity = DC[(i,j)] * DC_cap
                #print("\nCapacity:", capacity, "at", i, j)
                
                neighbors = get_neighbors(i, j, y, x)
                neighbors.append((i, j))
                #print("Neighbors:", neighbors)
                
                sorted_neighbors = sorted(neighbors, key=lambda coord: unserved_cars[coord], reverse=True)
                unserved_cars_neighbors = {index: unserved_cars[index] for index in neighbors}
                
                #print("Unserved cars neighbors:", unserved_cars_neighbors)
                #print("Sorted neighbors:", sorted_neighbors)
                
                diff = unserved_cars[sorted_neighbors[0]] - unserved_cars[sorted_neighbors[1]]
                if diff < capacity:
                    unserved_cars[sorted_neighbors[0]] -= diff
                    capacity -= diff
                    
                    for k in range(2, len(sorted_neighbors)):
                        diff = unserved_cars[sorted_neighbors[0]] - unserved_cars[sorted_neighbors[k]]
                        if diff==0 and k+1<len(sorted_neighbors) and unserved_cars[sorted_neighbors[k+1]]==0:
                            continue
                        if diff * k < capacity and diff >0:
                                for m in range(k):
                                    unserved_cars[sorted_neighbors[m]] -= diff
                                    capacity -= diff
                        else:
                            #print(k,capacity)
                            if capacity % (k+1) != 0:
                                for n in range(int(capacity % (k+1))):
                                    unserved_cars[sorted_neighbors[n]] -= 1
                                    capacity -= 1
                            
                            difference = capacity / (k+1)
                            for m in range((k+1)):
                                unserved_cars[sorted_neighbors[m]] -= difference
                                capacity -= difference
                else:
                    unserved_cars[sorted_neighbors[0]] -= capacity
                    capacity = 0
                #unserved_cars_neighbors = {index: unserved_cars[index] for index in neighbors}
                #print("Updated unserved cars neighbors:", unserved_cars_neighbors)
                #print("Remaining capacity:", capacity)
    #print(unserved_cars)
    sorted_unserved_cars = sorted(unserved_cars.items(), key=lambda item: item[1], reverse=True)
    #print(sorted_unserved_cars)
    #print(sorted_unserved_cars)
    diff = 0
    capacity = 0
    L2_ratio = L2_cost/L2_cap
    DC_ratio = DC_cost/DC_cap
    cost = 0
    cap = 0
    bool = False
    if DC_ratio <= L2_ratio:
        cost = DC_cost
        cap = DC_cap
        bool = True
    else:
        bool = False
        cost = L2_cost
        cap = L2_cap
    while budget>0 and (budget>=L2_cost or budget>=DC_cost):
        if unserved_cars[sorted_unserved_cars[0][0]]>=cap and budget>=cost and bool == False or (bool== True and budget<DC_cost and budget >=L2_cost) :
            unserved_cars[sorted_unserved_cars[0][0]]-=L2_cap
            budget-=L2_cost
            L2_built[sorted_unserved_cars[0][0]]+=1
            #print(budget)
        if (unserved_cars[sorted_unserved_cars[0][0]]<=DC_cap or bool == True) and budget>=DC_cost: 
            m, n = sorted_unserved_cars[0][0]
            neighbors = get_neighbors(m,n, y, x)
            neighbors.append(sorted_unserved_cars[0][0])
            sorted_neighbors = sorted(neighbors, key=lambda coord: unserved_cars[coord], reverse=True)

            if unserved_cars[sorted_neighbors[0]]-unserved_cars[sorted_neighbors[1]]>=DC_cap:
                #print("a")
                unserved_cars[sorted_neighbors[0]]=-DC_cap
            else:
                #print("s")
                #print(m,n,neighbors,budget)
                #unserved_cars_neighbors = {}
                #for index in neighbors:
                #    unserved_cars_neighbors[index]=unserved_cars[index]
                unserved_cars_neighbors = {index: unserved_cars[index] for index in neighbors}
                #print(unserved_cars_neighbors)
                diff = unserved_cars[sorted_neighbors[0]] - unserved_cars[sorted_neighbors[1]]
                capacity = DC_cap - diff
                #print(capacity)
                if diff < capacity:
                    unserved_cars[sorted_neighbors[0]] -= diff
                    capacity -= diff
                    
                    for k in range(2, len(sorted_neighbors)):
                        diff = unserved_cars[sorted_neighbors[0]] - unserved_cars[sorted_neighbors[k]]
                        if diff==0 and k+1<len(sorted_neighbors) and unserved_cars[sorted_neighbors[k+1]]==0:
                            continue
                        if diff * k < capacity and diff >0:
                                for m in range(k):
                                    unserved_cars[sorted_neighbors[m]] -= diff
                                    capacity -= diff
                        else:
                            #print(k,capacity)
                            if capacity % k != 0:
                                for n in range(int(capacity % k)):
                                    unserved_cars[sorted_neighbors[n]] -= 1
                                    capacity -= 1
                            
                            difference = capacity / k
                            for m in range(k):
                                unserved_cars[sorted_neighbors[m]] -= difference
                                capacity -= difference
                else:
                    unserved_cars[sorted_neighbors[0]] -= capacity
                    capacity = 0
                #unserved_cars_neighbors = {}
                unserved_cars_neighbors = {index: unserved_cars[index] for index in neighbors}
                #print(unserved_cars_neighbors)
            budget-=DC_cost
            DC_built[sorted_unserved_cars[0][0]]+=1
            #neighbors.append(sorted_unserved_cars[0])
        sorted_unserved_cars = sorted(unserved_cars.items(), key=lambda item: item[1], reverse=True)
        print(budget)
            
    #print(sorted_unserved_cars)
    L2_sorted = sorted(L2_built.items(), key=lambda item: item[1], reverse=True)
    DC_sorted = sorted(DC_built.items(), key=lambda item: item[1], reverse=False)
    total_L2 = sum(value for key, value in L2_sorted)
    print("Total sum of L2 chargers built:", total_L2)
    total_DC = sum(value for key, value in DC_sorted)
    print("Total sum of DC chargers built:", total_DC)
    #print(DC_sorted)
    #print(budget)
if __name__ == "__main__":
    main()