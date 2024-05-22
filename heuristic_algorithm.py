from read_data import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
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
    DC_cost = 10000
    budget = 9164000
    unserved_cars = {}
    #First process preexisting L2 chargers
    for i in range(y):
        for j in range(x):
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
                                print("y")
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
                unserved_cars_neighbors = {index: unserved_cars[index] for index in neighbors}
                #print("Updated unserved cars neighbors:", unserved_cars_neighbors)
                print("Remaining capacity:", capacity)
    '''
    for i in range(y):
        for j in range(x):
            if DC[(i,j)]>0:
                capacity = DC[(i,j)]*DC_cap
                print(" ")
                print(capacity,i,j)
                neighbors = get_neighbors(i,j,y,x)
                neighbors.append((i,j))
                print(neighbors)
                sorted_neighbors = sorted(neighbors, key=lambda coord: unserved_cars[coord], reverse=True)
                unserved_cars_neighbors = {}
                for index in neighbors:
                    unserved_cars_neighbors[index]=unserved_cars[index]
                print(unserved_cars_neighbors)
                print(sorted_neighbors)
                diff = unserved_cars[sorted_neighbors[0]] - unserved_cars[sorted_neighbors[1]]
                if diff < capacity:
                    unserved_cars[sorted_neighbors[0]]= unserved_cars[sorted_neighbors[0]] - diff
                    capacity = capacity - diff
                    for j in range(2,4):
                        if (unserved_cars[sorted_neighbors[0]] - unserved_cars[sorted_neighbors[j]])*j<capacity:
                            print("yep")
                            difference = unserved_cars[sorted_neighbors[0]] - unserved_cars[sorted_neighbors[j]]
                            for i in range(0,j-1):
                                unserved_cars[sorted_neighbors[i]]= unserved_cars[sorted_neighbors[i]] - difference
                                capacity = capacity - difference
                        else:
                            if capacity % j !=0:
                                unserved_cars[sorted_neighbors[0]]= unserved_cars[sorted_neighbors[0]] - capacity % j
                                capacity = capacity - capacity % j
                            
                            difference = capacity / j
                            for i in range(0,j-1):
                                unserved_cars[sorted_neighbors[i]]= unserved_cars[sorted_neighbors[i]] - difference
                                capacity = capacity - difference       
                else:
                    unserved_cars[sorted_neighbors[0]]= unserved_cars[sorted_neighbors[0]] - capacity
                    capacity = 0
                for index in neighbors:
                    unserved_cars_neighbors[index]=unserved_cars[index]
                print(unserved_cars_neighbors) 
                print(capacity)
            '''
    '''
                unserved_cars_neighbors = {}
                value = unserved_cars[(i,j)]
                biggest_index = (i,j)
                difference = 0
                small_diff =0
                for index in neighbors:
                    unserved_cars_neighbors[index]=unserved_cars[index]
                print(unserved_cars_neighbors)
                while capacity != 0:
                    for index in neighbors:
                        if unserved_cars[(i,j)] - unserved_cars[index] > small_diff:
                            small_diff = unserved_cars[(i,j)] - unserved_cars[index]
                        if unserved_cars[index] > unserved_cars[biggest_index]:
                            difference = unserved_cars[index] - unserved_cars[biggest_index]
                            biggest_index = index
                    if difference !=0:
                        if difference >=capacity:
                            unserved_cars[biggest_index]=unserved_cars[biggest_index]-capacity
                            capacity = 0
                            #if neighbors[index]
                        else:
                            unserved_cars[biggest_index]=unserved_cars[biggest_index]-difference
                            capacity = capacity - difference
                    else:
                        #(i,j) has biggest number of unused cars
                        if small_diff >=capacity:
                            unserved_cars[(i,j)]=unserved_cars[(i,j)] - capacity
                            capacity =0
                        else:
                            if small_diff == 0:
                                unserved_cars[(i,j)]=unserved_cars[(i,j)] - 1
                                capacity = capacity -1
                            else:
                                unserved_cars[(i,j)]=unserved_cars[(i,j)] - small_diff
                                capacity = capacity -small_diff
                print(difference, small_diff)
                for index in neighbors:
                    unserved_cars_neighbors[index]=unserved_cars[index]
                print(biggest_index)
                print(difference)
                print(unserved_cars[(i,j)],value)
                print(unserved_cars_neighbors)
                '''
    #print(unserved_cars)
    sorted_unserved_cars = sorted(unserved_cars.items(), key=lambda item: item[1], reverse=True)
    #print(sorted_unserved_cars)
    
if __name__ == "__main__":
    main()