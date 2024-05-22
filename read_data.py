import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def find_block(coordinate, north_east, south_west, grid_size):
    """
    Determine which block a coordinate is in within a specified grid.

    Args:
    coordinate (tuple): The (longitude, latitude) of the point.
    north_east (tuple): The (longitude, latitude) of the most northern east point.
    south_west (tuple): The (longitude, latitude) of the most southern west point.
    rows (int): The number of rows in the grid.
    cols (int): The number of columns in the grid.

    Returns:
    tuple: The (row, column) of the block the coordinate is in.
    """
    lon, lat = coordinate
    rows, cols = grid_size
    ne_lon, ne_lat = north_east
    sw_lon, sw_lat = south_west
    
    # Calculate the width and height of each block
    block_width = (ne_lon - sw_lon) / cols
    block_height = (ne_lat - sw_lat) / rows
    
    # Calculate the column (x) and row (y) index of the coordinate
    col = int((lon - sw_lon) / block_width)
    row = int((lat - sw_lat) / block_height)
    #print(row, col)
    
    # Ensure the coordinates are within bounds
    if col < 0 or col >= cols or row < 0 or row >= rows:
        raise ValueError(f"Coordinate ({lon},{lat}) is out of the grid bounds.")
    
    return row, col


def neighbour_adj_matrix(rows: int, columns: int):
    def get_neighbors(index, m, n):
        """Get the list of neighbors for a given index in an m x n grid."""
        neighbors = []
        row, col = divmod(index, n)

        # Up
        if row > 0:
            neighbors.append(index - n)
        # Down
        if row < m - 1:
            neighbors.append(index + n)
        # Left
        if col > 0:
            neighbors.append(index - 1)
        # Right
        if col < n - 1:
            neighbors.append(index + 1)

        return neighbors

    """Calculate the adjacency matrix for an m x n grid."""
    size = rows * columns
    adjacency_matrix = np.zeros((size, size), dtype=int)

    for index in range(size):
        neighbors = get_neighbors(index, rows, columns)
        for neighbor in neighbors:
            adjacency_matrix[index][neighbor] = 1
            adjacency_matrix[neighbor][index] = 1  # Because the matrix is symmetric
    print(adjacency_matrix)
    return adjacency_matrix
    adj = [[0 for j in range(columns)] for i in range(rows)]
    for i in range(rows):
        for j in range(columns):
            adj[i][j] = abs(j - i)

def main():
    # Example usage:
    most_eastern_north_point = (-116.9164, 49.0025) # is (15, 23)
    most_western_south_point = (-124.7333, 45.5434) # is (0,0)
    coordinate = (-116.9165, 49.0024) #(-124.7333, 45.5434) #(-122.30839, 47.610365)
    grid_size = (64, 96)
    y, x = grid_size

    df = pd.read_csv("Data/Electric_Vehicle_Population_Data.csv")
    # data cleaning
    filtered_df = df[df['State'] == 'WA']
    filtered_df = filtered_df[filtered_df['Vehicle Location'].notna()]

    neighbour_adj_matrix(y, x)

def read_data():
    # Example usage:
    most_eastern_north_point = (-116.9164, 49.0025) # is (15, 23)
    most_western_south_point = (-124.7333, 45.5434) # is (0,0)
    coordinate = (-116.9165, 49.0024) #(-124.7333, 45.5434) #(-122.30839, 47.610365)
    grid_size = (64, 96)
    y, x = grid_size

    df = pd.read_csv("Data/Electric_Vehicle_Population_Data.csv")
    filtered_df = df[df['State'] == 'WA']
    filtered_df = filtered_df[filtered_df['Vehicle Location'].notna()]

    cars = {}
    for i in range(y):
        for j in range(x):
            cars[(i, j)] = 0

    for index, location in filtered_df['Vehicle Location'].items():
        #print(location, type(location))
        location_str = location.replace('POINT ', '').strip('()')
        lon_str, lat_str = location_str.split()
        block = find_block((float(lon_str), float(lat_str)), most_eastern_north_point, most_western_south_point, grid_size)
        cars[block] += 1

    districts = [cars[(i, j)] for j in range(x) for i in range(y)]
    
    for i in range(y):
        for j in range(x):
            if cars[(i, j)] == 509:
                print(i, j, cars[(i, j)])

    #for i, data in enumerate(districts):
    #    print(i, data)

    """ for i in np.arange(y - 1, -1, -1):
        for j in range(x):
            print(f"Block ({i},{j}): {cars[(i, j)]}")
    
    matrix = np.zeros(grid_size)

    for (i, j), value in cars.items():
        matrix[i, j] = value

    plt.figure(figsize=(12, 8))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Number of Cars')
    plt.title('Car Distribution in Grid')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index') 
    plt.xticks(np.arange(x - 1))
    plt.yticks(np.arange(y - 1, -1, -1))
    plt.grid(False)
    plt.show() """




if __name__ == '__main__':
    read_data()
    main()
