import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gurobipy as gp
import json
import os
from gurobipy import GRB, Model

def find_block(coordinate: tuple, north_east: tuple, south_west: tuple, grid_size: tuple):
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
    col = (lon - sw_lon) // block_width # '//' cuts the values after comma, so 2.67 => 2
    row = (lat - sw_lat) // block_height
    
    # Ensure the coordinates are within bounds
    if col < 0 or col >= cols or row < 0 or row >= rows:
        raise ValueError(f"Coordinate ({lon},{lat}) is out of the grid bounds.")
    
    return row, col

def get_district(pos: tuple, size: tuple):
    return int(pos[0] * size[1] + pos[1] + 1)

def get_tuple(district: int, size: tuple):
    district -= 1
    return (district // size[1], district % size[1])

def print_data(cars, ecs, grid_size: tuple, filename='new_cars_ecs_distribution.png'):
    y, x = grid_size
    matrix_l2_cars = np.zeros(grid_size)
    matrix_dc_cars = np.zeros(grid_size)
    matrix_l2_ecs = np.zeros(grid_size)
    matrix_dc_ecs = np.zeros(grid_size)

    for (i, type), value in cars.items():
        if type == 1:  # Level 2
            matrix_l2_cars[get_tuple(i, grid_size)] = value
        elif type == 2:  # DC chargers
            matrix_dc_cars[get_tuple(i, grid_size)] = value

    for (i, type), value in ecs.items():
        if type == 1:  # Level 2
            matrix_l2_ecs[get_tuple(i, grid_size)] = value
        elif type == 2:  # DC chargers
            matrix_dc_ecs[get_tuple(i, grid_size)] = value

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    data = [
        (matrix_l2_cars, 'Cars covered by Level 2 charging station distribution', 'Number of cars covered by Level 2'),
        (matrix_dc_cars, 'Cars covered by DC charging station distribution', 'Number of cars covered by DC'),
        (matrix_l2_ecs, 'Level 2 charging station distribution', 'Number of Level 2 charging stations'),
        (matrix_dc_ecs, 'DC charging station distribution', 'Number of DC charging stations')
    ]

    for i in range(2):
        for j in range(2):
            matrix, title, colorbar_label = data[i * 2 + j]
            im = axs[i, j].imshow(matrix, cmap='viridis', interpolation='nearest')
            fig.colorbar(im, ax=axs[i, j], label=colorbar_label)
            axs[i, j].set_title(title)
            axs[i, j].set_xlabel('Column Index')
            axs[i, j].set_ylabel('Row Index') 
            axs[i, j].set_xticks(np.arange(x))
            axs[i, j].set_yticks(np.arange(y))
            axs[i, j].invert_yaxis()
            axs[i, j].grid(False)

    plt.tight_layout()
    # Save the plot to a file
    plt.savefig(filename)
    plt.show()

def safe_data(cars: dict, ecs: dict):
    new_cars = {str(key): value for key, value in cars.items()}
    new_ecs = {str(key): value for key, value in ecs.items()}
    with open('new_cars_distribution.json', 'w') as json_file:
        json.dump(new_cars, json_file)

    with open('new_ecs_distribution.json', 'w') as json_file:
        json.dump(new_ecs, json_file)

def read_data(grid_size: tuple, N: list):
    most_eastern_north_point = (-116.9164, 49.0025) # is (63, 95)
    most_western_south_point = (-124.7333, 45.5434) # is (0,0)

    df_cars = pd.read_csv("Data/Electric_Vehicle_Population_Data.csv")
    df_ecs = pd.read_csv("Data/alt_fuel_stations.csv")
    filtered_df_cars = df_cars[df_cars['State'] == 'WA']
    filtered_df_ecs = df_ecs[df_ecs['State'] == 'WA']
    filtered_df_cars = filtered_df_cars[filtered_df_cars['Vehicle Location'].notna()]
    filtered_df_ecs = filtered_df_ecs[filtered_df_ecs['Latitude'].notna() & filtered_df_ecs['Longitude'].notna()]

    y, x = grid_size
    I = range(1, y * x + 1)

    # cars in district i
    cars = {i: 0 for i in I}
    # electric charging stations in district i for type n
    ecs = {(i, n): 0 for i in I for n in N}

    # count number of cars for every block in the grid
    for index, location in filtered_df_cars['Vehicle Location'].items():
        location_str = location.replace('POINT ', '').strip('()')
        lon_str, lat_str = location_str.split()
        block = find_block((float(lon_str), float(lat_str)), most_eastern_north_point, most_western_south_point, grid_size)
        cars[get_district(block, grid_size)] += 1

    # count number of electric charging stations for every block in the grid
    for index, row in filtered_df_ecs.iterrows():
        lon = row['Longitude']
        lat = row['Latitude']
        lv2_charger = row['EV Level2 EVSE Num']
        dc_charger = row['EV DC Fast Count']
        block = find_block((lon, lat), most_eastern_north_point, most_western_south_point, grid_size)
        ecs[get_district(block, grid_size), 1] += int(lv2_charger) if not pd.isna(lv2_charger) else 0
        ecs[get_district(block, grid_size), 2] += int(dc_charger) if not pd.isna(dc_charger) else 0
    
    return cars, ecs

def gurobi():
    options = {
        "WLSACCESSID": "3edcfa2d-90a4-4606-8577-b824b7292e8c",
        "WLSSECRET": "17112247-3115-4ddd-a2c0-2f13e6351db6",
        "LICENSEID": 2508491
    }
    with gp.Env(params=options) as env, Model(env=env) as model:
        # Get the number of available CPU threads
        num_threads = os.cpu_count()
        # Set the Threads parameter
        model.setParam(GRB.Param.Threads, num_threads)

        grid_size = (32, 48)
        Y, X = grid_size
        print(model.getParamInfo(GRB.Param.Threads))

        # Define sets and parameters
        B = 9164000 # available budget
        N = [1, 2]  # Charger types: 1 (Level Two Charger), 2 (DC Charger)
        C = {1: 2000, 2: 10000} # Costs for the charger types
        C_n = {1: 3, 2: 16}  # Number of cars each charger can cover for types 1 and 2, respectively
        I = range(1, X * Y + 1) # number of districts
        V_i, D_in = read_data(grid_size, N) # Vi amount of cars, Di amount of chargers
        Z_ij = {(i, j): 1 if abs(i - j) == 1 else 0 for i in I for j in I} # adjacents matrix for neighbours in district

        lv2 = 0
        dc = 0
        for (i, n), value in D_in.items():
            if n == 1:
                lv2 += value
            else:
                dc += value
        print(f"Number of level2 chargers: {lv2}\nNumber of DC chargers: {dc}")
        print_data({(i, n): value for n in N for i, value in V_i.items()}, D_in, grid_size, filename='old_cars_ecs_distribution.png')

        x = model.addVars(I, N, vtype=GRB.CONTINUOUS, name="x")  # Number of cars covered by charger n in district i
        y = model.addVars(I, N, vtype=GRB.CONTINUOUS, name="y")  # Number of chargers of type n added to district i
        m = model.addVars(I, I, vtype=GRB.CONTINUOUS, name="m")  # Number of cars moved from district i to district j

        # Objective function
        model.setObjective(gp.quicksum(x[i, n] for i in I for n in N), GRB.MAXIMIZE)

        # Constraints
        # Total number of cars covered in each district for level2 charger
        model.addConstrs((x[i, 1] <= V_i[i] - gp.quicksum(m[i, j] * Z_ij[i, j] for j in I) for i in I), name="level2_in_out")

        # Total number of cars covered in each district for DC charger
        model.addConstrs((x[i, 2] <= V_i[i] + gp.quicksum((-m[i, j] + m[j, i]) * Z_ij[i, j] for j in I) for i in I), name="dc_in_out")

        # To dont count moved cars twice
        model.addConstrs((gp.quicksum(x[i, n] for n in N) <= V_i[i] + gp.quicksum(Z_ij[i, j] * (-m[i, j] + m[j, i]) for j in I) for i in I), name="coverage_total")

        # Number of cars covered by Level Two chargers
        model.addConstrs((x[i, 1] <= C_n[1] * (D_in[i, 1] + y[i, 1]) for i in I), name="level2_coverage")

        # Number of cars covered by DC chargers
        model.addConstrs((x[i, 2] <= C_n[2] * (D_in[i, 2] + y[i, 2]) for i in I), name="dc_coverage")

        # Budget constraint
        model.addConstr(gp.quicksum(y[i, 1] * C[1] + y[i, 2] * C[2] for i in I) <= B, name="budget")

        # Optimize model
        model.optimize()

        # create diagrams for resulted data for level2
        cars = {}
        ecs = {}
        for n in N:
            for i in I:
                cars[i, n] = x[i, n].X
                ecs[i, n] = y[i, n].X
        print_data(cars, ecs, grid_size)
        safe_data(cars, ecs)

        # print data that is not 0 to terminal
        for i in I:
            for n in N:
                if x[i, n].X > 0:
                    print(f"x[{i},{n}] = {round(x[i, n].X)}")
        for i in I:
            for n in N:
                if y[i, n].X > 0:
                    print(f"y[{i},{n}] = {y[i, n].X}")
        for i in I:
            for j in I:
                if m[i, j].X > 0:
                    print(f"m[{i},{j}] = {m[i, j].X}")

if __name__ == '__main__':
    gurobi()
