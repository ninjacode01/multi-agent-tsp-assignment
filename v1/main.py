import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from math import radians, sin, cos, sqrt, atan2
import folium
import random
from tqdm.auto import tqdm

FILE_PATH = "../TSP1.xlsx"

print("LOADING DATASETS")

locations_df = pd.read_excel(FILE_PATH, sheet_name="Lat-Long").dropna(subset=["Latitude", "Longitude"])
agents_df = pd.read_excel(FILE_PATH, sheet_name="TSP agents")

NUM_AGENTS = len(agents_df)
coords = locations_df[["Latitude", "Longitude"]].values

print(f"Locations Loaded : {len(locations_df)}")
print(f"Agents Loaded    : {NUM_AGENTS}")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

print("CREATING CLUSTERS")

kmeans = KMeans(n_clusters=NUM_AGENTS, random_state=12, n_init=20)

locations_df["cluster"] = kmeans.fit_predict(coords)

print("Cluster assignment completed")

def build_distance_matrix(cluster_points):
    n = len(cluster_points)
    matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Distance Matrix", leave=False, colour="cyan"):

        for j in range(n):

            if i != j:
                matrix[i][j] = haversine(
                    cluster_points[i][0],
                    cluster_points[i][1],
                    cluster_points[j][0],
                    cluster_points[j][1]
                )

    return matrix

def solve_tsp(distance_matrix):

    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return int(distance_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    with tqdm(total=1, desc="Solving Route", leave=False, colour="green") as pbar:
        solution = routing.SolveWithParameters(search_parameters)
        pbar.update(1)

    if solution is None:
        return None, None
    
    index = routing.Start(0)

    route = []
    route_distance = 0

    with tqdm(desc="Extracting Route", leave=False, colour="yellow") as pbar:

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            pbar.update(1)

    return route, route_distance / 1000

def random_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)

center_lat = locations_df["Latitude"].mean()
center_lon = locations_df["Longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

results = []
agent_routes = {}

print("STARTING MULTI-AGENT OPTIMIZATION")

overall_pbar = tqdm(range(NUM_AGENTS), desc="Overall Progress", colour="magenta")

for cluster_id in overall_pbar:

    cluster_df = locations_df[locations_df["cluster"] == cluster_id].reset_index(drop=True)
    cluster_points = cluster_df[["Latitude", "Longitude"]].values
    agent_name = agents_df.iloc[cluster_id]["Agent Name"]
    overall_pbar.set_postfix_str(f"Agent: {agent_name}")

    if len(cluster_points) <= 1:
        continue

    distance_matrix = build_distance_matrix(cluster_points)
    route, total_distance = solve_tsp(distance_matrix)
    if route is None:
        continue

    route_coords = []
    ordered_rows = []

    for stop_no, idx in enumerate(
        tqdm(route, desc=f"Rendering {agent_name}", leave=False, colour="blue"),
        start=1
    ):

        lat, lon = cluster_points[idx][0], cluster_points[idx][1]
        route_coords.append((lat, lon))
        row_data = cluster_df.iloc[idx].copy()
        row_data["Stop_No"] = stop_no
        ordered_rows.append(row_data)

    route_coords.append(route_coords[0])
    color = random_color()

    folium.PolyLine(
        route_coords,
        color=color,
        weight=4,
        opacity=0.8,
        tooltip=f"{agent_name} | {total_distance:.2f} km"
    ).add_to(m)

    for lat, lon in route_coords:

        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color=color,
            fill=True
        ).add_to(m)

    results.append({
        "Agent": agent_name,
        "Stops": len(cluster_points),
        "Distance_KM": round(total_distance, 2)
    })

    agent_routes[agent_name] = pd.DataFrame(ordered_rows)

print("SAVING OUTPUT FILES")

results_df = pd.DataFrame(results)

with pd.ExcelWriter("multi_agent_routes.xlsx", engine="openpyxl") as writer:

    results_df.to_excel(writer, sheet_name="Summary", index=False)

    for agent_name, df in agent_routes.items():
        sheet_name = str(agent_name)[:31]
        start_row = 2
        workbook = writer.book

        df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
        worksheet = writer.sheets[sheet_name]
        worksheet.cell(row=1, column=1).value = f"Agent Name : {agent_name}"

m.save("multi_agent_tsp_map.html")

print("\nOptimization Completed Successfully")

print(results_df)

print("\nGenerated Files:")
print("• multi_agent_routes.xlsx")
print("• multi_agent_tsp_map.html")