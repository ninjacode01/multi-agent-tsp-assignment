# =========================================================
# MULTI-AGENT ROUTE OPTIMIZATION
# Hybrid Heuristic + OSRM + Balanced Workload
# =========================================================

import pandas as pd
import numpy as np
import requests
import folium
import random
import time

from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# =========================================================
# CONFIG
# =========================================================

FILE_PATH = "../TSP1.xlsx"

OSRM_URL = "http://router.project-osrm.org"

MAX_BALANCE_ITERATIONS = 15
MAX_2OPT_ITERATIONS = 100

# =========================================================
# LOAD DATA
# =========================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

locations_df = pd.read_excel(
    FILE_PATH,
    sheet_name="Lat-Long"
)

agents_df = pd.read_excel(
    FILE_PATH,
    sheet_name="TSP agents"
)

locations_df = locations_df.dropna(
    subset=["Latitude", "Longitude"]
).reset_index(drop=True)

NUM_AGENTS = len(agents_df)

coords = locations_df[
    ["Latitude", "Longitude"]
].values

print(f"Locations : {len(locations_df)}")
print(f"Agents    : {NUM_AGENTS}")

# =========================================================
# INITIAL CLUSTERING
# =========================================================

print("\n" + "=" * 60)
print("CREATING INITIAL CLUSTERS")
print("=" * 60)

kmeans = KMeans(
    n_clusters=NUM_AGENTS,
    random_state=42,
    n_init=20
)

locations_df["cluster"] = kmeans.fit_predict(coords)

# =========================================================
# SAFE OSRM DISTANCE MATRIX
# =========================================================

def get_osrm_distance_matrix(
    cluster_points,
    max_retries=5,
    sleep_time=2
):

    n = len(cluster_points)

    matrix = np.zeros((n, n))

    BATCH_SIZE = 25

    for i_start in tqdm(
        range(0, n, BATCH_SIZE),
        desc="OSRM Requests",
        leave=False,
        colour="yellow"
    ):

        i_end = min(i_start + BATCH_SIZE, n)

        sources = list(
            range(i_start, i_end)
        )

        destinations = list(
            range(n)
        )

        coords_str = ";".join(
            [
                f"{lon},{lat}"
                for lat, lon in cluster_points
            ]
        )

        source_str = ";".join(
            map(str, sources)
        )

        dest_str = ";".join(
            map(str, destinations)
        )

        url = (
            f"{OSRM_URL}/table/v1/driving/"
            f"{coords_str}"
            f"?sources={source_str}"
            f"&destinations={dest_str}"
            f"&annotations=distance"
        )

        success = False

        for attempt in range(max_retries):

            try:

                response = requests.get(
                    url,
                    timeout=60
                )

                if response.status_code == 200:

                    data = response.json()

                    distances = np.array(
                        data["distances"]
                    ) / 1000.0

                    matrix[
                        i_start:i_end,
                        :
                    ] = distances

                    success = True

                    break

                else:

                    print(
                        f"\nOSRM Error "
                        f"{response.status_code}"
                    )

            except Exception as e:

                print(
                    f"\nRetry {attempt+1}: {e}"
                )

            time.sleep(sleep_time)

        if not success:

            raise Exception(
                "OSRM request failed "
                "after retries"
            )

        time.sleep(1.5)

    return matrix

# =========================================================
# GREEDY ROUTE CONSTRUCTION
# =========================================================

def nearest_neighbor_route(distance_matrix):

    n = len(distance_matrix)

    unvisited = set(range(1, n))

    route = [0]

    while unvisited:

        last = route[-1]

        next_city = min(
            unvisited,
            key=lambda x: distance_matrix[last][x]
        )

        route.append(next_city)

        unvisited.remove(next_city)

    route.append(0)

    return route

# =========================================================
# ROUTE DISTANCE
# =========================================================

def compute_route_distance(route, matrix):

    total = 0

    for i in range(len(route) - 1):

        total += matrix[
            route[i]
        ][
            route[i + 1]
        ]

    return total

# =========================================================
# 2-OPT OPTIMIZATION
# =========================================================

def two_opt(route, matrix):

    best = route

    best_distance = compute_route_distance(
        best,
        matrix
    )

    improved = True

    iteration = 0

    with tqdm(
        total=MAX_2OPT_ITERATIONS,
        desc="2-Opt",
        leave=False,
        colour="green"
    ) as pbar:

        while improved and iteration < MAX_2OPT_ITERATIONS:

            improved = False

            for i in range(1, len(best) - 2):

                for j in range(i + 1, len(best) - 1):

                    if j - i == 1:
                        continue

                    new_route = (
                        best[:i]
                        + best[i:j][::-1]
                        + best[j:]
                    )

                    new_distance = compute_route_distance(
                        new_route,
                        matrix
                    )

                    if new_distance < best_distance:

                        best = new_route
                        best_distance = new_distance
                        improved = True

            iteration += 1
            pbar.update(1)

    return best, best_distance

# =========================================================
# BALANCE CLUSTERS
# =========================================================

def balance_clusters(df):

    print("\n" + "=" * 60)
    print("BALANCING CLUSTERS")
    print("=" * 60)

    target = len(df) / NUM_AGENTS

    for iteration in tqdm(
        range(MAX_BALANCE_ITERATIONS),
        desc="Balancing",
        colour="cyan"
    ):

        cluster_sizes = (
            df["cluster"]
            .value_counts()
            .to_dict()
        )

        over_clusters = [
            c for c, size in cluster_sizes.items()
            if size > target + 1
        ]

        under_clusters = [
            c for c, size in cluster_sizes.items()
            if size < target - 1
        ]

        if not over_clusters or not under_clusters:
            break

        for over in over_clusters:

            over_points = df[
                df["cluster"] == over
            ]

            centroid_over = (
                over_points[
                    ["Latitude", "Longitude"]
                ]
                .mean()
                .values
            )

            distances = cdist(
                over_points[
                    ["Latitude", "Longitude"]
                ],
                [centroid_over]
            ).flatten()

            farthest_idx = over_points.index[
                np.argmax(distances)
            ]

            candidate = df.loc[farthest_idx]

            best_cluster = None
            best_distance = float("inf")

            for under in under_clusters:

                under_points = df[
                    df["cluster"] == under
                ]

                centroid_under = (
                    under_points[
                        ["Latitude", "Longitude"]
                    ]
                    .mean()
                    .values
                )

                d = np.linalg.norm(
                    candidate[
                        ["Latitude", "Longitude"]
                    ].values - centroid_under
                )

                if d < best_distance:

                    best_distance = d
                    best_cluster = under

            if best_cluster is not None:

                df.at[
                    farthest_idx,
                    "cluster"
                ] = best_cluster

    return df

locations_df = balance_clusters(
    locations_df
)

# =========================================================
# CREATE MAP
# =========================================================

center_lat = locations_df[
    "Latitude"
].mean()

center_lon = locations_df[
    "Longitude"
].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10
)

# =========================================================
# RANDOM COLORS
# =========================================================

def random_color():

    return "#%06x" % random.randint(
        0,
        0xFFFFFF
    )

# =========================================================
# PROCESS AGENTS
# =========================================================

print("\n" + "=" * 60)
print("STARTING ROUTE OPTIMIZATION")
print("=" * 60)

results = []

all_agent_routes = {}

agent_progress = tqdm(
    range(NUM_AGENTS),
    desc="Agents",
    colour="magenta"
)

for cluster_id in agent_progress:

    cluster_df = locations_df[
        locations_df["cluster"] == cluster_id
    ].reset_index(drop=True)

    if len(cluster_df) <= 2:
        continue

    agent_name = agents_df.iloc[
        cluster_id
    ]["Agent Name"]

    agent_progress.set_postfix_str(
        f"{agent_name}"
    )

    cluster_points = cluster_df[
        ["Latitude", "Longitude"]
    ].values

    # =====================================================
    # DISTANCE MATRIX
    # =====================================================

    matrix = get_osrm_distance_matrix(
        cluster_points
    )

    # =====================================================
    # INITIAL ROUTE
    # =====================================================

    initial_route = nearest_neighbor_route(
        matrix
    )

    # =====================================================
    # OPTIMIZATION
    # =====================================================

    optimized_route, total_distance = two_opt(
        initial_route,
        matrix
    )

    # =====================================================
    # BUILD ROUTE
    # =====================================================

    route_coords = []

    ordered_rows = []

    for stop_number, idx in enumerate(
        optimized_route[:-1],
        start=1
    ):

        row = cluster_df.iloc[idx].copy()

        row["Visit_Order"] = stop_number

        ordered_rows.append(row)

        lat = cluster_points[idx][0]
        lon = cluster_points[idx][1]

        route_coords.append(
            (lat, lon)
        )

    route_coords.append(route_coords[0])

    ordered_df = pd.DataFrame(
        ordered_rows
    )

    all_agent_routes[
        f"Agent_{cluster_id+1}"
    ] = {
        "agent_name": agent_name,
        "data": ordered_df
    }

    # =====================================================
    # MAP
    # =====================================================

    color = random_color()

    folium.PolyLine(
        route_coords,
        color=color,
        weight=4,
        opacity=0.8,
        tooltip=(
            f"{agent_name}"
            f" | {total_distance:.2f} km"
        )
    ).add_to(m)

    for lat, lon in route_coords:

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            fill=True
        ).add_to(m)

    # =====================================================
    # SUMMARY
    # =====================================================

    results.append({
        "Agent": agent_name,
        "Stops": len(cluster_points),
        "Distance_KM": round(
            total_distance,
            2
        )
    })

# =========================================================
# SUMMARY DF
# =========================================================

results_df = pd.DataFrame(results)

# =========================================================
# SAVE SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("SAVING SUMMARY")
print("=" * 60)

results_df.to_excel(
    "agent_route_summary.xlsx",
    index=False
)

results_df.to_csv(
    "agent_route_summary.csv",
    index=False
)

# =========================================================
# SAVE FULL ROUTES WORKBOOK
# =========================================================

print("\n" + "=" * 60)
print("SAVING FULL ROUTES")
print("=" * 60)

with pd.ExcelWriter(
    "full_agent_routes.xlsx",
    engine="openpyxl"
) as writer:

    for sheet_name, info in all_agent_routes.items():

        agent_name = info["agent_name"]

        df_sheet = info["data"]

        # write dataframe starting lower
        df_sheet.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=2,
            index=False
        )

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # header
        worksheet["A1"] = f"Agent Name: {agent_name}"

# =========================================================
# SAVE MAP
# =========================================================

m.save(
    "multi_agent_tsp_map.html"
)

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("OPTIMIZATION COMPLETE")
print("=" * 60)

print(results_df)

print("\nGenerated Files:")
print("• agent_route_summary.xlsx")
print("• agent_route_summary.csv")
print("• full_agent_routes.xlsx")
print("• multi_agent_tsp_map.html")