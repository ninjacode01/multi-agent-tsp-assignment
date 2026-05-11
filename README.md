# Multi-Agent TSP Assignment

This project solves a Multi-Agent Traveling Salesman Problem (TSP) using two different approaches.

---

# Project Versions

## V1 — ORTools + Haversine Distance

Location:
```text
v1/
```
Features:
- KMeans clustering
- Haversine distance calculation
- Google ORTools optimization
- Basic multi-agent route optimization
- Folium visualization

This version focuses on:
- simpler implementation,
- faster execution,
- solver-based optimization.

---

## V2 — Heuristic + Real Road Distances

Location:
```text
v2_balanced/
```

Features:
- Balanced KMeans clustering
- Custom cluster balancing heuristic
- Real road distances using OSRM
- Nearest Neighbor route construction
- 2-Opt route optimization
- Real road-path visualization
- Workload balancing across agents

This version focuses on:
- balanced agent workloads,
- realistic driving distances,
- explainable heuristic optimization,
- road-network visualization.

---

# Project Structure

```text
MULTI-AGENT-TSP-ASSIGNMENT/

│── v1/
│   │── main.py
│   │── multi_agent_tsp_map.html
│   │── multi_agent_routes.xlsx

│── v2_balanced/
│   │── main.py
│   │── visualization.py
│   │── full_agent_routes.xlsx
│   │── agent_route_summary.csv
│   │── multi_agent_tsp_map.html
│   │── road_route_visualization.html

│── TSP1.xlsx
│── requirements.txt
│── README.md
│── LICENSE
```

---

# Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

```bash
.venv\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

---

# Run V1

```bash
cd v1
python main.py
```

Outputs:
```text
multi_agent_routes.xlsx
multi_agent_tsp_map.html
```

---

# Run V2

```bash
cd v2_balanced
python main.py
```

Outputs:
```text
agent_route_summary.csv
full_agent_routes.xlsx
multi_agent_tsp_map.html
```

---

# Generate Real Road Visualization
Inside `v2_balanced/`:
```bash
python visualization.py
```

Output:
```text
road_route_visualization.html
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- ORTools
- OSRM
- Folium
- SciPy
- OpenPyXL

---

# Notes

- V1 uses Haversine distance approximation.
- V2 uses actual road-network distances from OSRM.
- V2 uses fully heuristic-based optimization without ORTools.
- Public OSRM APIs may become slower for very large datasets.