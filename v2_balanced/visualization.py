import pandas as pd
import requests
import folium
import random
import polyline

from tqdm.auto import tqdm

ROUTES_FILE="full_agent_routes.xlsx"
OSRM_URL="http://router.project-osrm.org"

print("\n"+"="*60)
print("LOADING ROUTES")
print("="*60)

xls=pd.ExcelFile(ROUTES_FILE)
sheet_names=xls.sheet_names

all_points=[]

for sheet in sheet_names:
    df=pd.read_excel(ROUTES_FILE,sheet_name=sheet,skiprows=2)
    all_points.append(df[["Latitude","Longitude"]])

combined=pd.concat(all_points)

center_lat=combined["Latitude"].mean()
center_lon=combined["Longitude"].mean()

m=folium.Map(location=[center_lat,center_lon],zoom_start=10)

def random_color():
    return "#%06x"%random.randint(0,0xFFFFFF)

def get_osrm_route(coords):

    coords_str=";".join([f"{lon},{lat}" for lat,lon in coords])

    url=f"{OSRM_URL}/route/v1/driving/{coords_str}?overview=full&geometries=polyline"

    response=requests.get(url,timeout=60)

    if response.status_code!=200:
        raise Exception(f"OSRM Route Error: {response.status_code}")

    data=response.json()

    geometry=data["routes"][0]["geometry"]

    return polyline.decode(geometry)

print("\n"+"="*60)
print("GENERATING ROAD VISUALIZATION")
print("="*60)

for sheet in tqdm(sheet_names,desc="Agents",colour="magenta"):

    df=pd.read_excel(ROUTES_FILE,sheet_name=sheet,skiprows=2)

    if len(df)==0:
        continue

    coords=list(zip(df["Latitude"],df["Longitude"]))

    coords.append(coords[0])

    road_path=get_osrm_route(coords)

    color=random_color()

    folium.PolyLine(road_path,color=color,weight=5,opacity=0.85,tooltip=sheet).add_to(m)

    for i,(lat,lon) in enumerate(coords[:-1],start=1):

        folium.CircleMarker(location=[lat,lon],radius=4,color=color,fill=True,fill_opacity=1,popup=f"{sheet}<br>Stop: {i}").add_to(m)

OUTPUT_FILE="road_route_visualization.html"

m.save(OUTPUT_FILE)

print("\n"+"="*60)
print("VISUALIZATION COMPLETE")
print("="*60)

print(f"\nGenerated File:")
print(f"• {OUTPUT_FILE}")