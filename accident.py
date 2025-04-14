!pip install googlemaps
!pip install geopy
!pip install pandas
!pip install requests
!pip install osmnx
!pip install networkx
!pip install typing
!pip install collections
import pandas as pd
import googlemaps
from googlemaps.convert import decode_polyline
from geopy.distance import geodesic
import numpy as np
import requests
import osmnx as ox
import networkx as nx
from typing import Tuple, List
from collections import defaultdict
from datetime import datetime
import json


API_KEY = "AIzaSyC5SPTutMRcaSxMvHExAb4e-eOLL4hhO8o"
gmaps = googlemaps.Client(key=API_KEY)


PREDICTION_URL = "https://model-6xuc.onrender.com/predict"


FEATURES = ['Road_Type_Number', 'Lanes', 'Oneway_Status', '+', 'T', 'complex', 'Curve', 'Speed_Limit', 'Traffic']


def get_coordinates():
    # Start an infinite loop to keep asking the user for input until valid coordinates are provided
    while True:
        try:
            # Ask the user to enter the starting coordinates in the format "latitude,longitude"
            start_input = input("Enter start coordinates (latitude, longitude) separated by comma: ")

            # Split the input string by comma and convert the two parts to float values
            start_lat, start_lon = map(float, start_input.split(','))

            # Ask the user to enter the ending coordinates in the format "latitude,longitude"
            end_input = input("Enter end coordinates (latitude, longitude) separated by comma: ")

            # Split the input string by comma and convert the two parts to float values
            end_lat, end_lon = map(float, end_input.split(','))

            # If no error occurs during the conversion, return both start and end coordinate pairs as tuples
            return (start_lat, start_lon), (end_lat, end_lon)

        # If the input format is incorrect or cannot be converted to float, catch the error
        except ValueError:
            # Display an error message to the user and prompt them to enter again
            print("Invalid input! Please enter coordinates in format: latitude,longitude (e.g., 10.4446,76.2591)")


# Function to get prediction from the deployed ML model API
def get_prediction(features_list):
    try:
        # Prepare the input data in JSON format (dictionary with "features" key)
        input_data = {"features": features_list}

        # Send a POST request to the prediction API with the input data as JSON
        response = requests.post(PREDICTION_URL, json=input_data)

        # Check if the request was successful (status code 200 means OK)
        if response.status_code == 200:
            # Parse the JSON response and extract the 'prediction' value
            return response.json().get('prediction', 'Unknown')
        else:
            # If response code is not 200, print the error message with status code
            print(f"Prediction API Error: {response.status_code} - {response.text}")
            return "Error"
    except Exception as e:
        # Catch any unexpected errors (e.g., network error) and print it
        print(f"Error calling prediction API: {e}")
        return "Error"


# Function to get detailed route coordinates between two points using Google Maps Directions API
def get_route_points(start, end):
    try:
        # Request driving directions between the start and end coordinates
        directions = gmaps.directions(start, end, mode="driving")

        # If no directions are returned, notify the user and return an empty list
        if not directions:
            print("🚨 No route found!")
            return []

        # Extract the encoded polyline from the first route returned
        polyline = directions[0]['overview_polyline']['points']

        # Decode the polyline into a list of points (latitude and longitude)
        route = decode_polyline(polyline)

        # Return the coordinates as a list of (lat, lon) tuples
        return [(point['lat'], point['lng']) for point in route]

    except Exception as e:
        # Handle any errors (e.g., network or API error)
        print(f"Error fetching route: {e}")
        return []


# Function to interpolate additional points along a route
# Adds points approximately every `distance` meters (default 50m)
def interpolate_points(route, distance=50):
    # Start with the first point of the route
    new_points = [route[0]]

    # Loop through each pair of consecutive points in the route
    for i in range(len(route) - 1):
        start = route[i]         # Current point
        end = route[i + 1]       # Next point

        # Calculate the distance between the two points in meters
        dist = geodesic(start, end).meters

        # If the distance is greater than the threshold, interpolate points
        if dist > distance:
            # Calculate how many points are needed to cover this distance
            num_points = int(dist // distance)

            # Calculate how much latitude and longitude to move per step
            lat_step = (end[0] - start[0]) / (num_points + 1)
            lng_step = (end[1] - start[1]) / (num_points + 1)

            # Add interpolated points between start and end
            for j in range(num_points):
                new_lat = start[0] + (lat_step * (j + 1))
                new_lng = start[1] + (lng_step * (j + 1))
                new_points.append((new_lat, new_lng))

        # Always add the actual endpoint of the current segment
        new_points.append(end)

    # Return the full list of points, including interpolated ones
    return new_points


# Function to calculate the angle (in degrees) formed at point p2 using points p1, p2, and p3
def calculate_angle(p1, p2, p3):
    # Create vector from p2 to p1
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    # Create vector from p2 to p3
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Calculate the length (magnitude) of each vector
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    # If either vector is zero-length, angle can't be calculated
    if mag_v1 == 0 or mag_v2 == 0:
        return np.nan

    # Calculate cosine of angle between vectors using dot product formula
    cos_theta = np.dot(v1, v2) / (mag_v1 * mag_v2)

    # Clip value between -1 and 1 to avoid math errors due to floating point precision
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in degrees
    angle = np.degrees(np.arccos(cos_theta))
    return angle

# Function to get road type, number of lanes, and one-way status from Overpass API using a lat/lon location
def get_road_details(latitude, longitude):
    buffer = 0.0001  # Small margin to create a bounding box around the location

    # Build the bounding box for searching roads near the point
    bbox = f"{latitude - buffer},{longitude - buffer},{latitude + buffer},{longitude + buffer}"

    # Overpass QL query to get all 'highway' type ways within the bounding box
    query = f"""
    [out:json];
    way({bbox})["highway"];
    out body;
    >;
    out skel qt;
    """

    # Send query to Overpass API
    response = requests.post("https://overpass-api.de/api/interpreter", data={'data': query})

    # Map tags to a custom number representing road type
    road_type_mapping = {
        'NH': 1, 'SH': 2, 'primary': 3, 'secondary': 3, 'tertiary': 4,
        'residential': 4, 'unclassified': 4, 'Not Found': 5, 'Error': 5
    }

    # If API call successful
    if response.status_code == 200:
        data = response.json()
        closest_road = None
        min_distance = float('inf')

        # Go through all the road 'ways' returned
        for element in data['elements']:
            if element['type'] == 'way' and 'tags' in element:
                tags = element['tags']
                road_nodes = element.get('nodes', [])

                # Find the closest node of this road to the input point
                for node_id in road_nodes:
                    node = next((n for n in data['elements'] if n['type'] == 'node' and n['id'] == node_id), None)
                    if node:
                        node_coords = (node['lat'], node['lon'])
                        distance = geodesic((latitude, longitude), node_coords).meters
                        if distance < min_distance:
                            min_distance = distance
                            closest_road = tags  # Save road info

        if closest_road:
            # Get road type, reference number, one-way status, and lanes
            road_type = closest_road.get('highway', 'Unknown')
            ref = closest_road.get('ref', '')
            oneway = closest_road.get('oneway', 'Unknown')
            lanes = closest_road.get('lanes', 'Unknown')

            # Map to custom road type number
            if 'NH' in ref:
                road_type_number = road_type_mapping['NH']
            elif 'SH' in ref:
                road_type_number = road_type_mapping['SH']
            else:
                road_type_number = road_type_mapping.get(road_type, 4)

            # Convert one-way to 1 or 0
            oneway_status = 1 if oneway in ['yes', '-1'] else 0

            # Estimate number of lanes if unknown
            if oneway_status == 0:
                lanes = '1' if lanes in ['2', 'Unknown'] else lanes
            elif oneway_status == 1:
                lanes = '1' if lanes == 'Unknown' else lanes

            return {
                'road_type_number': road_type_number,
                'lanes': lanes,
                'oneway_status': oneway_status
            }

        # If no matching road found
        return {'road_type_number': 5, 'lanes': 'Unknown', 'oneway_status': 0}
    else:
        print(f"Failed to fetch road data: {response.status_code}")
        return {'road_type_number': 5, 'lanes': 'Unknown', 'oneway_status': 0}


# Junction-related functions
def get_junction_and_road_types(lat: float, lon: float) -> dict:
    # Overpass API endpoint for querying OpenStreetMap data
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Overpass QL query to get all highway ways around a 20-meter radius of the point
    query = f"""
    [out:json];
    way(around:20, {lat}, {lon})["highway"];
    (._;>;);
    out;
    """

    try:
        # Send the query to Overpass API
        response = requests.get(overpass_url, params={'data': query})

        # Check if the API response is successful
        if response.status_code != 200:
            return {"junction_type": "Error", "connecting_road_types": []}

        # Parse the JSON response
        data = response.json()

        # If no road elements are returned
        if not data['elements']:
            return {"junction_type": "No roads found", "connecting_road_types": []}

        # Dictionaries to track how many times each node is used and what road types it connects
        node_count = defaultdict(int)         # node_id -> count of connections
        road_types = defaultdict(set)         # node_id -> set of road types (e.g., residential, primary)

        # Loop through all elements to process the road types and their nodes
        for element in data['elements']:
            if element['type'] == 'way' and 'tags' in element:
                road_type = element['tags'].get('highway', 'Unknown')  # Get the type of highway
                for node in element['nodes']:
                    node_count[node] += 1                # Count how many times this node appears
                    road_types[node].add(road_type)      # Record what types of roads meet at this node

        # Find the maximum number of roads that meet at any node
        max_connections = max(node_count.values(), default=1)

        # Determine the junction type based on the number of connections
        if max_connections == 1:
            junction_type = "No junction (Single road)"
        elif max_connections == 2:
            junction_type = "T-junction or Y-junction"
        elif max_connections in [3, 4]:
            junction_type = "Crossroad (+ type)"
        elif max_connections > 4:
            junction_type = "Complex junction"
        else:
            junction_type = "Unknown"

        # Get all unique road types involved in the junction
        connecting_road_types = set()
        for node, count in node_count.items():
            if count > 1 and node in road_types:
                connecting_road_types.update(road_types[node])

        # Return the junction type and all connecting road types
        return {
            "junction_type": junction_type,
            "connecting_road_types": list(connecting_road_types)
        }

    except Exception as e:
        # Return error details if something went wrong (e.g., network error)
        return {"junction_type": "Error", "connecting_road_types": [], "error": str(e)}


def is_on_path(start_coords: Tuple[float, float], end_coords: Tuple[float, float], point: Tuple[float, float], buffer_meters: float = 10) -> bool:
    start_to_point = geodesic(start_coords, point).meters
    point_to_end = geodesic(point, end_coords).meters
    start_to_end = geodesic(start_coords, end_coords).meters
    return abs(start_to_point + point_to_end - start_to_end) < buffer_meters

def get_traffic_status(lat, lon, offset=0.001):
    origin = f"{lat},{lon}"
    destination = f"{lat},{lon + offset}"
    now = datetime.now()
    try:
        directions_result = gmaps.directions(
            origin, destination, mode="driving", departure_time=now, traffic_model="best_guess"
        )
        if not directions_result or 'legs' not in directions_result[0]:
            return "No traffic data"
        leg = directions_result[0]['legs'][0]
        normal_duration = leg['duration']['value']
        if 'duration_in_traffic' not in leg:
            return "No traffic data"
        traffic_duration = leg['duration_in_traffic']['value']
        traffic_ratio = traffic_duration / normal_duration if normal_duration > 0 else 1
        if traffic_ratio <= 0.9:
            return 0  # No traffic
        elif traffic_ratio <= 1:
            return 1  # Mild traffic
        else:
            return 2  # Severe traffic
    except Exception as e:
        print(f"Traffic error: {e}")
        return "No traffic data"

def get_speed_limit(road_type_number):
    speed_limits = {1: 80, 2: 70, 3: 60, 4: 40, 5: 30}
    return speed_limits.get(road_type_number, 30)

def get_junctions(start_coords: Tuple[float, float], end_coords: Tuple[float, float], ref_coords: Tuple[float, float], dist: float = 3000) -> List[dict]:
    approx_dist = geodesic(start_coords, end_coords).meters
    network_dist = max(dist, approx_dist * 1.5)
    center_point = ((start_coords[0] + end_coords[0]) / 2, (start_coords[1] + end_coords[1]) / 2)
    G = ox.graph_from_point(center_point, dist=network_dist, network_type='drive')

    start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
    end_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])

    if start_node == end_node:
        return []

    try:
        route = nx.shortest_path(G, start_node, end_node, weight='length')
    except nx.NetworkXNoPath:
        return []

    junctions = []
    for node in route[1:-1]:
        degree = G.degree[node]
        current_coords = (G.nodes[node]['y'], G.nodes[node]['x'])
        if degree > 2 and is_on_path(start_coords, end_coords, current_coords):
            junction_info = get_junction_and_road_types(current_coords[0], current_coords[1])
            distance_from_ref = geodesic(ref_coords, current_coords).meters
            junctions.append({
                "latitude": current_coords[0],
                "longitude": current_coords[1],
                "junction_type": junction_info["junction_type"],
                "connecting_road_types": junction_info["connecting_road_types"],
                "distance_from_ref_meters": round(distance_from_ref, 2)
            })
    return junctions

def process_row(row):
    start_coords = (row['Start_Latitude'], row['Start_Longitude'])
    end_coords = (row['End_Latitude'], row['End_Longitude'])
    ref_coords = (row['Latitude'], row['Longitude'])

    junctions = get_junctions(start_coords, end_coords, ref_coords)

    plus_dist = -1
    t_dist = -1
    complex_dist = -1

    for junction in junctions:
        if junction["junction_type"] == "Crossroad (+ type)":
            plus_dist = junction["distance_from_ref_meters"]
        elif junction["junction_type"] == "T-junction or Y-junction":
            t_dist = junction["distance_from_ref_meters"]
        elif junction["junction_type"] == "Complex junction":
            complex_dist = junction["distance_from_ref_meters"]

    return plus_dist, t_dist, complex_dist

# Function to generate HTML file with Google Maps JavaScript API
def generate_interactive_map_from_excel(excel_file, api_key, output_file="route_map.html"):
    # Read the Excel file
    df = pd.read_excel(excel_file)

    # Ensure necessary columns are present
    required_columns = ['Latitude', 'Longitude', 'Prediction']
    if not all(col in df.columns for col in required_columns):
        print("Error: Excel file must contain Latitude, Longitude, and Prediction columns.")
        return

    # Convert Prediction to numeric, handling any non-numeric values
    df['Prediction'] = pd.to_numeric(df['Prediction'], errors='coerce').fillna(0).astype(int)

    # Get start and end points for the map bounds
    start_location = (df['Latitude'].iloc[0], df['Longitude'].iloc[0])
    end_location = (df['Latitude'].iloc[-1], df['Longitude'].iloc[-1])

    # Prepare points with their predictions
    points = [{'lat': row['Latitude'], 'lng': row['Longitude'], 'is_prone': row['Prediction'] == 1}
              for _, row in df.iterrows()]

    # Generate segments from consecutive points
    segments = []
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        # Segment is red if the start point is prone (Prediction == 1), giving red priority
        is_prone = start_point['is_prone']
        segments.append({
            'start': {'lat': start_point['lat'], 'lng': start_point['lng']},
            'end': {'lat': end_point['lat'], 'lng': end_point['lng']},
            'is_prone': int(is_prone)  # Convert to int for JavaScript compatibility
        })

    # HTML and JavaScript content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Route Map with Accident-Prone Areas</title>
        <style>
            #map {{
                height: 100%;
            }}
            html, body {{
                height: 100%;
                margin: 0;
                padding: 0;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            function initMap() {{
                const map = new google.maps.Map(document.getElementById("map"), {{
                    zoom: 13,
                    center: {{ lat: {start_location[0]}, lng: {start_location[1]} }},
                }});

                // Define segments
                const segments = {json.dumps(segments)};

                // Draw each segment with appropriate color
                segments.forEach(segment => {{
                    const path = [segment.start, segment.end];
                    const polyline = new google.maps.Polyline({{
                        path: path,
                        geodesic: true,
                        strokeColor: segment.is_prone ? '#FF0000' : '#0000FF', // Red if prone, Blue if not
                        strokeOpacity: 1.0,
                        strokeWeight: 5,
                    }});
                    polyline.setMap(map);
                }});

                // Add markers for start and end
                new google.maps.Marker({{
                    position: {{ lat: {start_location[0]}, lng: {start_location[1]} }},
                    map: map,
                    title: "Start",
                }});
                new google.maps.Marker({{
                    position: {{ lat: {end_location[0]}, lng: {end_location[1]} }},
                    map: map,
                    title: "End",
                }});
            }}
        </script>
        <script async defer src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap"></script>
    </body>
    </html>
    """

    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"🌍 Interactive map saved as {output_file}")
    print(f"Open {output_file} in a browser to view the route with accident-prone areas (Prediction == 1) in red and safe areas in blue!")


# 🔥 Main execution
print("Please provide the start and end coordinates for your route.")
start_location, end_location = get_coordinates()
print(f"Starting route calculation from {start_location} to {end_location}")

route_points = get_route_points(start_location, end_location)

if not route_points:
    print("No route points found. Exiting...")
else:
    # 🔄 Interpolate every 50 meters
    interpolated_points = interpolate_points(route_points, distance=50)
    print(f"Number of interpolated points: {len(interpolated_points)}")

    # 📝 Convert to DataFrame
    df = pd.DataFrame(interpolated_points, columns=["Latitude", "Longitude"])

    # 🚦 Create Start & End Columns
    df["Start_Latitude"] = df["Latitude"].shift(1)
    df["Start_Longitude"] = df["Longitude"].shift(1)
    df["End_Latitude"] = df["Latitude"].shift(-1)
    df["End_Longitude"] = df["Longitude"].shift(-1)

    # Clean the dataset
    df_cleaned = df.dropna().reset_index(drop=True)

    # Convert to numeric
    for col in ["Latitude", "Longitude", "Start_Latitude", "Start_Longitude", "End_Latitude", "End_Longitude"]:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # Compute angles
    df_cleaned["Curve"] = df_cleaned.apply(
        lambda row: calculate_angle(
            (row["Start_Latitude"], row["Start_Longitude"]),
            (row["Latitude"], row["Longitude"]),
            (row["End_Latitude"], row["End_Longitude"])
        ), axis=1
    )

    # Add columns
    df_cleaned['Road_Type_Number'] = ''
    df_cleaned['Lanes'] = ''
    df_cleaned['Oneway_Status'] = ''
    df_cleaned['+'] = -1
    df_cleaned['T'] = -1
    df_cleaned['complex'] = -1
    df_cleaned['Traffic'] = ''
    df_cleaned['Speed_Limit'] = ''
    df_cleaned['Prediction'] = ''  # New column for predictions

    # Process each row
    for index, row in df_cleaned.iterrows():
        latitude = row['Latitude']
        longitude = row['Longitude']

        # Get road details
        road_info = get_road_details(latitude, longitude)
        df_cleaned.at[index, 'Road_Type_Number'] = road_info['road_type_number']
        df_cleaned.at[index, 'Lanes'] = road_info['lanes']
        df_cleaned.at[index, 'Oneway_Status'] = road_info['oneway_status']

        # Get junction distances
        plus_dist, t_dist, complex_dist = process_row(row)
        df_cleaned.at[index, '+'] = plus_dist
        df_cleaned.at[index, 'T'] = t_dist
        df_cleaned.at[index, 'complex'] = complex_dist

        # Get traffic status
        traffic_status = get_traffic_status(latitude, longitude)
        df_cleaned.at[index, 'Traffic'] = traffic_status

        # Get speed limit
        speed_limit = get_speed_limit(road_info['road_type_number'])
        df_cleaned.at[index, 'Speed_Limit'] = speed_limit

        # Prepare features for prediction
        features_list = [
            road_info['road_type_number'],
            int(road_info['lanes']) if road_info['lanes'].isdigit() else 1,  # Convert lanes to int, default to 1 if unknown
            road_info['oneway_status'],
            plus_dist,
            t_dist,
            complex_dist,
            row['Curve'] if not pd.isna(row['Curve']) else 0,  # Handle NaN in Curve
            speed_limit,
            traffic_status if isinstance(traffic_status, int) else 0  # Default to 0 if no traffic data
        ]

        # Get prediction from API
        prediction = get_prediction(features_list)
        df_cleaned.at[index, 'Prediction'] = prediction

        print(f"Processed row {index + 1}/{len(df_cleaned)}: Lat {latitude}, Lon {longitude}, "
              f"+, T, complex = {plus_dist}, {t_dist}, {complex_dist}, Traffic = {traffic_status}, "
              f"Prediction = {prediction}")

    # Debug: Print statistics and sample data
    print("\nCurve angle statistics:")
    print(df_cleaned["Curve"].describe())
    print("\nTraffic statistics:")
    print(df_cleaned["Traffic"].value_counts())
    print("\nPrediction statistics:")
    print(df_cleaned["Prediction"].value_counts())
    print("\nSample data (first 10 rows):")
    print(df_cleaned.head(10))

    # 📤 Save as Excel
    output_file = "route_with_all_details_and_predictions.xlsx"
    df_cleaned.to_excel(output_file, index=False)

    print(f"✅ Data with all details and predictions saved as {output_file}")
    print("Please check your local directory for the output file!")

    # Generate interactive map
    generate_interactive_map_from_excel(output_file, API_KEY)
