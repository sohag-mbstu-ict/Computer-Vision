import pandas as pd
import folium
import json
import user_location
from user_data import get_location

class BangladeshMap:
    def __init__(self, district_data, geojson_path, output_file):
        """
        Initialize the BangladeshMap class.
        
        :param district_data: DataFrame containing district information with 'Name', 'lat', and 'lon' columns.
        :param geojson_path: Path to the GeoJSON file for Bangladesh boundaries.
        :param output_file: Name of the output HTML file.
        """
        self.district_data = district_data
        self.geojson_path = geojson_path
        self.output_file = output_file
        self.map = folium.Map(location=[23.6850, 90.3563], zoom_start=7, max_bounds=True)


    def add_markers(self):
        """
        Add district markers to the map as green points.
        """
        for _, row in self.district_data.iterrows():
            # Add a green circle marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=1,  # Size of the circle
                color='green',  # Border color
                fill=True,  # Fill the circle
                fill_color='green',  # Fill color
                fill_opacity=0.7  # Fill opacity
            ).add_to(self.map)

            
    def add_geojson_boundary(self):
        """
        Add the GeoJSON boundary of Bangladesh to the map.
        """
        try:
            with open(self.geojson_path, 'r') as f:
                bangladesh_geojson = json.load(f)

            folium.GeoJson(
                bangladesh_geojson,
                name="Bangladesh Boundary",
                style_function=lambda x: {
                    'fillColor': 'blue',
                    'color': 'red',
                    'weight': 0.01,
                    'fillOpacity': 0.1
                }
            ).add_to(self.map)
        except FileNotFoundError:
            print(f"Error: GeoJSON file not found at {self.geojson_path}")

    def save_map(self):
        """
        Save the map to the specified HTML file.
        """
        self.map.save(self.output_file)
        print(f"Map with boundary saved as {self.output_file}")

    def generate_map(self):
        """
        Generate the map with markers and boundary, then save it.
        """
        self.add_markers()
        self.add_geojson_boundary()
        self.save_map()


if __name__ == "__main__":
    
    district_data = get_location()
    print("-- : ",district_data)
    # Path to GeoJSON file
    geojson_path = '/home/mtl/Music/Data_Analysis/gadm41_BGD_3.json'

    # Output HTML file
    output_file = '/home/mtl/Music/Data_Analysis/output/Bangladesh_districts_with_boun.html'

    # Instantiate and generate the map
    bangladesh_map = BangladeshMap(district_data, geojson_path, output_file)
    bangladesh_map.generate_map()


