import copy

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium, folium_static
import geopandas as gpd
from folium import folium, GeoJson, GeoJsonTooltip, LayerControl
from folium.plugins import FastMarkerCluster

st.header('Summer Temperatures in Biel / Bienne')

class MapDisplay:

    def __init__(self, zoom_start = 10, landuse_path = '/home/tge/masterthesis/vizdata/landuseqgis.geojson', elevation_path = '/home/tge/masterthesis/vizdata/dem.geojson',location = [47.1403889, 7.25302778], points_path = '/home/tge/masterthesis/database/sensorpoints'):
        self.points = self.load_points(points_path)
        self.base_map = self.load_base_map(location, zoom_start)
        self.elevation_contour = self.load_elevation_contour(elevation_path)
        self.landuse = self.load_landuse(landuse_path)

    def load_points(self, points_path):
        points = gpd.read_file(points_path)
        names = points.Name.str.split(' ', expand=True)[1]
        names = pd.to_numeric(names, errors='coerce')
        points.Name = names
        points = points.dropna(subset='Name').sort_values(by='Name')
        points = points.to_crs(epsg=4326)
        return points

    def load_base_map(self, location, zoom_start):
        m = folium.Map(location = location, zoom_start = zoom_start)
        return m

    def add_points(self):
        newmap = copy.copy(self.base_map)
        GeoJson(self.points, tooltip=GeoJsonTooltip(fields=['Name'])).add_to(newmap)
        return newmap

    def load_elevation_contour(self, elevation_path):
        return gpd.read_file(elevation_path)
    def add_elevation_contour(self):
        newmap = copy.copy(self.base_map)
        newmap = self.add_points()
        GeoJson(self.elevation_contour).add_to(newmap)
        return newmap

    def load_landuse(self, landuse_path):
        return gpd.read_file(landuse_path)

    def add_landuse(self):
        newmap = self.add_points()
        # Add GeoJSON with style function and tooltip
        GeoJson(self.landuse).add_to(newmap)
        LayerControl().add_to(newmap)
        return newmap


import geopandas as gpd

def check_geojson(landuse_path):
    # Load the GeoDataFrame from the GeoJSON file
    gdf = gpd.read_file(landuse_path)

    # Print the first few rows of the GeoDataFrame
    print(gdf.head())
    print(gdf.crs)

    # Print the total number of rows in the GeoDataFrame
    print(f"Total rows: {len(gdf)}")

    # Check if there are any missing values in the GeoDataFrame
    print("Missing values:")
    print(gdf.isnull().sum())

    # Check the coordinate reference system of the GeoDataFrame
    print(f"Coordinate reference system: {gdf.crs}")

# Call the function with the path to your GeoJSON file
check_geojson('/home/tge/masterthesis/vizdata/landuseqgis.geojson')

def main():
    map_display = MapDisplay()
    # elevationmap = map_display.add_elevation_contour()
    # st_folium(elevationmap)
    landusemap = map_display.add_landuse()
    st_folium(landusemap)
main()
