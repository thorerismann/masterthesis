import geopandas as gpd
import pandas as pd
from folium import folium, GeoJson, GeoJsonTooltip


class MapDisplay:

    def __init__(self, zoom_start = 12, elevation_path = '/home/tge/masterthesis/vizdata/dem.geojson',location = [47.1403889, 7.25302778], points_path = '/home/tge/masterthesis/database/sensorpoints'):
        self.points = self.load_points(points_path)
        self.base_map = self.load_base_map(location, zoom_start)
        self.elevation_contour = self.load_elevation_contour(elevation_path)

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
        GeoJson(self.points, tooltip=GeoJsonTooltip(fields=['Name'])).add_to(m)
        return m

    def load_elevation_contour(self, elevation_path):
        return gpd.read_file(elevation_path)
    def plot_elevation_contour(self):
        GeoJson(self.elevation_contour).add_to(self.base_map)
        return self.base_map
