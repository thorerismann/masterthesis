from pathlib import Path
import sqlite3

import numpy as np
import rasterio
import xarray as xr
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask


class PrepareData:
    def __init__(self):
        self.meteo_path_23 = '/home/tge/masterthesis/database/meteo/biel_23.txt'
        self.meteo_path_22 = '/home/tge/masterthesis/database/meteo/biel_22.txt'
        self.biel_path_23 = '/home/tge/masterthesis/database/sensordata/biel23.nc'
        self.biel_path_22 = '/home/tge/masterthesis/database/sensordata/biel22.nc'

    def format_meteo(self, meteo_path, skiprows):
        """Format meteo data for use in the model."""
        df = pd.read_table(meteo_path, sep=r'\s+', skiprows=skiprows)
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M', errors='coerce')
        rename_cols = {
            'tre200s0': 'airtemp',
            'rre150z0': 'precipitation',
            'ure200s0': 'humidity',
            'fve010z0': 'windspeed',
            'dkl010z0': 'winddir'
        }
        df = df.rename(columns=rename_cols)
        df = df[df.stn == 'GRE']
        airtemp = df[['time', 'airtemp', 'humidity']].dropna()
        airtemp.set_index('time', inplace=True)
        return airtemp

    def combine_meteo_data(self):
        airtemp_23 = self.format_meteo(self.meteo_path_23, 1)
        airtemp_22 = self.format_meteo(self.meteo_path_22, 0)
        combined = pd.concat([airtemp_23, airtemp_22])
        print(combined)
        directory = Path('/home/tge/masterthesis/app/database')
        directory.mkdir(exist_ok=True)
        combined.reset_index().to_csv(directory / 'meteo.csv', index=False)


    def format_station_data(self):
        ds23 = xr.open_dataset(self.biel_path_23)
        temp23 = ds23.to_dataframe().reset_index()
        temp23 = temp23[['time', 'logger', 'temperature']].copy()
        temp23 = temp23.dropna()
        ds22 = xr.open_dataset(self.biel_path_22)
        temp22 = ds22.to_dataframe().reset_index()
        temp22 = temp22.rename(columns={'sensor': 'logger', 'temp': 'temperature'})
        temp22 = temp22[['time', 'logger', 'temperature']].copy()
        temp22['logger'] = pd.to_numeric(temp22['logger'], errors='coerce')
        temp22 = temp22.dropna()
        print(temp22)
        temp22['logger'] = temp22['logger'].astype(int)
        df = pd.concat([temp22, temp23])
        df = df.sort_values(['time', 'logger'])
        path = Path('/home/tge/masterthesis/app/database')
        path.mkdir(exist_ok=True)
        df.to_csv(path / 'stations.csv', index=False)

    def collect_geodata(self):
        pass



dataprep = PrepareData()
dataprep.combine_meteo_data()
stationdata = dataprep.format_station_data()
print('meteo and sensor data prepped')

class GeoDataCollector:
    def __init__(self):

        self.raster_paths = dict(fitnahtemp='/home/tge/masterthesis/database/fitnahtemp/reprojected_temp.tif',
                                 fitnahuhispace='/home/tge/masterthesis/database/fitnahuhi/winss20n_reproj.tif',
                                 fitnahuhistreet='/home/tge/masterthesis/database/fitnahuhi/winsv20n_reproj.tif',
                                 dem='/home/tge/masterthesis/database/dem_ch/dem25.tif',
                                 landuse='/home/tge/masterthesis/database/landuse/ntzg10m_final_ug_mrandom_rev00.tif'
                                 )
        self.shape_paths = dict(wind='/home/tge/masterthesis/database/Strömung')
        self.point_path = '/home/tge/masterthesis/app/database/sensorpoints'
        self.save_path = '/home/tge/masterthesis/app/database'
        self.points = GeoDataCollector.load_points(self.point_path)
        self.buffers = [5, 10, 20, 50, 75, 100, 150, 200, 250, 300, 500, 750, 1000]


    @staticmethod
    def load_points(points_path):
        points = gpd.read_file(points_path)
        names = points.Name.str.split(' ', expand=True)[1]
        names = pd.to_numeric(names, errors='coerce')
        points.Name = names
        points = points.dropna(subset='Name').sort_values(by='Name').set_index('Name', drop=True)
        return points

    @property
    def create_buffers(self):
        # do something to points
        buffer_frame = gpd.GeoDataFrame(index=self.points.index, columns=self.buffers)
        for buffer in self.buffers:
            buffer_frame[buffer] = self.points.buffer(buffer)
        return buffer_frame

    def get_raster_image(self, src, geometry):
        out_image, out_transform = mask(src, [geometry], crop=True, all_touched=True)
        return out_image[0]

    def calculate_statistics(self, data):
        """Calculate statistical metrics from the raster data.
        """
        stats = {
            'mean': np.nan,
            'max': np.nan,
            'min': np.nan,
            'median': np.nan,
            'count': 0,  # Count of non-NaN grid squares
        }

        if data.size > 0:
            stats.update({
                'mean': np.nanmean(data),
                'max': np.nanmax(data),
                'min': np.nanmin(data),
                'median': np.nanmedian(data),
                'count': data.size,
            })

        return stats

    def get_raster_stats(self, path):
        buffer_data = self.create_buffers
        with rasterio.open(path) as src:
            buffered_data = []
            for buffer in buffer_data.columns:  # Iterate over each buffer size
                for idx, row in buffer_data.iterrows():
                    geometry = row[buffer]  # Access the geometry for the current buffer size
                    image = self.get_raster_image(src, geometry)
                    data_clean = image[image != src.nodata]  # Clean the data to ignore nodata
                    stationstats = self.calculate_statistics(data_clean)
                    stationstats['buffer'] = int(buffer)
                    stationstats['logger'] = int(idx)
                    buffered_data.append(stationstats)  # Organize results by buffer size

        df = pd.DataFrame(buffered_data)
        return df

    def calculate_landuse_counts(self, data, unique_categories):
        """Calculate counts for each land use category within the data."""
        counts = {int(category): np.sum(data == category) for category in unique_categories}
        return counts

    def get_landuse_stats(self):
        buffer_geometries = self.create_buffers  # Assuming this returns a GeoDataFrame with geometries as columns
        with rasterio.open(self.raster_paths['landuse']) as src:
            # If unique categories are not predefined, you could determine them dynamically:
            entire_image = src.read(1)
            unique_categories = np.unique(entire_image)
            unique_categories = unique_categories[unique_categories != src.nodata]  # Exclude nodata

            buffered_data = []
            for buffer in buffer_geometries.columns[:-1]:  # Exclude the logger column
                for idx, row in buffer_geometries.iterrows():
                    geometry = row[buffer]
                    image = self.get_raster_image(src, geometry)
                    land_use_counts = self.calculate_landuse_counts(image, unique_categories)
                    land_use_counts['buffer'] = int(buffer)
                    land_use_counts['logger'] = int(idx)
                    buffered_data.append(land_use_counts)

        return pd.DataFrame(buffered_data)

    def save_buffered_data(self):
        data = self.calculate_rasters()
        Path(self.save_path).mkdir(exist_ok=True)
        path = Path(self.save_path) / 'buffered_data.csv'
        data = data[data['dtype'] != 'landuse']
        data.to_csv(path, index=False)
        data_lu = self.get_landuse_stats()
        path_lu = Path(self.save_path) / 'buffered_landuse.csv'
        data_lu.to_csv(path_lu, index=False)

    def calculate_rasters(self):
        buffered_data = []
        for name, path in self.raster_paths.items():
            data = self.get_raster_stats(path)
            data['dtype'] = name
            buffered_data.append(data)
        data = pd.concat(buffered_data)
        return data

gdc = GeoDataCollector()
gdc.save_buffered_data()
print('geodata saved')