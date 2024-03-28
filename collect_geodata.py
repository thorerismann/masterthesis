import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


# noinspection PyTypeChecker,PyMethodMayBeStatic
class DataCollector:
    def __init__(self, buffer_list):

        self.raster_paths = dict(fitnahtemp='/home/tge/masterthesis/database/fitnahtemp/reprojected_temp.tif',
                                 fitnahuhispace='/home/tge/masterthesis/database/fitnahuhi/winss20n_reproj.tif',
                                 fitnahuhistreet='/home/tge/masterthesis/database/fitnahuhi/winsv20n_reproj.tif',
                                 dem='/home/tge/masterthesis/database/dem_ch/dem25.tif',
                                 landuse='/home/tge/masterthesis/database/landuse/ntzg10m_final_ug_mrandom_rev00.tif'
                                 )

        self.shape_paths = dict(wind='/home/tge/masterthesis/database/Strömung')
        self.buffers = buffer_list
        self.points = DataCollector.load_points()

    @staticmethod
    def load_points():
        points = gpd.read_file('/home/tge/masterthesis/database/sensorpoints')
        names = points.Name.str.split(' ', expand=True)[1]
        names = pd.to_numeric(names, errors='coerce')
        points.Name = names
        points = points.dropna(subset='Name').sort_values(by='Name').set_index('Name', drop=True)
        return points

    def load_shape(self):
        pass

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

    def calculate_rasters(self):
        buffered_data = []
        for name, path in self.raster_paths.items():
            if name == 'landuse':
                continue
            else:
                data = self.get_raster_stats(path)
                data['dtype'] = name
                buffered_data.append(data)
        data = pd.concat(buffered_data)
        return data


buffers: list[int] = [5, 10, 20, 50, 75, 100, 150, 200, 300, 500]
dc = DataCollector(buffers)
all_data = dc.calculate_rasters()
lu_data = dc.get_landuse_stats()
