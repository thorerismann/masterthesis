import os
from pathlib import Path

import geopandas as gpd
from osgeo import gdal, ogr, osr
import pandas as pd
gdal.DontUseExceptions()

class PrepGeoDataViz:
    def __init__(self, points_path = '/home/tge/masterthesis/database/sensorpoints', bounds = 1000):
        self.points = PrepGeoDataViz.load_points(points_path)
        self.bbox = self.bounding_box(bounds)
        self.raster_paths = dict(fitnahtemp='/home/tge/masterthesis/database/fitnahtemp/reprojected_temp.tif',
                                 fitnahuhispace='/home/tge/masterthesis/database/fitnahuhi/winss20n_reproj.tif',
                                 fitnahuhistreet='/home/tge/masterthesis/database/fitnahuhi/winsv20n_reproj.tif',
                                 dem='/home/tge/masterthesis/database/dem_ch/dem25.tif',
                                 landuse='/home/tge/masterthesis/database/landuse/ntzg10m_final_ug_mrandom_rev00.tif'
                                 )
    @staticmethod
    def load_points(points_path):
        points = gpd.read_file(points_path)
        names = points.Name.str.split(' ', expand=True)[1]
        names = pd.to_numeric(names, errors='coerce')
        points.Name = names
        points = points.dropna(subset='Name').sort_values(by='Name').set_index('Name', drop=True)
        return points

    def bounding_box(self, bounds):
        excluded = self.points[(self.points.index >206) & (self.points.index < 208)]
        bounds = excluded.total_bounds
        extended_bounds = [bounds[0]-1000, bounds[1]-1000, bounds[2]+1000, bounds[3] + 1000]
        gdal_bounds = [extended_bounds[0], extended_bounds[3], extended_bounds[2], extended_bounds[1]]
        return gdal_bounds

    def crop_raster(self, path, name):
        """
        Crop a raster file to the specified bounding box and save the output.

        Parameters:
        - input_raster_path: Path to the input raster file.
        - output_raster_path: Path where the cropped raster file will be saved.
        - bbox: A tuple of (minX, minY, maxX, maxY) defining the bounding box for cropping.
        """
        # Open the input raster
        src_ds = gdal.Open(path)

        translate_options = gdal.TranslateOptions(projWin=self.bbox)
        # Perform the cropping and save the result
        base = Path.cwd() / 'cutdata'
        base.mkdir(exist_ok=True)
        gdal.Translate(Path.cwd() / 'cutdata' / f'{name}.tif', src_ds, options=translate_options)
        # Clean up
        src_ds = None

    def crop_all_rasters(self):
        for name, path in self.raster_paths.items():
            self.crop_raster(path,name)

    def raster_to_contours(self, input_raster_path, output_geojson_path, contour_interval):
        # Open the source raster
        output_geojson_path = str(output_geojson_path)
        input_raster_path = str(input_raster_path)
        src_ds = gdal.Open(str(input_raster_path))
        band = src_ds.GetRasterBand(1)
        print('contour loaded')

        # Prepare the output GeoJSON
        drv = ogr.GetDriverByName('GeoJSON')
        if os.path.exists(output_geojson_path):
            drv.DeleteDataSource(output_geojson_path)
        out_ds = drv.CreateDataSource(output_geojson_path)
        print('data source created')

        # Create the spatial reference from the raster
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjection())

        # Create a layer in the GeoJSON DataSource
        out_layer = out_ds.CreateLayer(output_geojson_path, srs=srs)
        print('out layer crated')

        # Perform contour generation
        gdal.ContourGenerate(band, contour_interval, 0, [], 0, 0, out_layer, 0, 0)
        print('contour generated')

        # Cleanup
        src_ds = None
        out_ds = None

    def prepare_visualization(self, contour_interval=20):
        self.crop_all_rasters()
        self.raster_to_contours(Path.cwd() / 'cutdata' / 'dem.tif', Path.cwd() / 'cutdata' / 'dem.geojson', contour_interval)


vgd= PrepGeoDataViz()
vgd.crop_all_rasters()
vgd.prepare_visualization(contour_interval=10)

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
check_geojson('/vizdata/landuseqgis.geojson')