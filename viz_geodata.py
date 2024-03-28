import os
from pathlib import Path

import geopandas as gpd
from osgeo import gdal, ogr, osr
import pandas as pd
gdal.DontUseExceptions()

class VizGeoData:
    def __init__(self, points_path = '/home/tge/masterthesis/database/sensorpoints', bounds = 1000):
        self.points = VizGeoData.load_points(points_path)
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
        bounds = self.points.total_bounds
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

    def raster_to_geojson(self, input_raster_path, output_geojson_path):
        # Open the source raster
        src_ds = gdal.Open(str(input_raster_path))
        srcband = src_ds.GetRasterBand(1)

        # Prepare the output GeoJSON
        driver = ogr.GetDriverByName('GeoJSON')
        if os.path.exists(output_geojson_path):
            driver.DeleteDataSource(output_geojson_path)
        out_ds = driver.CreateDataSource(str(output_geojson_path))

        # Create the spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjection())

        # Create the layer
        out_layer = out_ds.CreateLayer(str(output_geojson_path), srs=srs)

        # Add an ID field
        id_field = ogr.FieldDefn("id", ogr.OFTInteger)
        out_layer.CreateField(id_field)

        # Polygonize
        gdal.Polygonize(srcband, None, out_layer, 0, [], callback=None)
        print('polygonized')

        # Cleanup
        src_ds = None
        out_ds = None

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

    def raster_to_png(self, input_raster_path, output_png_path):
        # Convert raster to PNG for direct visualization
        src_ds = gdal.Open(input_raster_path)
        gdal.Translate(output_png_path, src_ds, format='PNG')
        src_ds = None

    def prepare_visualization(self, contour_interval=20):
        self.crop_all_rasters()
        self.raster_to_geojson(Path.cwd() / 'cutdata' / 'landuse.tif', Path.cwd() / 'cutdata' / 'landuse.geojson')
        self.raster_to_contours(Path.cwd() / 'cutdata' / 'dem.tif', Path.cwd() / 'cutdata' / 'dem.geojson', contour_interval)
        # for name in ['fitnahtemp', 'fitnahuhispace', 'fitnahuhistreet']:
        #     self.raster_to_png(Path.cwd() / 'cutdata' / self.raster_paths[name], base / f'{name}.png')

    def fix_geojson(self, name, min_area=100, simplify_tolerance=0.1):
        load = Path.cwd() / 'cutdata' / name
        save = Path.cwd() / 'vizdata'
        save.mkdir(exist_ok=True)

        gdf = gpd.read_file(load)
        # Simplify geometries to smooth edges
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=simplify_tolerance, preserve_topology=True)
        gdf = gdf[gdf['geometry'].area > min_area]
        gdf.to_file(save / name)

vgd= VizGeoData()
vgd.crop_all_rasters()
vgd.prepare_visualization()
vgd.fix_geojson('landuse.geojson')
