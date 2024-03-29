import sqlite3
from pathlib import Path
import streamlit as st
import xarray as xr
import pandas as pd


class StationData:
    def __init__(self, meta):
        self.meta = meta
        self.city = 'biel'
        self.station_data = None
        self.rural_data = None

    def get_biel_stations(self):
        ds = xr.open_dataset('/home/tge/masterthesis/database/sensordata/biel23.nc')
        df = ds.to_dataframe().reset_index()
        df.set_index('time', inplace=True, drop=True)
        rural = df[(df.logger == 206) | (df.logger == 207)]
        if self.meta['stations'] == 'all stations':
            stations = list(range(201, 240))
        if self.meta['stations'] == 'core stations':
            stations = [201, 202, 203, 204, 205, 207, 208, 209, 210, 228, 218, 217, 220, 222, 221, 225, 232, 231, 238, 235, 207]
        if self.meta['stations'] == 'pilot stations':
            stations = [201, 202, 203, 204, 205]
        maindf = df[df['logger'].isin(stations)]

        if self.meta['stations'] == 'pilot stations':
            ds_pilot = xr.open_dataset(Path.cwd() / 'database_biel' / 'biel22.nc')
            df_pilot = ds_pilot.to_dataframe.reset_index()
            maindf = pd.concat([maindf, df_pilot])
        return maindf, rural

    def prepare_dependent_data(self):
        maindf, rural = self.get_biel_stations()
        airtemp = maindf.reset_index().pivot(index='time', columns='logger', values='temperature')
        with sqlite3.connect(Path.cwd() / st.session_state.foldername / 'biel.db') as conn:
            if 'city_index' in self.meta['dependent']:
                airtemp_mean = airtemp.mean(axis=1)
                city_diffs = airtemp.sub(airtemp_mean, axis=0)
                st.write(city_diffs)
                city_diffs.to_sql(name='cityindex', con=conn, if_exists='replace',index=True)
                airtemp_mean.to_sql(name='citymean', con=conn, if_exists='replace', index=True)
            if 'uhi' in self.meta['dependent']:
                airtemp_rural = rural.reset_index().pivot(index='time', columns='logger', values='temperature')
                airtemp_rural_mean = airtemp_rural.mean(axis=1)
                uhi = airtemp.sub(airtemp_rural_mean,axis=0)
                st.write('uhi')
                uhi.to_sql(name='uhi',con=conn, if_exists='replace', index=True)
                rural.to_sql(name='ruralmean', con=conn, if_exists='replace', index=True)
            if 'airtemp' in self.meta['dependent']:
                st.write('airtemp')
                airtemp.to_sql(name='airtemp', con=sqlite3.connect('biel.db'), if_exists='replace')
                rural.to_sql(name='rural', con=sqlite3.connect('biel.db'), if_exists='replace')

