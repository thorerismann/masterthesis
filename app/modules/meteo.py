import sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd
path = '/home/tge/masterthesis/database/meteo/biel_23.txt'
df = pd.read_table(path, sep=r'\s+', skiprows=1)
print(df.columns)
print(len(df))

class MeteoData:
    """
    Class to handle meteo data
    """
    def __init__(self, meteometa, datameta):
        self.meteometa = meteometa
        self.datameta = datameta
        self.rename_cols ={'tre200s0': 'temperature', 'rre150z0': 'precipitation',
                       'ure200s0': 'humidity', 'fve010z0': 'windspeed',
                       'dkl010z0': 'winddir'}
        self.path = '/home/tge/masterthesis/database/meteo/biel_23.txt'
        self.rename_cols =rename_cols = {'tre200s0': 'airt', 'rre150z0': 'precip', 'ure200s0': 'relhum', 'fve010z0': 'wspd', 'dkl010z0': 'wdir'}

    def get_meteo_data(self):
        """Load the meteo data from the file"""
        df = pd.read_table(self.path, sep=r'\s+', skiprows=1)
        return df
    def prepare_meteo_data(self):
        """
        Prepare the meteo data for further processing
        :return: DataFrame
        """
        df = self.get_meteo_data()
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M', errors='coerce')
        df = df.dropna(subset=['time'])
        df = df.set_index('time')
        df = df.rename(columns=self.rename_cols)
        name = df.stn.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.stn = name
        return df


    def select_station_data(self):
        """
        Select the station data for the station
        :return: DataFrame
        """
        df = self.prepare_meteo_data()
        st.write(df)
        station = ''
        if self.meteometa['station'] == 'grenchen':
            station = 'GRE'
        if self.meteometa['station'] == 'cressier':
            station = 'CRE'
        selected = df[df.stn == station]
        selected = selected.drop(columns=['stn'])
        return selected

    def resample_data(self, df):
        st.write('Resampling data')
        st.write(df)
        if self.datameta['resampling'] == 'hourly':
            df = df.resample('H').mean()
        if self.datameta['resampling'] == 'daily':
            df = df.resample('D').mean()
        return df

    def select_meteo_data(self):
        """
        Select the meteo data for the station
        :return: DataFrame
        """
        selected = self.select_station_data()
        resampled = self.resample_data(selected)
        path = Path.cwd() / st.session_state.foldername / 'meteo.db'
        st.write('Writing meteo data to database')
        st.write(resampled)
        with sqlite3.connect(path) as conn:
            resampled.to_sql(name='meteo', con=conn, if_exists='replace')