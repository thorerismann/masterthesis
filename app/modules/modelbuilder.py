import itertools
import sqlite3
from pathlib import Path
from random import random

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import streamlit as st


class ModelFactory:
    def __init__(self, geometa, datameta, meteometa, modelmeta):
        self.geometa = geometa
        self.datameta = datameta
        self.meteometa = meteometa
        self.modelmeta = modelmeta

    def base_model(self):
        if self.model_name == 'Random Forest':
            model = RandomForestRegressor()
        if self.model_name == 'Linear Regression':
            model = LinearRegression()
        return model

    def retrieve_geodata(self, geodata):
        with sqlite3.connect(Path.cwd() / st.session_state.foldername / 'geodata.db') as conn:
            geodata_name = geodata[0]
            if geodata[0] == 'temp':
                geodata_name = self.geometa['temp']
            query = f"SELECT mean,logger,buffer FROM {geodata_name} WHERE buffer IN {geodata[1]}"
            data = pd.read_sql(query, conn)
            return data

    def retrieve_stationdata(self):
        with sqlite3.connect(Path.cwd() / st.session_state.foldername / 'biel.db') as conn:
            query = f"SELECT * FROM {self.datameta['dependent']}"
            data = pd.read_sql(query, conn)
            return data
    def retrieve_meteodata(self, meteo):
        with sqlite3.connect(Path.cwd() / st.session_state.foldername / 'meteo.db') as conn:
            query = f"SELECT {meteo[0]}, {meteo[1]}, time FROM meteo"
            data = pd.read_sql(query, conn)
            return data

    def retrieve_interactions(self, feature, meteo):
        pass# generate interaction terms

    def limit_combinations(self, max_per_feature):
        geo_combinations = []
        temp_combinations = []

        # Limit combinations for 'feature' and 'buffer' pairs
        for feature in self.geometa['feature']:
            # Get all possible buffer combinations for this feature
            buffer_combinations = list(itertools.combinations(self.geometa['buffers'], 2))
            # Append the selected combinations to the result list
            geo_combinations.extend([(feature, buffer) for buffer in buffer_combinations])

        # Assuming similar logic for 'temp' combinations (only 2 buffer sizes)
        for buffer_combination in itertools.combinations(self.geometa['buffers'], 2):
            temp_combinations.append(('temp', buffer_combination))

        return geo_combinations, temp_combinations

    def create_combinations(self):
        meteo_vars = [x for x in self.meteometa['data'] if x != 'temperature']

        # Generate combinations of length 1 and 2 for meteorological variables
        meteo_combinations = []
        for r in range(1, 3):
            meteo_combinations.extend(itertools.combinations(meteo_vars, r))

        # Filter combinations to only include at most one variable other than temperature
        meteo_combinations_temp = [meteo_combo + ('temperature',) for meteo_combo in meteo_combinations if
                                   len(meteo_combo) <= 1]
        print("length of meteo combinations:", len(meteo_combinations_temp))

        geo_combinations, temp_combinations = self.limit_combinations(2)
        print("length of temp combinations:", len(temp_combinations))
        print("length of geo combinations:", len(geo_combinations))

        # Concatenate the combinations
        all_combinations = list(itertools.product(meteo_combinations_temp, temp_combinations, geo_combinations))
        st.write(f"Total number of combinations: {len(all_combinations)}")
        return all_combinations

    def get_data(self, combination):
        st.write(len(combination))
        meteo, temp, geo = combination
        meteo_data = self.retrieve_meteodata(meteo)
        temp_data = self.retrieve_geodata(temp)
        geo_data = self.retrieve_geodata(geo)
        station_data = self.retrieve_stationdata()
        return meteo_data, temp_data, geo_data, station_data

    def assemble_model(self, meteodata, tempdata, geodata, stationdata):
        model = self.base_model()
        model.fit(data[0], data[1])
        return model



class Model:
    def __init__(self, model, data):
        self.model = model
        self.data = data


class FactoryManager:
    def __init__(self, geometa, datameta, meteometa, modelmeta):
        self.geometa = geometa
        self.datameta = datameta
        self.meteometa = meteometa
        self.modelmeta = modelmeta

    def create_model(self):
        mf = ModelFactory(self.geometa, self.datameta, self.meteometa, self.modelmeta)
        combis = st.session_state.get('combinations')
        if not combis:
            st.session_state['combinations'] = mf.create_combinations()
        data = mf.get_data(st.session_state.combinations[0])
        st.write(data[0])
        st.write(data[1])
        st.write(data[2])
        st.write(data[3])