import itertools

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import streamlit as st


class ModelFactory:
    def __init__(self, geometa, datameta, meteometa, modelmeta):
        self.geometa = geometa
        self.datameta = datameta
        self.meteometa = meteometa

    def base_model(self):
        if self.model_name == 'Random Forest':
            model = RandomForestRegressor()
        if self.model_name == 'Linear Regression':
            model = LinearRegression()
        return model

    def retrieve_geodata(self, feature, buffer):
        pass

    def retrieve_stationdata(self, dependent):
        pass

    def retrieve_meteodata(self):
        pass

    def retrieve_interactions(self, feature, meteo):
        pass# generate interaction terms

    def create_model(self):
        meteo_vars = [var for var in self.meteometa['data'] if var != 'temperature']
        meteo_combinations = itertools.combinations(meteo_vars)
        st.write(len(meteo_combinations))
        st.write(meteo_combinations)
        # Generate combinations for geo variables with up to 2 buffers each
        # geo_combinations = list(itertools.product(self.geometa['feature'], self.geometa['buffer']))

        # # Add temperature to each meteo combination
        # meteo_combinations = [temperature_alone + list(combo) for combo in meteo_combinations]
        #
        # # Create combinations of meteo and geo variables
        # final_combinations = list(itertools.product(meteo_combinations, geo_combinations))
        #
        # # Filter combinations based on the max number of variables allowed
        # filtered_combinations = [combo for combo in final_combinations if
        #                          len(combo[0]) + len(combo[1]) <= self.meteometa['maxvars']]
        # st.write(filtered_combinations)
        #
        # # Log the number of valid combinations
        # st.write(
        #     f"There are {len(filtered_combinations)} possible combinations of features, buffers, and meteo data following the new rules.")
        #

class FactoryManager:
    def __init__(self, geometa, datameta, meteometa, modelmeta):
        self.geometa = geometa
        self.datameta = datameta
        self.meteometa = meteometa
        self.modelmeta = modelmeta

    def create_model(self):
        mf = ModelFactory(self.geometa, self.datameta, self.meteometa, self.modelmeta)
        mf.create_model()
        return None