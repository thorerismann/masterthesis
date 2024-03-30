import shutil
from pathlib import Path
import streamlit as st
from .collect_geodata import GeoDataCollector
from .sensordata import StationData
from .modelbuilder import FactoryManager
from .meteo import MeteoData
class PrepareData:

    @staticmethod
    def display_data_maker():
        with st.form(key='dataform'):
            buffers = st.multiselect('Select buffer sizes',
                                     [5, 10, 20, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500])
            geodata = st.multiselect('select variable', ['fitnahtemp', 'fitnahuhispace', 'fitnahuhistreet', 'dem', 'landuse'])
            foldername = st.text_input('Foldername', 'temp')
            # default = st.toggle('Default')
            dependent_data = st.selectbox('Select dependent data', ['city_index', 'uhi', 'airtemp'])
            stations = st.selectbox('Select stations', ['all stations', 'core stations', 'pilot stations'])
            resampling = st.selectbox('Resampling', ['daily', 'hourly', 'none'])

            meteodata = st.multiselect('Select meteo data', ['temperature', 'humidity', 'windspeed', 'winddir', 'precipitation'])
            meteostation = st.selectbox('Select meteo station', ['grenchen', 'cressier'])

            submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            if (len(buffers) < 1) | (len(dependent_data) < 1) | (len(stations) < 1):
                st.error('Please select at least one buffer size, dependent data and station type')
            elif len(geodata) < 1:
                st.error('Please select at least two data sources')
            else:
                st.session_state['geometa']= {'buffers': buffers, 'data': geodata}
                st.session_state['foldername'] = foldername
                st.session_state['datameta'] = {'dependent': dependent_data, 'stations': stations, 'resampling': resampling}
                st.session_state['meteometa'] = {'data': meteodata, 'station': meteostation}
                PrepareData.create_temp_folder()

    @staticmethod
    def create_temp_folder():
        temp_path = Path.cwd() / st.session_state.foldername
        if temp_path.exists() and temp_path.is_dir():
            shutil.rmtree(temp_path)
        temp_path.mkdir(exist_ok=True)

    @staticmethod
    def collect_new_data():
        st.header('Data Creator')
        PrepareData.display_data_maker()
        geochoices = st.session_state.get('geometa')
        if geochoices:
            gdc = GeoDataCollector(geochoices)
            gdc.save_buffered_data()
            if 'landuse' in geochoices['data']:
                lu_data = gdc.get_landuse_stats()
                lu_data.to_csv(Path.cwd() / st.session_state.foldername / 'bufferdata_lu.csv')
            st.success(f"Buffered geo data created successfully in {str(Path.cwd() / st.session_state.foldername)}")
        datachoices = st.session_state.get('datameta')
        if datachoices:
            psd = StationData(datachoices)
            psd.prepare_dependent_data()
            st.success(f"Station data created successfully in {str(Path.cwd() / st.session_state.foldername)}")
            st.session_state['subactivity'] = 'model_building'
        meteochoices = st.session_state.get('meteometa')
        if meteochoices:
            md = MeteoData(meteochoices, datachoices)
            md.select_meteo_data()
            st.success(f"Meteo data created successfully in {str(Path.cwd() / st.session_state.foldername)}")


class ModelBuilder:

    @staticmethod
    def create_base_model():
        st.write('Model Building')
        with st.form(key='model_form'):
            model = st.selectbox('Select Model', ['Random Forest', 'Linear Regression'])
            modelruns = st.number_input('Number of runs', 1, 100, value=10)
            max_variables = st.number_input('Max Variables', 1, 8, value=4)
            submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            st.session_state['model'] = {'model': model, 'runs': modelruns, 'maxvars': max_variables}
    @staticmethod
    def main():
        st.header('Model Building')
        PrepareData.collect_new_data()
        if st.session_state.get('subactivity') == 'model_building':
            ModelBuilder.create_base_model()
        modelmeta = st.session_state.get('model')
        if modelmeta:
            fm = FactoryManager(st.session_state['geometa'], st.session_state['datameta'], st.session_state['meteometa'], modelmeta)
            fm.create_model()
