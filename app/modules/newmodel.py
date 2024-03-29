import shutil
from pathlib import Path
import streamlit as st
from .collect_geodata import GeoDataCollector
from .sensordata import StationData
class PrepareData:

    @staticmethod
    def display_data_maker():
        with st.form(key='dataform'):
            buffers = st.multiselect('Select buffer sizes',
                                     [5, 10, 20, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500])
            data = st.multiselect('select variable', ['fitnahtemp', 'fitnahuhispace', 'fitnahuhistreet', 'dem', 'landuse'])
            foldername = st.text_input('Foldername', 'temp')
            # default = st.toggle('Default')
            dependent_data = st.multiselect('Select dependent data', ['city_index', 'uhi', 'airtemp'])
            stations = st.selectbox('Select stations', ['all stations', 'core stations', 'pilot stations'])

            submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            if (len(buffers) < 1) | (len(dependent_data) < 1) | (len(stations) < 1):
                st.error('Please select at least one buffer size, dependent data and station type')
            elif len(data) < 1:
                st.error('Please select at least two data sources')
            else:
                st.session_state['geochoices']= {'buffers': buffers, 'data': data}
                st.session_state['foldername'] = foldername
                st.session_state['datachoices'] = {'dependent': dependent_data, 'stations': stations}
                PrepareData.create_temp_folder(foldername)

    @staticmethod
    def create_temp_folder(foldername):
        temp_path = Path.cwd() / st.session_state.foldername
        if temp_path.exists() and temp_path.is_dir():
            shutil.rmtree(temp_path)
        temp_path.mkdir(exist_ok=True)

    @staticmethod
    def collect_new_data():
        st.header('Data Creator')
        PrepareData.display_data_maker()
        geochoices = st.session_state.get('geochoices')
        if geochoices:
            gdc = GeoDataCollector(geochoices)
            all_data = gdc.calculate_rasters()
            all_data.to_csv(Path.cwd() / st.session_state.foldername / 'bufferdata.csv')
            if 'landuse' in geochoices['data']:
                lu_data = gdc.get_landuse_stats()
                lu_data.to_csv(Path.cwd() / st.session_state.foldername / 'bufferdata_lu.csv')
            st.success(f"Buffered geo data created successfully in {str(Path.cwd() / st.session_state.foldername)}")
        datachoices = st.session_state.get('datachoices')
        if datachoices:
            psd = StationData(datachoices)
            psd.prepare_dependent_data()
            st.success(f"Station data created successfully in {str(Path.cwd() / st.session_state.foldername)}")



