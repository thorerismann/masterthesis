import shutil
from pathlib import Path
import streamlit as st
from .collect_geodata import GeoDataCollector

class PrepareData:

    @staticmethod
    def display_data_maker():
        with st.form(key='dataform'):
            buffers = st.multiselect('Select buffer sizes',
                                     [5, 10, 20, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500])
            data = st.multiselect('select variable', ['fitnahtemp', 'fitnahuhispace', 'fitnahuhistreet', 'dem', 'landuse'])
            foldername = st.text_input('Foldername', 'temp')
            default = st.toggle('Default')
            submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            if len(buffers) < 1:
                st.error('Please select at least one buffer size')
            elif len(data) < 1:
                st.error('Please select at least two data sources')
            else:
                st.session_state['choices']= {'buffers': buffers, 'data': data}
                st.session_state['foldername'] = foldername
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
        choices = st.session_state.get('choices')
        if choices:
            gdc = GeoDataCollector(choices)
            all_data = gdc.calculate_rasters()
            all_data.to_csv(Path.cwd() / st.session_state.foldername / 'bufferdata.csv')
            if 'landuse' in choices['data']:
                lu_data = gdc.get_landuse_stats()
                lu_data.to_csv(Path.cwd() / st.session_state.foldername / 'bufferdata_lu.csv')
            st.write(f"Data created successfully in {str(Path.cwd() / st.session_state.foldername)}")
