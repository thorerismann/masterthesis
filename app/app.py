
import streamlit as st
import streamlit_folium as st_folium
from modules.mapviz import MapDisplay
from modules.newmodel import PrepareData


def start_display():
    with st.sidebar.form(key='my_form'):
        st.write('Please select an activity')
        activities = ['About', 'Data Explorer', 'New Model']
        choice = st.selectbox('Select Activity', activities)
        st.session_state.activity = choice
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        st.session_state.activity = choice

def main():
    st.header('Summer Temperatures in Biel / Bienne')

    start_display()

    activity = st.session_state.get('activity')

    if activity == 'Data Explorer':
        st.write('Data Explorer')
    if activity == 'New Model':
        st.write('New Model')
        PrepareData.collect_new_data()
    if activity == 'About':
        st.write('About')
        map_display = MapDisplay()
        st_folium.folium_static(map_display.plot_elevation_contour())
    if not activity:
        st.write('Please select an activity')


main()
