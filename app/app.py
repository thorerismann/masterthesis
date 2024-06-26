
import streamlit as st
import streamlit_folium as st_folium
from modules.mapviz import MapDisplay
from modules.newmodel import ModelBuilder
from modules.bambimodels import BambiBuildInterface, BambiUseInterface


def start_display():
    with st.sidebar.form(key='my_form'):
        st.write('Please select an activity')
        activities = ['About', 'Data Explorer', 'New Bambi Model', 'New regression model', 'View Model']
        choice = st.selectbox('Select Activity', activities)
        st.session_state.activity = choice
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        st.session_state.activity = choice

def clear_session():
    with st.container():
        if st.sidebar.button('Clear Session'):
            st.session_state.clear()
            st.cache_data.clear()

def main():
    st.header('Summer Temperatures in Biel / Bienne')
    clear_session()
    start_display()

    activity = st.session_state.get('activity')

    if activity == 'Data Explorer':
        st.write('Data Explorer')
    if activity == 'New Regression Model':
        st.write('New Regression Model')
        ModelBuilder.main()
    if activity == 'New Bambi Model':
        BambiBuildInterface.main()
        st.write('Now select view model to see the results.')
    if activity == 'View Model':
        BambiUseInterface.main()
    if activity == 'About':
        st.write('About')
        map_display = MapDisplay()
        st_folium.folium_static(map_display.plot_elevation_contour())
    if not activity:
        st.write('Please select an activity')


main()
