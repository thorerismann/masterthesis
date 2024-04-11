import json
import shutil

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from .collectdata import BambiData
import bambi as bmb
import warnings
import streamlit as st
import pymc as pm
from pathlib import Path
import xarray as xr
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning, message="Detected reference count inconsistency after CVM construction")

class BambiModelMaker:
    def __init__(self, train, test):
        self.train = train
        self.test = test
    def simple_bernoulli_model(self):
        model = bmb.Model(formula="tn ~ airtemp + (1 + airtemp|logger)", data=self.train, family='bernoulli')
        return model, "tn ~ airtemp + (1 + airtemp|logger)"

    def simple_bernoulli_lag_model(self):
        model = bmb.Model(formula="tn ~ airtemp + lag_airtemp + (1 + airtemp|logger)", data=self.train, family='bernoulli')
        return model, "tn ~ airtemp + lag_airtemp + (1 + airtemp|logger)"

    def simple_continuous_model(self):
        model = bmb.Model(formula="temperature ~ airtemp + (1 + airtemp|logger)", data=self.train, family='gaussian')
        return model, "temperature ~ airtemp + (1 + airtemp|logger)"

    def simple_continuous_lag_model(self):
        model = bmb.Model(formula="temperature ~ airtemp + lag_airtemp + (1 + airtemp|logger)", data=self.train, family='gaussian')
        return model, "temperature ~ airtemp + lag_airtemp + (1 + airtemp|logger)"

    def fit_model(self, model, draws, tune, target_accept):
        return model.fit(draws=draws, tune=tune, target_accept=target_accept)

    def complete_model(self, model, idata):
        prior_pred = model.prior_predictive(draws=500)
        if 'prior' not in idata.groups():
            idata.add_groups(dict(prior=prior_pred.prior))
        if 'prior_predictive' not in idata.groups():
            idata.add_groups(dict(prior_predictive=prior_pred.prior_predictive))
        if 'posterior_predictive' not in idata.groups():
            post_pred = pm.sample_posterior_predictive(idata.posterior, model=model.backend.model)
            idata.add_groups({'posterior_predictive': post_pred.posterior_predictive})
        return idata

    def save_model(self, idata, model_string):
        path = Path.cwd() / 'models' / 'bambi' / st.session_state['bambi_params']['model_name']
        path.mkdir(parents=True, exist_ok=True)
        for key, value in idata.items():
            value.to_netcdf(path / f'{key}.nc')
        data_to_save = {k:v for k,v in st.session_state['bambi_params'].items() if k in ['modeltype', 'distro', 'loggers', 'threshold', 'split_type', 'model_name']}
        data_to_save.update({'model_string': model_string, 'observed': st.session_state['bambi_params']['observed']})
        with open(path / 'model_params.json', 'w') as f:
            json.dump(data_to_save, f)


class BambiBuildInterface:
    @staticmethod
    def display_input():
        st.header('Bambi Model Builder')
        with st.form(key='bambi_form'):
            modeltype = st.selectbox('Model Type', ['simple', 'lag'])
            distro = st.selectbox('Distribution', ['bernoulli', 'gaussian'])
            loggers = st.multiselect('Loggers', [str(x) for x in range(201, 239)])
            threshold = st.number_input('Tropical Night Threshold', value=20.0, min_value=20.0, max_value=25.0)
            split_type = st.selectbox('Split Type', ['yearly', 'random', 'timeseries'])
            start_23 = st.date_input('Start Date 2023', pd.Timestamp('2023-06-15'))
            tune = st.number_input('Tune', value=1000, min_value=100, max_value=10000)
            target_accept = st.number_input('Target Accept', value=0.98, min_value=0.8, max_value=0.99)
            draw = st.number_input('Draws', value=1000, min_value=100, max_value=10000)
            model_name = st.text_input('Model Name', 'bambi_model')
            bambi_sumbit = st.form_submit_button('Create Bambi Model')
        if bambi_sumbit:
            period_23 = [pd.Timestamp(start_23), pd.Timestamp('2023-09-15')]
            period_22 = [pd.Timestamp('2022-07-09'), pd.Timestamp('2023-09-15')]
            split_type = 'custom'
            split_date = pd.Timestamp('2022-12-31')
            if distro == 'bernoulli':
                observed = 'tn'
            if distro == 'gaussian':
                observed = 'temperature'
            st.session_state['bambi_params'] = {
                'modeltype': modeltype,
                'distro': distro,
                'loggers': loggers,
                'threshold': threshold,
                'split_type': split_type,
                'split_date': split_date,
                'period_22': period_22,
                'period_23': period_23,
                'tune': tune,
                'target_accept': target_accept,
                'draws': draw,
                'model_name': model_name,
                'observed': observed
            }

    @staticmethod
    def create_simple_model():
        params = st.session_state.get('bambi_params')
        bambiboy = BambiData()
        bambiboy.load_data()

        df = bambiboy.prepare_tropical_nights([int(x) for x in params['loggers']], params['threshold'])
        sliced = bambiboy.slice_times(df, params['period_22'], params['period_23'])
        train, test = bambiboy.split_data(sliced, params['split_type'], params['split_date'])
        path = Path.cwd() / 'models' / 'bambi' / params['model_name']
        if path.is_dir():
            shutil.rmtree(path)

        # Now recreate the directory
        path.mkdir(parents=True, exist_ok=True)

        # remove file if it exists
        train.to_csv(path / 'train.csv')
        test.to_csv(path / 'test.csv')
        bmm = BambiModelMaker(train, test)
        model = None
        string = None
        if (params['modeltype'] == 'simple') & (params['distro'] == 'bernoulli'):
            model, string = bmm.simple_bernoulli_model()
        if (params['modeltype'] == 'lag') & (params['distro'] == 'bernoulli'):
            model, string = bmm.simple_bernoulli_lag_model()
        if (params['modeltype'] == 'simple') & (params['distro'] == 'gaussian'):
            model, string = bmm.simple_continuous_model()
        if (params['modeltype'] == 'lag') & (params['distro'] == 'gaussian'):
            model, string = bmm.simple_continuous_lag_model()
        idata, model = cache_model(bmm, model, params)
        idata = bmm.complete_model(model, idata)
        bmm.save_model(idata, string)
    @staticmethod
    def main():
        BambiBuildInterface.display_input()
        if st.session_state.get('bambi_params'):
            st.success('Parameters Set')
            st.write('Please be patient, this may take a while.')
            BambiBuildInterface.create_simple_model()
            st.success('Model Created and Saved')

@st.cache_resource
def cache_model(_bambi_model_maker, _model, params):
    return _bambi_model_maker.fit_model(_model, params['draws'], params['tune'], params['target_accept']), _model


class LoadBambiModel:
    @staticmethod
    def load_models(model_name):
        path = Path.cwd() / 'models' / 'bambi' / model_name
        idata = az.InferenceData()
        for file in path.iterdir():
            if '.nc' in file.suffix:
                ds = xr.open_dataset(file)
                idata.add_groups({file.stem: ds})
                st.write(f'{file.stem} loaded')
        datadict = {}
        st.write('hello')
        for file in path.iterdir():
            if '.csv' in file.suffix:
                data = pd.read_csv(file)
                data.logger = pd.Categorical(data.logger)
                data.time = pd.to_datetime(data.time)
                data.tn = data.tn.astype(int)
                data.lag_airtemp = data.lag_airtemp.astype(float)
                data.airtemp = data.airtemp.astype(float)
                if 'train' in file.stem:
                    datadict['train'] = data
                if 'test' in file.stem:
                    datadict['test'] = data
        session_variables = json.load(open(path / 'model_params.json', 'r'))
        st.session_state['bambi_model_params'] = session_variables
        model = bmb.Model(formula= session_variables['model_string'], data=datadict['train'], family=session_variables['distro'])
        # add some bounds to my priors

        return idata, model, datadict

    @staticmethod
    def make_predictions(model, idata, test):
        return model.predict(idata=idata, data=test, inplace=False)

@st.cache_resource
def load_model(model_name):
    _mymodel = LoadBambiModel.load_models(model_name)
    return _mymodel

class BambiUseInterface:

    @staticmethod
    def display_input():
        st.header('Bambi Model Viewer')
        model_parent = Path.cwd() / 'models' / 'bambi'
        model_list = [x.name for x in model_parent.iterdir() if x.is_dir()]
        model_list = ['None'] + model_list
        model_to_load = st.selectbox('Select Model to Load', model_list)
        if model_to_load != 'None':
            st.session_state['model_name'] = model_to_load

    @staticmethod
    def main():
        BambiUseInterface.display_input()
        if st.session_state.get('model_name'):
            st.subheader("Loading model parameters")
            idata, model, datadict = LoadBambiModel.load_models(st.session_state.model_name)
            st.session_state['bambi_model_params']

            st.success('Model Loaded')
            if st.session_state.get('bambi_model_params'):
                if st.toggle('view basic outputs'):
                    observed = st.session_state.bambi_model_params['observed']
                    BambiUseInterface.test_stuff(idata, datadict)
                    BambiUseInterface.show_histograms(idata, datadict)
                    BambiUseInterface.plot_priors(idata)
                    BambiUseInterface.plot_ppcs(idata)
                    BambiUseInterface.plot_trace(idata, coords=None, varnames='airtemp|logger', combined=False,compact=False)
                    if st.session_state.bambi_model_params['distro'] == 'bernoulli':
                        BambiUseInterface.view_predictions_bernoulli(idata, datadict, observed)
                    if st.session_state.bambi_model_params['distro'] == 'gaussian':
                        BambiUseInterface.view_predictions_continuous(idata, datadict, observed)


    @staticmethod
    def test_stuff(idata, datadict):
        st.subheader('place for testing')
        st.write('place for testing')
        st.write(az.summary(idata))

    @staticmethod
    def show_histograms(idata, datadict):
        obs_data = idata.observed_data.to_dataframe()
        sns.histplot(obs_data)
        st.pyplot(plt)
        plt.clf()
        airtemp = datadict['train'].airtemp
        sns.histplot(airtemp)
        st.pyplot(plt)
        plt.clf()

    @staticmethod
    def view_predictions_continuous(idata, datadict, observed):
        df = az.summary(idata)
        global_airtemp = df.loc['airtemp', 'mean']
        global_intercept = df.loc['Intercept', 'mean']
        logger_data = df.loc[df.index.str.contains('1|logger'), 'mean'].copy()
        logger_data.index = logger_data.index.str[-4:-1]
        logger_data = logger_data.reset_index()
        logger_data.columns = ['logger', 'logger_intercept']
        logger_data.logger = pd.to_numeric(logger_data.logger, errors='coerce')
        logger_data.dropna(inplace=True)
        logger_data.logger = logger_data.logger.astype(int)
        mydata = datadict['train'].copy()
        newdata = mydata.merge(logger_data, on='logger', how='left')
        newdata['predicted_temperature'] = global_intercept + global_airtemp * newdata.airtemp + newdata.logger_intercept
        st.subheader('Predicted vs Observed air temperature at each station')
        sns.scatterplot(data=newdata, x='temperature', y='predicted_temperature', hue='logger')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

    @staticmethod
    def plot_priors(idata, var_names=None):
        st.subheader('Prior Distributions')
        az.plot_density(idata, group='prior', var_names=var_names)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

    @staticmethod
    def plot_ppcs(idata, var_names=None):
        st.subheader('Posterior Predictive Checks - doesnt work! quite right')
        az.plot_ppc(idata, group='prior', var_names=var_names)
        st.pyplot(plt)
        plt.clf()
        az.plot_ppc(idata, group='posterior', var_names=var_names)
        st.pyplot(plt)
        plt.clf()

    @staticmethod
    def plot_trace(idata, coords, varnames, combined, compact):
        st.subheader('Trace Plots')
        az.plot_trace(idata, coords=coords, var_names=varnames, legend=True, compact=compact, combined=combined)
        plt.title('Trace Plot, station specific airtemp')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

    @staticmethod
    def view_predictions_bernoulli(idata, mydata, observed):
        st.subheader('Predictions')
        df_tn = idata.posterior_predictive[observed].mean(dim=['chain', 'draw']).to_dataframe().reset_index()
        train = mydata['train'].reset_index(drop=False).rename(columns={'index': f'{observed}_obs'})
        df = df_tn.merge(train, on=f'{observed}_obs')
        df = df.rename(columns={f'{observed}_x': f'{observed}_pred', f'{observed}_y': f'{observed}_observ'})
        df_hm = df.copy()
        df_hm.time = df_hm.time.dt.day_of_year
        df_pred = df_hm.pivot(index='logger', columns='time', values=f'{observed}_pred')
        st.write(df_pred)
        # Plot tn_x as a line
        sns.heatmap(data=df_pred, cmap="magma", cbar=False)
        st.pyplot(plt)
        # Plot tn_y as points
        plt.clf()
        df_observ = df_hm.pivot(index='logger', columns='time', values=f'{observed}_observ')
        sns.heatmap(data=df_observ, cmap=["blue", "red"], cbar=False)
        plt.title('some chart')
        st.pyplot(plt)
        plt.clf()