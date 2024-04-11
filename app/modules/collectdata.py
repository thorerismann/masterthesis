from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
class DataExplorer:
    def __init__(self):
        self.station_path = Path.cwd() / 'database' / 'stations.csv'
        self.meteo_path = Path.cwd() / 'database' / 'meteo.csv'
        self.geodata_path = Path.cwd() / 'database' / 'geodata.csv'
        self.sensor_path = Path.cwd() / 'database' / 'sensorpoints'
        self.stations = None
        self.meteo = None
        self.geodata = None
        self.sensorpoints = None

    def load_data(self):
        self.stations = pd.read_csv(self.station_path)
        self.meteo = pd.read_csv(self.meteo_path)
        self.geodata = pd.read_csv(self.geodata_path)


class BambiData:
    def __init__(self):
        self.station_path = '/home/tge/masterthesis/app/database/stations.csv'
        self.meteo_path = '/home/tge/masterthesis/app/database/meteo.csv'
        self.geodata_path = '/home/tge/masterthesis/app/database/buffered_data.csv'
        self.landuse_path = '/home/tge/masterthesis/app/database/buffered_landuse.csv'
        self.sensor_path = Path.cwd() / 'database' / 'sensorpoints'
        self.load_data()

    def load_data(self):
        self.stations = pd.read_csv(self.station_path)
        self.stations.time = pd.to_datetime(self.stations.time)
        self.meteo = pd.read_csv(self.meteo_path)
        self.meteo.time = pd.to_datetime(self.meteo.time)
        self.geodata = pd.read_csv(self.geodata_path)
        self.landuse = pd.read_csv(self.landuse_path)

    def prepare_tropical_nights(self, loggers, threshold, meteovars=['airtemp'], geodata=None):
        # deal with geodata
        if geodata is not None:
            gdatas = []
            for k, v in geodata.items():
                gdata = self.geodata[self.geodata.dtype == k]
                gdata = gdata[gdata.buffer == v]
                gdata = gdata[gdata.logger.isin(loggers)]
                gdata = gdata[['mean', 'logger']].copy()
                gdata.columns = [k, 'logger']
                gdata.set_index('logger', inplace=True)
                gdata.fillna(0, inplace=True)
                gdatas.append(gdata)
            gdata = pd.concat(gdatas, axis=1)
            gdata.reset_index(inplace=True)

        # deal with meteo data
        mdata = self.meteo.copy()
        mdata = mdata[meteovars + ['time']]
        mdata = mdata.set_index('time').resample('D').min()
        mdata['lag_airtemp'] = mdata.airtemp.shift(1)
        mdata.dropna(inplace=True)
        mdata = mdata.sort_index()
        # deal with sensor data
        stationdata = self.stations.copy()
        stationdata = stationdata[stationdata.logger.isin(loggers)]
        pivoted = stationdata.pivot(index='time', columns='logger', values='temperature')
        pivoted = pivoted.sort_index()
        pivoted = pivoted.resample('D').min()
        stationdata = pivoted.reset_index().melt(id_vars='time', value_vars=loggers, var_name='logger', value_name='temperature')

        stationdata['tn'] = stationdata.temperature > threshold
        stationdata['doy']= stationdata.time.dt.dayofyear
        df = stationdata.merge(mdata, on='time', how='left')
        if geodata is not None:
            df = df.merge(gdata, on='logger', how='left')
        df = df.dropna()
        df.reset_index(inplace=True, drop=True)
        df['logger'] = df['logger'].astype('category')
        return df

    def split_data(self, data, split_type, split_date):
        if split_type == 'custom':
            train = data[data.time > split_date]
            test = data[data.time <= split_date]
            return train, test
        if split_type == 'random':
            train = data.sample(frac=0.8)
            test = data.drop(train.index)
            return train, test
        if split_type == 'time':
            train, test = TimeSeriesSplit(n_splits=2).split(data)
            return train, test

    def slice_times(self, data, period23, period22=None):
        if period22 is None:
            data = data[(data.time >= pd.Timestamp(period23[0])) & (data.time <= pd.Timestamp(period23[1])) | (data.time.year == 2022)]
        else:
            data = data[((data.time >= pd.Timestamp(period22[0])) & (data.time <= pd.Timestamp(period22[1]))) | ((data.time >= pd.Timestamp(period23[0])) & (data.time <= pd.Timestamp(period23[1])))]
        data.reset_index(inplace=True, drop=True)
        return data

