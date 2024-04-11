from dataprep import BambiDataPrep
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tn_threshold = 21
loggers = [x for x in range(201, 208)]
period_23 = [pd.Timestamp('2023-07-01'), pd.Timestamp('2023-08-31')]
period_22 = [pd.Timestamp('2022-07-01'), pd.Timestamp('2022-08-31')]

# get the data
bambiprep = BambiDataPrep(tn_threshold, loggers, period_23, period_22)
data = bambiprep.get_data()
train, test = bambiprep.split_data(data, pd.to_datetime('2022-12-31'))
print('loaded')

sns.lineplot(data=train, x='time', y='tn', hue='logger')
plt.show()
sns.lineplot(data=test, x='time', y='tn', hue='logger')
plt.show()
print('seen')
