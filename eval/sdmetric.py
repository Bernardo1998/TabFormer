## Evaluate Generation Performance
import pandas as pd

from sdmetrics import load_demo
from sdmetrics.reports.single_table import QualityReport

real_data = pd.read_csv('data/dvlog/preprocessed/acoustic.encoded.csv')
synth_data = pd.read_csv('sampled/sampled_dvlog_cleanedAndSorted.csv')
#real_data, synthetic_data = df, df_generated

metadata = {'primary_key':None, "columns":{}}
numerical_cols = ['Acoustic'+str(i) for i in range(25)] + ["User", "Timestamp"]

for c in real_data.columns:
  if c in numerical_cols:
    metadata['columns'][c] = {'sdtype':'numerical'}
  else:
    metadata['columns'][c] = {'sdtype':'categorical'}

my_report = QualityReport()
my_report.generate(real_data, synth_data, metadata)

print(my_report.get_details(property_name='Column Shapes')) 