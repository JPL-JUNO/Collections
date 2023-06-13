"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-05 10:38:38
"""
import os
os.chdir('C:/Notes/Collections/scorecard')
from method.temp.temp import ReadData
from method.process import ScoreCardProcess
r_d = ReadData('./train.csv')
raw_data = r_d.read_table()

model = ScoreCardProcess(raw_data, label='subscribe', show_plot=False)
model.check_missing_value(print_result=True)
model.pro_check_data(remove_blank=True)
model.pro_feature_filter(positive='yes|1')

rm_reason = model.rm_reason
epo = model.epo
sample = model.data.head().T

# Stage 3 衍生特征，编码

# Stage 4 特征工程
model.feature_process()
