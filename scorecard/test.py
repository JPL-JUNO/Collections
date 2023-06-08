"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-05 10:38:38
"""
from method.temp.temp import ReadData
from method.process import ScoreCardProcess
r_d = ReadData('./train.csv')
raw_data = r_d.read_table()

model = ScoreCardProcess(raw_data, label='subscribe', show_plot=False)
model.check_missing_value(print_result=True)
model.pro_check_data(remove_blank=True)
model.pro_feature_filter()
