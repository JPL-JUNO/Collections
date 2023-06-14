"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-14 11:08:33
"""
import os
os.chdir('C:/Notes/Collections/ScoreCard2')
from sc_model import SCModel
from data_loader import DataLoader
dl = DataLoader('./iris.data')
data = dl.read_data()

model = SCModel(data, label='label')
model.data_view()
