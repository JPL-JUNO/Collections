"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-09 17:54:36
"""
import pandas as pd
df.groupby('type').agg(['count', 'sum']).values
