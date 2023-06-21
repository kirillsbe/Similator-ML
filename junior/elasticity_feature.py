import pandas as pd
import numpy as np
from scipy import stats

def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby('sku')
    elasticity_df = pd.DataFrame.from_dict({'sku':[], 'elasticity':[]})
    for n,g in groups:
        X = np.log(g['qty'] + 1)
        y = g['price']
        res = stats.linregress(X, y)
        temp_df = {'sku': g['sku'].iloc[0], 'elasticity':res.rvalue**2}
        elasticity_df = elasticity_df.append(temp_df, ignore_index = True)
    elasticity_df['sku'] = elasticity_df['sku'].astype(np.int64)
    return elasticity_df