import pandas as pd
import numpy as np

def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    mask1 = df_copy['gmv'] / df_copy['price'] > df_copy['stock']
    df_copy['gmv'] = np.where(mask1, df_copy['price'] * df_copy['stock'], df_copy['gmv'])
    mask2 = np.floor(df_copy['gmv'] / df_copy['price']) - (df_copy['gmv'] / df_copy['price']) != 0
    df_copy['gmv'] = np.where(mask2, df_copy['price'] * np.floor(df_copy['gmv'] / df_copy['price']), df_copy['gmv'])
    return df_copy