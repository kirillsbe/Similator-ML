import numpy as np
import pandas as pd

def agg_comp_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    ...
    """
    groups = df.groupby('sku')
    new_df = pd.DataFrame.from_dict({'sku':[], 'agg':[], 'base_price':[], 'comp_price':[]})
    for index,gr in groups:
        if gr['agg'].iloc[0] == 'avg':
            new_comp_price = gr['comp_price'].mean()
        elif gr['agg'].iloc[0] == 'med':
            new_comp_price = gr['comp_price'].median()
        elif gr['agg'].iloc[0] == 'min':
            new_comp_price = gr['comp_price'].min()
        elif gr['agg'].iloc[0] == 'max':
            new_comp_price = gr['comp_price'].max()
        elif gr['agg'].iloc[0] == 'rnk' and gr['rank'].min() != -1:
            new_comp_price = gr.loc[gr['rank'] == gr['rank'].min()]['comp_price'].values[0]
        elif gr['rank'].min() == -1:
            new_comp_price = np.nan
        temp_df = {'sku': gr['sku'].iloc[0],
                   'agg':gr['agg'].iloc[0],
                   'base_price':gr['base_price'].iloc[0],
                   'comp_price':new_comp_price}
        new_df = new_df.append(temp_df, ignore_index = True)
        new_df['sku'] = new_df['sku'].astype(int)
    new_df['new_price'] = np.where(new_df['comp_price'].isnull(), new_df['base_price'],
                          np.where(np.abs(new_df['comp_price'] - new_df['base_price']) / new_df['base_price'] <= 0.2,
                          new_df['comp_price'], new_df['base_price']))
    return new_df
