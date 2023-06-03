import pandas as pd


def fillna_with_mean(df: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[target] = df_copy.groupby(group)[target].transform(
        lambda x: x.fillna(round(x.mean()))
    )
    return df_copy
