import pandas as pd

def read_csv(filepath, index_col=0):
    return pd.read_csv(filepath, index_col=index_col)

def merge_dataframes(df1, df2, on, how='left', fill_na=None, rename_columns=None):
    df_merged = pd.merge(df1, df2, on=on, how=how)
    if fill_na is not None:
        for column, fill_value in fill_na.items():
            df_merged[column] = df_merged[column].fillna(fill_value)
    if rename_columns is not None:
        df_merged = df_merged.rename(columns=rename_columns)
    return df_merged
