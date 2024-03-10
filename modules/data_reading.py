import pandas as pd

def read_csv(filepath, index_col=0):
    return pd.read_csv(filepath, index_col=index_col)

def merge_dataframes(df1, df2, on, how='left', fill_na=None, rename_columns=None):
    """Merge two DataFrames with optional filling of NaNs and renaming of columns."""
    df_merged = pd.merge(df1, df2, on=on, how=how)
    if fill_na:
        for column, value in fill_na.items():
            df_merged[column] = df_merged[column].fillna(value)
    if rename_columns:
        df_merged = df_merged.rename(columns=rename_columns)
    return df_merged