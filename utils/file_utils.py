import pandas as pd
import os
import re
import time

def create_excel_with_sheets(filename):
    """
    Create an Excel file with specified sheets.
    
    Args:
        filename: Path to the Excel file
    """
    illegal_chars = r'[:*?"<>|\[\]]'
    filename = re.sub(illegal_chars, '_', filename)
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    df = pd.DataFrame()
    while not os.path.exists(filename):
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for sheet in ['pearsonr', 'kendalltau', 'spearmanr', 'MAE', 'MSE']:
                    df.to_excel(writer, sheet_name=sheet, index=False)
            time.sleep(1)
        except Exception as e:
            print(f"Error occurred while creating file:{e}")
            time.sleep(1)

def append_to_excel(filename, values_val, values_err):
    """
    Append values to the Excel file in the specified format.
    
    Args:
        filename: Path to the Excel file
        values_val: List of mean values for each metric
        values_err: List of error values for each metric
    """
    sheet_names = ['pearsonr', 'kendalltau', 'spearmanr', 'MAE', 'MSE']
    if not os.path.exists(filename):
        create_excel_with_sheets(filename)
    try:
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            for i, sheet in enumerate(sheet_names):
                df_existing = pd.read_excel(filename, sheet_name=sheet)
                current_col = len(df_existing.columns)
                if df_existing.empty:
                    df_existing = pd.DataFrame({0: [values_val[i], values_err[i]]})
                else:
                    if len(df_existing) < 2:
                        df_existing = df_existing.reindex(range(2))
                    df_existing.at[0, current_col] = values_val[i]
                    df_existing.at[1, current_col] = values_err[i]
                df_existing.to_excel(writer, sheet_name=sheet, index=False, header=False)
    except Exception as e:
        raise Exception(f"Error occurred while writing to Excel file:{e}")