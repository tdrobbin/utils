"""
utilites used throughout the cap package
"""
import pandas as pd
import numpy as np
import tarfile
import logging
import re
import datetime
from pathlib import Path
import pandas as pd

import cap
from cap import config
from loguru import logger
from pandas import DataFrame as D, Series as S, Timestamp as T
from fuzzywuzzy import process
from tqdm.auto import tqdm
from fuzzywuzzy import fuzz, process


# def to_sql(df: pd.DataFrame, table: str):
#     """
#     write a dataframe to a table in the database
#
#     Args:
#         df (pd.DataFrame): dataframe to be written
#         table (str): table to be written to
#     """
#     schema = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", con=config['test_db_uri'])
#
#     df = df[[schema.columns]]
#
#     df.to_sql(table, con=config['test_db_uri'], if_exists='append', index=False)


# def get_missing_records(old_df: pd.DataFrame, new_df: pd.DataFrame, primary_key_columns: list):
#     """
#     finds records in new_df that are missing from old_df, only considering the columns in primary_key_columns
#     then returns a df of the missing records. ignores index

#     Args:
#         old_df (pd.DataFrame): old dataframe
#         new_df (pd.DataFrame): new dataframe
#         primary_key_columns (list): list of columns to consider when determining if a record is missing

#     Returns:
#         pd.DataFrame: dataframe of missing records
#     """


def get_missing_records(old_df: pd.DataFrame, new_df: pd.DataFrame, primary_key_columns: list) -> pd.DataFrame:
    """
    finds records in new_df that are missing from old_df, only considering the columns in primary_key_columns
    then returns a df of the missing records. ignores index

    Args:
        old_df (pd.DataFrame): old dataframe
        new_df (pd.DataFrame): new dataframe
        primary_key_columns (list): list of columns to consider when determining if a record is missing

    Returns:
        pd.DataFrame: dataframe of missing records
    """
    old_df_set = old_df.copy().set_index(primary_key_columns)
    new_df_set = new_df.copy().set_index(primary_key_columns)
    
    # get the difference between two dataframes
    diff_df = new_df_set.loc[new_df_set.index.difference(old_df_set.index)]

    # reset index
    return diff_df.reset_index()




def to_sql_list(strings):
    sql_list = "("
    sql_list += ", ".join(["'" + s.replace("'", "''") + "'" for s in strings])
    sql_list += ")"
    return sql_list

def df_info(df):
    from IPython.display import display

    display(df.info(verbose=True))
    display(df.describe(include='all'))
    display(df.sample(10))


def display_dfs_inline(dfs, captions=None, margin=5):
    from functools import reduce
    from IPython.display import display_html
    
    captions = [''] * len(dfs) if captions is None else captions
    stylers = [D(df).style.set_table_attributes(f'style="display:inline; margin:{margin}px;"').set_caption(c) for df, c in zip(dfs, captions)]
    display_html(reduce(lambda x, y: x + y, (s._repr_html_() for s in stylers)), raw=True)


# def smart_convert_df(df, infer_objects=True):
#     if infer_objects:
#         df = df.infer_objects().copy()
#     else:
#         df = df.copy()
#
#     for c in df.columns:
#         try:
#             if df[c].dtype == object:
#                 df[c] = df[c].fillna(np.nan).astype(float)
#         except (ValueError, TypeError):
#             pass
#
#     return df


def extract_targz(fpath):
    extracted_fpath = str(fpath) + '_extracted'

    # https://stackoverflow.com/questions/30887979/i-want-to-create-a-script-for-unzip-tar-gz-file-via-python
    if (fpath.endswith("tar.gz")):
        tar = tarfile.open(fpath, "r:gz")
        tar.extractall(extracted_fpath)
        tar.close()

    elif (fpath.endswith("tar")):
        tar = tarfile.open(fpath, "r:")
        tar.extractall(extracted_fpath)
        tar.close()

    logger.info('extracted : {} -> {}'.format(fpath, extracted_fpath))
    return extracted_fpath


def to_posix_ms(ts, unit='ms'):
    mul_fac = {
        's': 1,
        'ms': 1e3,
        'ns': 1e6
    }

    return str(int(np.round(pd.Timestamp(ts).timestamp() * mul_fac[unit], decimals=0)))


def make_column_name_sql_pd_safe(s):
    return re.sub(r'[^0-9a-zA-Z\s]+', '', s).lower().lstrip().rstrip().replace(' ', '_')


def make_column_value_numeric_safe(s):
    return re.sub(r'[^0-9.]+', '', s)


def make_str_alphanumeric(s: str):
    """
    make a string alphanumeric. strip leading and trailing spaces, convert multiple consecutive spaces to single space, 
    replace spaces and any non alphanumeric with underscores, and convert to lowercase
    
    Args:
        s (str): string to be made alphanumeric
    """
    s = s.lower().lstrip().rstrip()
    s = re.sub(r'\s+', ' ', s)
    # s = re.sub(r'[^a-zA-Z0-9]+', '_', s)
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)


    return s

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# def clean_column_names(df):
#     # Replace percentage sign with '_pct_'
#     df.columns = df.columns.str.replace('%', '_pct_')
#
#     # Convert column names to snake case
#     df.columns = df.columns.map(camel_to_snake)
#
#     # Replace various characters with underscores
#     df.columns = df.columns.str.replace('[\.\s\-\/]', '_')
#
#     # Remove unwanted characters
#     df.columns = df.columns.str.replace('[\(\)\?\!\';:,=\+\*&^$#@]', '')
#
#     # Replace two or more consecutive underscores with a single underscore
#     df.columns = df.columns.str.replace('_+', '_')
#
#     # Remove leading and trailing underscores
#     df.columns = df.columns.str.strip('_')

def clean_column_name(column_name):
    # Replace percentage sign with '_pct_'
    column_name = column_name.replace('%', '_pct_')

    # Convert column names to snake case
    column_name = camel_to_snake(column_name)

    # Replace various characters with underscores
    column_name = re.sub('[\.\s\-\/]', '_', column_name)

    # Remove unwanted characters
    column_name = re.sub('[\(\)\?\!\';:,=\+\*&^$#@]', '', column_name)

    # Replace two or more consecutive underscores with a single underscore
    column_name = re.sub('_+', '_', column_name)

    # Remove leading and trailing underscores
    column_name = column_name.strip('_')

    return column_name


def convert_date_columns(df):
    df = df.copy()
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                print(f"Warning: Unable to convert column '{col}' to datetime. Reason: {e}")
                continue

    return df

def validate_date(date, as_str=True):
    if isinstance(date, (pd.Timestamp, datetime.date)):
        date = date.strftime('%Y%m%d')

    elif isinstance(date, int):
        date = (pd.Timestamp.now() - pd.offsets.Day(date)).strftime('%Y%m%d')

    elif isinstance(date, str):
        date = pd.to_datetime(date).strftime('%Y%m%d')

    else:
        raise ValueError(f'unable to validate date: {date}')

    if as_str:
        return date

    return pd.Timestamp(date)


def utctoday():
    return validate_date(pd.Timestamp.utcnow(), as_str=False)


def today():
    return validate_date(pd.Timestamp.now(), as_str=False)


def zscore(df, wdw, closed_interval=False, min_periods_thresh=.5):
    if wdw < 2:
        return df
    
    # mp = min_periods_thresh * 
    if closed_interval:
        return (df - df.rolling(wdw, min_periods=2).mean()) / df.rolling(wdw, min_periods=2).std()
    else:
        return (df - df.rolling(wdw, min_periods=2).mean().shift(1)) / df.rolling(wdw, min_periods=2).std().shift(1)


def loc_around(df: pd.DataFrame, loc: int, wdw: int=5, highlight: bool=True):
    """
    return a sub dataframe around a location in a dataframe

    Args:
        df (pd.DataFrame): dataframe to be sliced
        loc (int): location to be sliced around
        wdw (int, optional): window size. Defaults to 5.
        highlight (bool, optional): highlight the location. Defaults to True.
    
    Returns:
        pd.DataFrame: sub dataframe
    """
    loc_idx = df.index.get_loc(loc)
    subdf = df.iloc[loc_idx - wdw: loc_idx + wdw + 1].copy()
    
    if highlight:
        def row_style(row):
            if row.name == loc:
                    return pd.Series('background-color: mint', row.index)
            else:
                return pd.Series('', row.index)
        
        subdf = subdf.style.apply(row_style, axis=1)
    
    return subdf 


def try_load_to_in_memory_sqlite(df: pd.DataFrame):
    df.to_sql('holdings_raw', con='sqlite://', if_exists='replace', index=False)


def binary_saerch_df_for_malformed_sql_col_or_row(df: pd.DataFrame, axis: int=1) -> str:
    """
    do binary search on columns or rows in df to see which column is causing load to sql to fail.
    use the try_load_to_in_memory_sqlite function to test the load. and return the column name
    or row index that is causing the load to fail.

    Args:
        df (pd.DataFrame): dataframe to be searched
        axis (int): axis to search. Defaults to 1 for columns.

    Returns:
        str: column name that is causing the load to fail
    """
    if axis == 1 and len(df.columns) == 1:
        print('Found the column that is causing the load to fail')
        print(df.columns[0])
        return df.columns[0]
    elif axis == 0 and len(df) == 1:
        print('Found the row that is causing the load to fail')
        print(df.index[0])
        return df.index[0]

    else:
        if axis == 1:
            mid = len(df.columns) // 2
            first_half = df.iloc[:, :mid]
            second_half = df.iloc[:, mid:]
        elif axis == 0:
            mid = len(df) // 2
            first_half = df.iloc[:mid, :]
            second_half = df.iloc[mid:, :]

        first_half_error = False
        second_half_error = False

        try:
            try_load_to_in_memory_sqlite(first_half)
            second_half_error = True
        except Exception as e:
            try_load_to_in_memory_sqlite(second_half)
            first_half_error = True
    
        if axis == 1:
            print(dict(num_cols=len(df.columns), first_half_error=first_half_error, second_half_error=second_half_error))
        elif axis == 0:
            print(dict(num_rows=len(df), first_half_error=first_half_error, second_half_error=second_half_error))

        if first_half_error:
            return binary_saerch_df_for_malformed_sql_col_or_row(first_half, axis=axis)
        elif second_half_error:
            return binary_saerch_df_for_malformed_sql_col_or_row(second_half, axis=axis)
        

def build_fuzzy_matches(provider1_names: list[str], provider2_names: list[str], case_sensitive: bool=False) -> dict[str, str]:
    """
    Build fuzzy matches between two lists of names. returns a dictionary of matches
    where keys are names from provider1 and values are corresponding names from provider2
    that are best matches.

    Args:
        provider1_names (list[str]): list of names from provider1
        provider2_names (list[str]): list of names from provider2
        case_sensitive (bool, optional): whether to do case sensitive matching. Defaults to False.

    Returns:
        dict: dictionary of matches. keys are names from provider1 and values are names from provider2
    """
    matches = {}

    for name_1 in tqdm(provider1_names):
        best_match, match_score = process.extractOne(name_1, provider2_names)
        if match_score > 80:  # You can set a threshold for match quality, e.g., 80% similarity
            matches[name_1] = best_match

    return matches


from cap.utils import make_column_name_sql_pd_safe, make_str_alphanumeric

# function to replace . with _ and convert camel case to snake case
def clean_column_names_old(df):
    df.columns = df.columns.str.replace('.', '_')
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('-', '_')
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    df.columns = df.columns.str.replace('/', '_')
    df.columns = df.columns.str.replace('%', 'pct')
    df.columns = df.columns.str.replace('?', '')
    df.columns = df.columns.str.replace('!', '')
    df.columns = df.columns.str.replace('\'', '')
    df.columns = df.columns.str.replace('\"', '')
    df.columns = df.columns.str.replace(';', '')
    df.columns = df.columns.str.replace(':', '')
    df.columns = df.columns.str.replace(',', '')
    df.columns = df.columns.str.replace('=', '')
    df.columns = df.columns.str.replace('+', '')
    df.columns = df.columns.str.replace('*', '')
    df.columns = df.columns.str.replace('&', '')
    df.columns = df.columns.str.replace('^', '')
    df.columns = df.columns.str.replace('$', '')
    df.columns = df.columns.str.replace('#', '')
    df.columns = df.columns.str.replace('@', '')

    return df


def build_fuzzy_matches(provider1_names: list[str], provider2_names: list[str], case_sensitive: bool=False) -> dict[str, str]:
    """
    Build fuzzy matches between two lists of names. returns a dictionary of matches
    where keys are names from provider1 and values are corresponding names from provider2
    that are best matches.

    Args:
        provider1_names (list[str]): list of names from provider1
        provider2_names (list[str]): list of names from provider2
        case_sensitive (bool, optional): whether to do case sensitive matching. Defaults to False.

    Returns:
        dict: dictionary of matches. keys are names from provider1 and values are names from provider2
    """
    if not case_sensitive:
        provider1_mapping = {name.lower(): name for name in provider1_names}
        provider2_mapping = {name.lower(): name for name in provider2_names}
        provider1_names_lower = list(provider1_mapping.keys())
        provider2_names_lower = list(provider2_mapping.keys())
    else:
        provider1_mapping = {name: name for name in provider1_names}
        provider2_mapping = {name: name for name in provider2_names}
        provider1_names_lower = provider1_names
        provider2_names_lower = provider2_names

    matches = {}
    for name1_lower in provider1_names_lower:
        best_match = process.extractOne(name1_lower, provider2_names_lower)
        if best_match:
            matches[provider1_mapping[name1_lower]] = provider2_mapping[best_match[0]]

    return matches

def clean_name(name: str, case_sensitive: bool, ignore_substrings: list[str]) -> str:
    if not case_sensitive:
        name = name.lower()
    for substring in ignore_substrings:
        name = name.replace(substring, '')
    return name.strip()


def build_fuzzy_matches(provider1_names: list[str], provider2_names: list[str], case_sensitive: bool = False,
                        ignore_substrings: list[str] = ['LP', 'L.P.', 'LLC', 'L.L.C.']) -> dict[str, str]:
    """
    Build fuzzy matches between two lists of names. returns a dictionary of matches
    where keys are names from provider1 and values are corresponding names from provider2
    that are best matches.

    make a series prodiver1_names_raw_to_clean_mapping that maps the raw names to the cleaned names
    incorperating the ignore_substrings and case_sensitive arguments. Then make the clean to raw mapping
    then do the same for provice2_names. Once that is done make 2 new lists: provider1_names_clean and provider2_names_clean
    do the fuzzy matching on the cleaned names. Finally right before returning the matches, map the cleaned names back to the raw names
    in both the keys and values of the matches dictionary. use pd.Series instead of dictionaries for the mappings

    Args:
        provider1_names (list[str]): list of names from provider1
        provider2_names (list[str]): list of names from provider2
        case_sensitive (bool, optional): whether to do case sensitive matching. Defaults to False.
        ignore_substrings (list[str], optional): list of substrings to ignore when matching. Defaults to ['LP', 'L.P.', 'LLC', 'L.L.C.'].

    Returns:
        dict: dictionary of matches. keys are names from provider1 and values are names from provider2
    """
    provider1_names_clean = [clean_name(name, case_sensitive, ignore_substrings) for name in provider1_names]
    provider2_names_clean = [clean_name(name, case_sensitive, ignore_substrings) for name in provider2_names]

    provider1_names_raw_to_clean_mapping = pd.Series(provider1_names_clean, index=provider1_names)
    provider1_names_clean_to_raw_mapping = pd.Series(provider1_names, index=provider1_names_clean)

    provider2_names_raw_to_clean_mapping = pd.Series(provider2_names_clean, index=provider2_names)
    provider2_names_clean_to_raw_mapping = pd.Series(provider2_names, index=provider2_names_clean)

    matches = {}
    for name_clean in provider1_names_clean:
        best_match, best_score = process.extractOne(name_clean, provider2_names_clean, scorer=fuzz.token_sort_ratio)
        # best_match, best_score = process.extractOne(name_clean, provider2_names_clean, scorer=fuzz.token_set_ratio)
        if best_score >= 80:  # You can adjust the threshold as needed
            raw_name = provider1_names_clean_to_raw_mapping[name_clean]
            raw_best_match = provider2_names_clean_to_raw_mapping[best_match]
            matches[raw_name] = raw_best_match

    return matches

import pandas as pd
from typing import List
from fuzzywuzzy import process

def clean_names_v2(names: list[str], case_sensitive: bool, ignore_substrings: list[str]) -> list[str]:
    if not case_sensitive:
        names = [name.upper() for name in names]

    for substr in ignore_substrings:
        names = [name.replace(substr, "") for name in names]

    return names

def build_fuzzy_matches_v2(provider1_names: List[str], provider2_names: List[str], case_sensitive: bool = False,
                        ignore_substrings: List[str] = ['LP', 'L.P.', 'LLC', 'L.L.C.']) -> dict[str, str]:

    provider1_names_clean = clean_names_v2(provider1_names, case_sensitive, ignore_substrings)
    provider2_names_clean = clean_names_v2(provider2_names, case_sensitive, ignore_substrings)

    provider1_raw_to_clean = pd.Series(provider1_names_clean, index=provider1_names)
    provider2_raw_to_clean = pd.Series(provider2_names_clean, index=provider2_names)

    matches = {}

    for name1 in provider1_names_clean:
        best_match, _ = process.extractOne(name1, provider2_names_clean)
        matches[name1] = best_match

    # Map cleaned names back to raw names
    matches_raw = {}
    for raw_name1, cleaned_name1 in provider1_raw_to_clean.items():
        cleaned_name2 = matches[cleaned_name1]
        raw_name2 = provider2_raw_to_clean[provider2_raw_to_clean == cleaned_name2].index[0]
        matches_raw[raw_name1] = raw_name2

    return matches_raw

    # import pandas as pd

# def infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Takes a DataFrame and infers data types for its columns by attempting to convert each column to datetime, int,
#     float, boolean, or object/str.
#
#     Args:
#         df (pd.DataFrame): The input DataFrame
#
#     Returns:
#         pd.DataFrame: A DataFrame with inferred data types for its columns
#     """
#
#     # Create a new DataFrame with the same index and columns as the input DataFrame
#     inferred_df = pd.DataFrame(index=df.index, columns=df.columns)
#
#     # Iterate through each column in the input DataFrame
#     for col in df.columns:
#         # Try to convert the column to datetime
#         try:
#             inferred_df[col] = pd.to_datetime(df[col])
#             continue
#         except ValueError:
#             pass
#
#         # Try to convert the column to integer
#         try:
#             inferred_df[col] = pd.to_numeric(df[col], errors='raise', downcast='integer')
#             continue
#         except ValueError:
#             pass
#
#         # Try to convert the column to float
#         try:
#             inferred_df[col] = pd.to_numeric(df[col], errors='raise', downcast='float')
#             continue
#         except ValueError:
#             pass
#
#         # Try to convert the column to boolean
#         try:
#             inferred_df[col] = df[col].astype(bool)
#             continue
#         except ValueError:
#             pass
#
#         # If all the above conversions failed, keep the column as object/string
#         inferred_df[col] = df[col].astype(object)
#
#     return inferred_df


import pandas as pd
import pyarrow as pa

import pandas as pd
import pyarrow as pa

import pandas as pd
import pyarrow as pa

def smart_convert_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to convert columns to the best available type, including datetime, boolean, float, int, and string. If a column
    is all nulls, convert it to string. Ensures compatibility with both numpy and pyarrow backends.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with columns converted to appropriate data types
    """

    # Create a new DataFrame with the same index and columns as the input DataFrame
    converted_df = df.copy()

    # Iterate through each column in the input DataFrame
    for col in df.columns:
        # If the column is all nulls, convert it to string
        if df[col].isnull().all():
            if '_date_' in col or '_date' in col or 'date_' in col:
                converted_df[col] = pd.to_datetime(df[col], errors='raise')
                continue
            else:
                converted_df[col] = df[col].astype(pd.StringDtype())
                continue

        # Check if the column is numeric
        # is_numeric = pd.to_numeric(df[col], errors='coerce').notnull().any()
        try:
            as_float = pd.to_numeric(df[col].astype(pd.StringDtype()))
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        # If the column is not numeric, try to convert the column to datetime
        if not is_numeric:
            try:
                converted_df[col] = pd.to_datetime(df[col], errors='raise')
                continue
            except (ValueError, TypeError):
                pass

        # Try to convert the column to boolean
        try:
            converted_df[col] = df[col].astype(pd.BooleanDtype())
            continue
        except (ValueError, TypeError):
            pass

        # Try to convert the column to integer
        try:
            converted_df[col] = pd.to_numeric(df[col], errors='raise', downcast='integer').astype(pd.Int64Dtype())
            continue
        except (ValueError, TypeError):
            pass

        # Try to convert the column to float
        try:
            converted_df[col] = pd.to_numeric(df[col], errors='raise', downcast='float').astype(pd.Float64Dtype())
            continue
        except (ValueError, TypeError):
            pass

        # If all the above conversions failed, keep the column as object/string
        converted_df[col] = df[col].astype(pd.StringDtype())

    return converted_df



def generate_sqlalchemy_table_generation_str_old(df: pd.DataFrame, table_name: str, metadata_obj_var_name: str) -> str:
    """
    Takes in a pandas DataFrame and generates a string of the corresponding SQLAlchemy table with correct datatypes for each column.

    Args:
        df (pd.DataFrame): DataFrame to be converted to SQLAlchemy table.
        table_name (str): Name of the table.
        metadata_obj_var_name (str): Name of the metadata object variable.

    Returns:
        str: String of SQLAlchemy table generation statement.
    """

    def find_nearest_anchor(length: int, anchors: list[int] = [5, 10, 20, 50, 100, 200, 500, 1000]) -> int:
        return min(anchors, key=lambda x: abs(x - length))

    def get_sqlalchemy_type(column: pd.Series) -> str:
        dtype = column.dtype
        if np.issubdtype(dtype, np.datetime64):
            return "DateTime"
        elif np.issubdtype(dtype, np.number) or pd.api.types.is_numeric_dtype(dtype):
            if np.issubdtype(dtype, np.integer) or pd.api.types.is_integer_dtype(dtype):
                return "Integer"
            else:
                return "Float"
        elif np.issubdtype(dtype, np.bool_) or pd.api.types.is_bool_dtype(dtype):
            return "Boolean"
        elif column.isna().all():
            default_length = 50
            return f"String({default_length})"
        else:
            max_length = column.astype(str).str.len().max()
            anchor_length = find_nearest_anchor(max_length * 2)
            return f"String({anchor_length})"

    columns = []
    for column_name, column in df.iteritems():
        sqlalchemy_type = get_sqlalchemy_type(column)
        columns.append(f"Column('{column_name}', {sqlalchemy_type})")

    columns_str = ",\n    ".join(columns)
    result = f"{table_name} = Table(\n    '{table_name}', {metadata_obj_var_name},\n    {columns_str}\n)"
    return result

import numpy as np
import pandas as pd
import pyarrow as pa

def generate_sqlalchemy_table_generation_str(df: pd.DataFrame, table_name: str, metadata_obj_var_name: str) -> str:
    """
    Takes in a pandas DataFrame and generates a string of the corresponding SQLAlchemy table with correct datatypes for each column.

    Args:
        df (pd.DataFrame): DataFrame to be converted to SQLAlchemy table.
        table_name (str): Name of the table.
        metadata_obj_var_name (str): Name of the metadata object variable.

    Returns:
        str: String of SQLAlchemy table generation statement.
    """

    def find_nearest_anchor(length: int, anchors: list[int] = [5, 10, 20, 50, 100, 200, 500, 1000]) -> int:
        return min(anchors, key=lambda x: abs(x - length))

    def get_sqlalchemy_type(column: pd.Series) -> str:
        dtype = column.dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "DateTime"
        elif pd.api.types.is_integer_dtype(dtype):
            return "Integer"
        elif pd.api.types.is_float_dtype(dtype):
            return "Float"
        elif pd.api.types.is_bool_dtype(dtype):
            return "Boolean"
        elif pd.api.types.is_string_dtype(dtype) or column.isna().all():
            if column.isna().all():
                default_length = 50
            else:
                max_length = column.astype(str).str.len().max()
                default_length = find_nearest_anchor(max_length * 2)
            return f"String({default_length})"
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    columns = []
    for column_name, column in df.iteritems():
        sqlalchemy_type = get_sqlalchemy_type(column)
        columns.append(f"Column('{column_name}', {sqlalchemy_type})")

    columns_str = ",\n    ".join(columns)
    result = f"{table_name} = Table(\n    '{table_name}', {metadata_obj_var_name},\n    {columns_str}\n)"
    return result



def generate_sqlalchemy_tables_file_generation_str(dfs: dict[str, pd.DataFrame], metadata_obj_var_name: str = "metadata") -> str:
    """
    Generates a string containing SQLAlchemy table generation statements with schemas for all DataFrames in the input dictionary.

    Args:
        dfs (Dict[str, pd.DataFrame]): A dictionary with keys as table names and values as DataFrames.
        metadata_obj_var_name (str, optional): The variable name of the MetaData object. Defaults to "metadata".

    Returns:
        str: A string containing SQLAlchemy table generation statements for each DataFrame with their schemas.
    """
    from tqdm.auto import tqdm
    result = []
    for table_name, df in tqdm(dfs.items(), total=len(dfs)):
        table_str = generate_sqlalchemy_table_generation_str(df, table_name, metadata_obj_var_name)
        result.append(table_str)

    all_tables_code = "\n\n".join(result)

    header_code = f"""
from sqlalchemy import Column, DateTime, MetaData, Table, Text, String, Float, Integer, \\
    BigInteger, ForeignKey, ForeignKeyConstraint, PrimaryKeyConstraint, Boolean
import pandas as pd
from io import StringIO


{metadata_obj_var_name} = MetaData()


"""
    final_code = header_code + all_tables_code

    return final_code


import pysftp

def get_sftp_cnopts() -> pysftp.CnOpts:
    """
    reads the known_hosts file and returns a pysftp.CnOpts object with the host keys added

    Returns:
        pysftp.CnOpts: CnOpts object with host keys added
    """
    known_hosts_file = 'known_hosts'
    known_hosts_fpath = Path(cap.cap_repo_path) / known_hosts_file
    known_hosts_fpath = known_hosts_fpath.resolve().absolute()

    cnopts = pysftp.CnOpts()
    cnopts.hostkeys.load(known_hosts_fpath)

    return cnopts

    # def remove_substrings(name: str, substrings: list[str]) -> str:
    #     for substring in substrings:
    #         name = name.replace(substring, '')
    #     return name.strip()
    #
    # if not case_sensitive:
    #     provider1_mapping = {name.lower(): name for name in provider1_names}
    #     provider2_mapping = {name.lower(): name for name in provider2_names}
    #     provider1_names_lower = list(provider1_mapping.keys())
    #     provider2_names_lower = list(provider2_mapping.keys())
    # else:
    #     provider1_mapping = {name: name for name in provider1_names}
    #     provider2_mapping = {name: name for name in provider2_names}
    #     provider1_names_lower = provider1_names
    #     provider2_names_lower = provider2_names
    #
    # # Remove the specified substrings before performing fuzzy matching
    # provider1_names_clean = [remove_substrings(name, ignore_substrings) for name in provider1_names_lower]
    # provider2_names_clean = [remove_substrings(name, ignore_substrings) for name in provider2_names_lower]
    #
    # matches = {}
    # for name1_clean, name1_lower in zip(provider1_names_clean, provider1_names_lower):
    #     best_match = process.extractOne(name1_clean, provider2_names_clean)
    #     if best_match:
    #         matches[provider1_mapping[name1_lower]] = provider2_mapping[best_match[0]]
    # return matches