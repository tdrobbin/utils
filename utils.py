import pandas as pd
import numpy as np
import tarfile
import logging
import re
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def get_cache_dir():
    pth = Path('~/Documents/temp/dtkcache')
    if not pth.exists():
        pth.mkdir(parents=True)

    return pth


def df_info(df):
    display(df.info(verbose=True))
    display(df.describe(include='all'))
    display(df.sample(10))


def display_dfs_inline(dfs, captions=None, margin=5):
    from functools import reduce
    from IPython.display import display_html
    
    captions = [''] * len(dfs) if captions is None else captions
    stylers = [D(df).style.set_table_attributes(f'style="display:inline; margin:{margin}px;"').set_caption(c) for df, c in zip(dfs, captions)]
    display_html(reduce(lambda x, y: x + y, (s._repr_html_() for s in stylers)), raw=True)


def smart_convert_df(df, infer_objects=True):
    if infer_objects:
        df = df.infer_objects().copy()
    else:
        df = df.copy()
    
    for c in df.columns:
        try:
            if df[c].dtype == object:
                df[c] = df[c].fillna(np.nan).astype(float)
        except (ValueError, TypeError):
            pass
    
    return df


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



def get_volatility_data(klines_df, standardize_periods=1440):
    wdws = [2**i for i in range(1, 10)]
    prc = klines_df[['close']]
    ret = prc.close.pct_change()
    
    up_ret = ret.copy()
    up_ret.loc[ret < 0] = np.nan
    
    down_ret = ret.copy()
    down_ret.loc[ret > 0] = np.nan
    
    vol_df = prc.copy()
    for wdw in tqdm(wdws):
        tot_vol = ret.rolling(wdw).std() * np.sqrt(standardize_periods / wdw)
        vol_df[f'tot_vol_{wdw}'] = tot_vol
        
        up_vol = up_ret.rolling(wdw, min_periods=2).std() * np.sqrt(standardize_periods / wdw)
        vol_df[f'up_vol_{wdw}'] = up_vol
        
        down_vol = down_ret.rolling(wdw, min_periods=2).std() * np.sqrt(standardize_periods / wdw)
        vol_df[f'down_vol_{wdw}'] = down_vol
        
        vol_df[f'ud_vol_ratio_{wdw}'] = up_vol / down_vol
    
    return vol_df


def loc_around(df, loc, wdw=5, highlight=True):
    loc_idx = df.index.get_loc(loc)
    subdf = df.iloc[loc_idx - wdw: loc_idx + wdw + 1].copy()
    
    if highlight:
        def row_style(row):
            if row.name == loc:
                    return pd.Series('background-color: mint', row.index)
            else:
                return pd.Series('', row.index)
        
        # def index_style(idx):
        #     if idx == loc:
        #         return pd.Series('background-color: yellow', row.index)
        #     else:
        #         return pd.Series('', row.index)
        
        subdf = subdf.style.apply(row_style, axis=1)
        # .apply_index(index_style, axis=1)
        # subdf = subdf.style.apply_index(index_style, axis=1)
    
    return subdf 