__all__ = ['logger', 'process_multiple_ts', 'Info', 'TimeSeriesDataclass', 'get_holiday_dates', 'holiday_kernel',
           'create_calendar_variables', 'create_us_holiday_distance_variables', 'US_FEDERAL_HOLIDAYS', 'TimeFeature',
           'SecondOfMinute', 'MinuteOfHour', 'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear', 'MonthOfYear',
           'WeekOfYear', 'time_features_from_frequency_str']

# Cell
import logging
import requests
import subprocess
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_multiple_ts(y_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Transforms multiple timeseries as columns to long format."""
    y_df['date'] = pd.to_datetime(y_df['date'])
    y_df.rename(columns={'date': 'ds'}, inplace=True)
    u_ids = y_df.columns.to_list()
    u_ids.remove('ds')

    time_cls = time_features_from_frequency_str('h')
    for cls_ in time_cls:
        cls_name = cls_.__class__.__name__
        y_df[cls_name] = cls_(y_df['ds'].dt)

    X_df = y_df.drop(u_ids, axis=1)
    y_df = y_df.filter(items=['ds'] + u_ids)
    y_df = y_df.set_index('ds').stack()
    y_df = y_df.rename('y').rename_axis(['ds', 'unique_id']).reset_index()
    y_df['unique_id'] = pd.Categorical(y_df['unique_id'], u_ids)
    y_df = y_df[['unique_id', 'ds', 'y']].sort_values(['unique_id', 'ds'])

    X_df = y_df[['unique_id', 'ds']].merge(X_df, how='left', on=['ds'])

    return y_df, X_df

# Cell
@dataclass
class Info:
    """
    Info Dataclass of datasets.
    Args:
        groups (Tuple): Tuple of str groups
        class_groups (Tuple): Tuple of dataclasses.
    """
    groups: Tuple[str]
    class_groups: Tuple[dataclass]

    def get_group(self, group: str):
        """Gets dataclass of group."""
        if group not in self.groups:
            raise Exception(f'Unkown group {group}')

        return self.class_groups[self.groups.index(group)]

    def __getitem__(self, group: str):
        """Gets dataclass of group."""
        if group not in self.groups:
            raise Exception(f'Unkown group {group}')

        return self.class_groups[self.groups.index(group)]

    def __iter__(self):
        for group in self.groups:
            yield group, self.get_group(group)


# Cell
@dataclass
class TimeSeriesDataclass:
    """
    Args:
        S (pd.DataFrame): DataFrame of static features of shape
            (n_time_series, n_features).
        X (pd.DataFrame): DataFrame of exogenous variables of shape
            (sum n_periods_i for i=1..n_time_series, n_exogenous).
        Y (pd.DataFrame): DataFrame of target variable of shape
            (sum n_periods_i for i=1..n_time_series, 1).
        idx_categorical_static (list, optional): List of categorical indexes
            of S.
        group (str, optional): Group name if applies.
            Example: 'Yearly'
    """
    S: pd.DataFrame
    X: pd.DataFrame
    Y: pd.DataFrame
    idx_categorical_static: Optional[List] = None
    group: Union[str, List[str]] = None

# Cell
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    USMartinLutherKingJr,
    USPresidentsDay,
    USMemorialDay,
    USLaborDay,
    USColumbusDay,
    USThanksgivingDay,
    nearest_workday
)

US_FEDERAL_HOLIDAYS = {'new_year': Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
                       'martin_luther_king': USMartinLutherKingJr,
                       'presidents': USPresidentsDay,
                       'memorial': USMemorialDay,
                       'independence': Holiday("July 4th", month=7, day=4, observance=nearest_workday),
                       'labor': USLaborDay,
                       'columbus': USColumbusDay,
                       'veterans': Holiday("Veterans Day", month=11, day=11, observance=nearest_workday),
                       'thanksgiving': USThanksgivingDay,
                       'christmas': Holiday("Christmas", month=12, day=25, observance=nearest_workday)}

def get_holiday_dates(holiday, dates):
    start_date = min(dates) + pd.DateOffset(days=-366)
    end_date = max(dates) + pd.DateOffset(days=366)
    holiday_calendar = AbstractHolidayCalendar(rules=[US_FEDERAL_HOLIDAYS[holiday]])
    holiday_dates = holiday_calendar.holidays(start=start_date, end=end_date)
    return np.array(holiday_dates)

def holiday_kernel(holiday, dates):
    # Get holidays around dates
    dates = pd.DatetimeIndex(dates)
    dates_np = np.array(dates).astype('datetime64[D]')
    holiday_dates = get_holiday_dates(holiday, dates)
    holiday_dates_np = np.array(pd.DatetimeIndex(holiday_dates)).astype('datetime64[D]')

    # Compute day distance to holiday
    nearest_holiday_idx = np.expand_dims(dates_np, axis=1) - np.expand_dims(holiday_dates_np, axis=0)
    nearest_holiday_idx = np.argmin(np.abs(nearest_holiday_idx), axis=1)
    nearest_holiday = pd.DatetimeIndex([holiday_dates[idx] for idx in nearest_holiday_idx])
    holiday_diff = (dates - nearest_holiday).days.values
    return holiday_diff

def create_calendar_variables(X_df: pd.DataFrame):
    X_df['day_of_year'] = X_df.ds.dt.dayofyear
    X_df['day_of_week'] = X_df.ds.dt.dayofweek
    X_df['hour'] = X_df.ds.dt.hour
    return X_df

def create_us_holiday_distance_variables(X_df: pd.DataFrame):
    dates = X_df.ds.dt.date
    for holiday in US_FEDERAL_HOLIDAYS.keys():
        X_df[f'holiday_dist_{holiday}'] = holiday_kernel(holiday=holiday,
                                                         dates=dates)
    return X_df

# Cell
## This code was taken from:
# https://github.com/zhouhaoyi/Informer2020/blob/429f8ace8dde71655d8f8a5aad1a36303a2b2dfe/utils/timefeatures.py#L114
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]