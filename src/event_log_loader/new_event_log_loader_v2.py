import pandas as pd
import numpy as np
from functools import partial
import sklearn
import sklearn.preprocessing
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from typing import Optional, Dict, Any, Iterable
from sklearn.base import BaseEstimator, TransformerMixin

# raw dataframe of whole event log: creates a dataframe with dynamic attributes as lists and static as value
class RawDataFrameLoader:
    def __init__(self,
                 event_log_dir : str,
                 timestamp_name : str,
                 case_name : str,
                 categorical_columns : list[str],
                 continuous_columns : list[str],
                 continuous_positive_columns : list[str],
                 static_categorical_columns : Optional[list[str]] = None,
                 static_continuous_columns : Optional[list[str]] = None,
                 time_since_case_start_column : str | None = None,
                 time_since_last_event_column : str | None = None,
                 day_in_week_column : str | None = None,
                 seconds_in_day_column : str | None = None,
                 date_format : str = '%Y-%m-%d %H:%M:%S.%f',
                 min_suffix_size : int = 1,
                 **kwargs):
        
        self.df = pd.read_csv(event_log_dir)
        
        self.case_name = case_name
        
        self.timestamp_name = timestamp_name
        
        self.time_since_case_start_column = time_since_case_start_column
        self.time_since_last_event_column = time_since_last_event_column
        self.day_in_week_column = day_in_week_column
        self.seconds_in_day_column = seconds_in_day_column
        self.date_format = date_format
        self.min_suffix_size = min_suffix_size
        
        # dynamic attributes
        self.categorical_columns = list(categorical_columns or [])
        self.continuous_columns = list(continuous_columns or [])
        # dynamic (log-normal) continuous attributes
        self.continuous_positive_columns = list(continuous_positive_columns or [])
        # static attributes
        self.static_categorical_columns = list(static_categorical_columns or [])
        self.static_continuous_columns = list(static_continuous_columns or [])

        self.df[self.timestamp_name] = pd.to_datetime(self.df[self.timestamp_name], format=date_format, errors='coerce')
        
    @staticmethod
    def __extract_static_value(series: pd.Series) -> object:
        cleaned = series.dropna()
        if cleaned.empty:
            return np.nan
        if cleaned.dtype == object or cleaned.dtype.name == 'category':
            cleaned = cleaned[cleaned != 'EOS']
            if cleaned.empty:
                return np.nan
        return cleaned.iloc[0]

    def create_case_level_dataframe(self, event_level_df : pd.DataFrame) -> pd.DataFrame:
        grouped = event_level_df.groupby(self.case_name, sort=False)
        records = []
        for case_id, group in grouped:
            row = {self.case_name: case_id}
            for col in self.categorical_columns + self.continuous_columns + self.continuous_positive_columns:
                if col in group.columns:
                    row[col] = group[col].tolist()
            for col in self.static_categorical_columns:
                if col in group.columns:
                    row[col] = self.__extract_static_value(group[col])
            for col in self.static_continuous_columns:
                if col in group.columns:
                    row[col] = self.__extract_static_value(group[col])
            records.append(row)
        return pd.DataFrame(records)

# base object for the dataset creation
class CSV2EventLog(RawDataFrameLoader):                               
    def __init__(self, *args, **kwargs):
        # load raw constructor
        super().__init__(*args, **kwargs)
        
        # Time values
        # create new time since case started column if desired
        if self.time_since_case_start_column:
            self.__create_time_since_case_start_column()

        # create new offset time to last event column if desired
        if self.time_since_last_event_column:
            self.__create_time_since_last_event_column()

        # create new day in week column if desired
        if self.day_in_week_column:
            self.__create_day_in_week_column()

        # create new seconds in day column if desired
        if self.seconds_in_day_column:
            self.__create_seconds_in_day_column()
            
        # raw dataframe befor split containing categorical and continuous attributes
        self.raw_df = self.create_case_level_dataframe(self.df.copy())
        
        # Add EOS to every case
        self.df = self.df.groupby(self.case_name, group_keys=False).apply(
            lambda group : self.__add_last_rows(group)).reset_index(drop=True)

        for categorical_col in self.categorical_columns:
            self.df[categorical_col] = self.df[categorical_col].apply(lambda x: x if pd.isna(x) else str(x))
            self.df[categorical_col] = self.df[categorical_col].astype(object)

        for continuous_col in self.continuous_columns:
            self.df[continuous_col] = self.df[continuous_col].astype('float32')
        for continuous_col in self.continuous_positive_columns:
            self.df[continuous_col] = self.df[continuous_col].astype('float32')

    def __create_time_since_case_start_column(self):
        case_start_times = self.df.groupby(self.case_name)[self.timestamp_name].transform('min')
        time_offset = self.df[self.timestamp_name] - case_start_times
        time_offset_seconds = time_offset.dt.total_seconds()
        self.df[self.time_since_case_start_column] = time_offset_seconds
        self.max_case_length = self.df.groupby(self.case_name).size().max()

    @staticmethod
    def __min_timestamp_before_event(group, timestamp_name, new_column_name):
        min_values = []
        for i, row in group.iterrows():
            before_values = group[(group[timestamp_name] < row[timestamp_name])][timestamp_name]
            if not before_values.empty:
                min_values.append(before_values.max())
            else:
                min_values.append(np.nan)
        group[new_column_name] = min_values
        return group
                                   
    def __create_time_since_last_event_column(self):
        min_timestamp_before = partial(CSV2EventLog.__min_timestamp_before_event,
                                       timestamp_name = self.timestamp_name,
                                       new_column_name = self.time_since_last_event_column)
        self.df = self.df.groupby(self.case_name).apply(min_timestamp_before).reset_index(drop=True)
        self.df[self.time_since_last_event_column] = (self.df[self.timestamp_name] - self.df[self.time_since_last_event_column]).dt.total_seconds()

    def __create_day_in_week_column(self):
        self.df[self.day_in_week_column] =  self.df[self.timestamp_name].dt.weekday

    def __create_seconds_in_day_column(self):
        self.df[self.seconds_in_day_column] = self.df[self.timestamp_name].dt.hour * 3600 + \
            self.df[self.timestamp_name].dt.minute * 60 + \
            self.df[self.timestamp_name].dt.second

    def __add_last_rows(self, group):
        new_row = {}
        for col in group.columns:
            if col == self.case_name:
                new_row[col] = group.name
            elif group[col].dtype == 'object' or group[col].dtype.name == 'category':
                new_row[col] = 'EOS'

        # Explain more in detail
        max_case_len = ((len(group) + self.min_suffix_size-1) -len(group))
        
        eos_rows = pd.DataFrame(max_case_len * [new_row])
        concat_case = pd.concat([group.sort_values(by=self.timestamp_name), eos_rows])
        return concat_case

#  split dataframes 
class EventLogSplitter:
    def __init__(self,
                 train_validation_size : float,
                 test_validation_size : float,
                 **kwargs):
        self.train_validation_size = train_validation_size
        self.test_validation_size = test_validation_size

    def split(self,
              event_log : CSV2EventLog):
        cases = event_log.df[event_log.case_name].unique()
        np.random.shuffle(cases)

        train_validation_ix = int(self.train_validation_size * len(cases))
        test_validation_ix = train_validation_ix + int(self.test_validation_size * len(cases))

        train_validation_cases = cases[:train_validation_ix]
        test_validation_cases = cases[train_validation_ix:test_validation_ix]
        train_cases = cases[test_validation_ix:]

        train_df = event_log.df[event_log.df[event_log.case_name].isin(train_cases)]
        train_validation_df = event_log.df[event_log.df[event_log.case_name].isin(train_validation_cases)]
        test_validation_df = event_log.df[event_log.df[event_log.case_name].isin(test_validation_cases)]

        return train_df, train_validation_df, test_validation_df

# class that provides intermediate steps: create dataframe of prefixes for marking determination:
class PrefixesDataFrameLoader:
    def __init__(self, event_log_location: str, event_log_properties: Dict[str, Any]):
        if not event_log_properties:
            raise ValueError("event_log_properties are required")

        self.case_name = event_log_properties['case_name']
        self.min_suffix_size = event_log_properties.get('min_suffix_size', 1)
        self.window_size_setting = event_log_properties.get('window_size', 'auto')

        self.categorical_columns = list(event_log_properties.get('categorical_columns') or [])
        self.continuous_columns = list(event_log_properties.get('continuous_columns') or [])
        self.continuous_positive_columns = list(event_log_properties.get('continuous_positive_columns') or [])
        
        self.static_categorical_columns = list(event_log_properties.get('static_categorical_columns') or [])
        self.static_continuous_columns = list(event_log_properties.get('static_continuous_columns') or [])

        # create processed event log with EOS rows and engineered columns
        self.csv2event_log = CSV2EventLog(event_log_location, **event_log_properties)
        self.event_log = self.csv2event_log.df.copy()

        # configure window size so we can trim sequences later on
        self.window_size = self._resolve_window_size()

        # splitting initialization
        train_size = event_log_properties.get('train_validation_size', 0.1)
        test_size = event_log_properties.get('test_validation_size', 0.1)
        splitter = EventLogSplitter(train_validation_size=train_size, test_validation_size=test_size)
        train_df, val_df, test_df = splitter.split(self.csv2event_log)

        
        self.processed_splits: Dict[str, pd.DataFrame] = {'train': train_df.reset_index(drop=True).copy(),
                                                          'val': val_df.reset_index(drop=True).copy(),
                                                          'test': test_df.reset_index(drop=True).copy()}
        self.train_df = self.processed_splits['train']
        self.val_df = self.processed_splits['val']
        self.test_df = self.processed_splits['test']

    def get_raw_dataframe(self) -> pd.DataFrame:
        # return self.csv2event_log.raw_df.copy()
        return self.csv2event_log.raw_df.copy()
        
    def _resolve_window_size(self) -> int:
        setting = self.window_size_setting if self.window_size_setting is not None else 'auto'
        if isinstance(setting, str) and setting.lower() == 'auto':
            case_sizes = self.event_log.groupby(self.case_name).size()
            if case_sizes.empty:
                return self.min_suffix_size
            auto_window = round(case_sizes.quantile(1 - 0.015)) + self.min_suffix_size
            return max(self.min_suffix_size, int(auto_window))
        return int(setting)

    @staticmethod
    def _extract_static_value(series: pd.Series) -> object:
        cleaned = series.dropna()
        if cleaned.empty:
            return np.nan
        return cleaned.iloc[0]

    def _limit_sequence(self, values):
        seq = list(values)
        if len(seq) <= self.window_size:
            return seq
        return seq[-self.window_size:]

    def transform(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        
        working_df = self.event_log if df is None else df
    
        rows = []
        grouped = working_df.groupby(self.case_name, sort=False)
        for case_id, group in grouped:
            # case
            group = group.reset_index(drop=True)
            total_len = len(group)
            
            # check if correct
            for prefix_len in range(self.min_suffix_size+1, total_len+1):
                row = {self.case_name: case_id,
                       'prefix_length': prefix_len - self.min_suffix_size}
                # cat
                for col in self.categorical_columns:
                    if col in group.columns:
                        values = group[col].iloc[:prefix_len - self.min_suffix_size].tolist()
                    else:
                        values = []
                    row[col] = self._limit_sequence(values)
                # con
                for col in self.continuous_columns + self.continuous_positive_columns:
                    if col in group.columns:
                        values = group[col].iloc[:prefix_len - self.min_suffix_size].tolist()
                    else:
                        values = []
                    row[col] = self._limit_sequence(values)
                # static
                for col in self.static_categorical_columns:
                    if col in group.columns:
                        row[col] = self._extract_static_value(group[col])
                    else:
                        row[col] = np.nan
                for col in self.static_continuous_columns:
                    if col in group.columns:
                        row[col] = self._extract_static_value(group[col])
                    else:
                        row[col] = np.nan
                rows.append(row)
        return pd.DataFrame(rows)

    def get_dataset(self, type : str):
        if type == 'train':
            df = self.transform(df=self.train_df)
        elif type == 'val':
            df = self.transform(df=self.val_df)
        elif type == 'test':
            df = self.transform(df=self.test_df)
        return df

# standardization for log-normal             
class PositiveStandardizer_normed(BaseEstimator, TransformerMixin):
    """
    Standard scaler for log normal attributes.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        print("Positive Standardization")
        log_x = np.log1p(X)
        print("min,25%,50%,75%,max:", np.percentile(log_x, [0,25,50,75,100]))
        # Standardize values
        self.mean_ = np.mean(log_x, axis=0)
        print("Mean: ", self.mean_)
        self.std_ = np.std(log_x, axis=0)
        print("Std: ", self.std_)
        return self
        
    def transform(self, X):
        # log the observations to assume normal PDF
        log_x = np.log1p(X)
        x_enc = (log_x - self.mean_) / self.std_ 
        return x_enc
    
    def inverse_transform(self, X_enc):
        # Destandardization
        log_x = X_enc * self.std_ + self.mean_
        # Exponentiation:
        x = np.expm1(log_x)
        return x

# responsible for tensor encoding of event log data
class TensorEncoderDecoder:
    def __init__(self,
                 event_log : pd.DataFrame,
                 case_name : str,
                 concept_name : str,
                 window_size : int,
                 min_suffix_size : int,
                 categorical_columns : Optional[list[str]] = None,
                 continuous_columns : Optional[list[str]] = None,
                 continuous_positive_columns : Optional[list[str]] = None,
                 static_categorical_columns : Optional[list[str]] = None,
                 static_continuous_columns : Optional[list[str]] = None,
                 **kwargs):

        self.event_log = event_log
        self.case_name = case_name
        self.concept_name = concept_name
        self.min_suffix_size = min_suffix_size
        if window_size == 'auto':
            # get max. length of (100-1.5)% of the longest cases as prefix
            # and add the min. suffix_size
            case_sizes = self.event_log.groupby(case_name).size()
            self.window_size = round(case_sizes.quantile(1 - 0.015)) + self.min_suffix_size
        else:
            self.window_size = window_size
        self.categorical_columns = list(categorical_columns or [])
        self.continuous_columns = list(continuous_columns or [])
        self.continuous_positive_columns = list(continuous_positive_columns or [])
        self.static_categorical_columns = list(static_categorical_columns or [])
        self.static_continuous_columns = list(static_continuous_columns or [])

        self.categorical_imputers : dict[str, SimpleImputer] = dict()
        self.categorical_encoders : dict[str, sklearn.preprocessing.OrdinalEncoder]  = dict()
        for categorical_column in self.categorical_columns + self.static_categorical_columns:
            if categorical_column not in self.categorical_encoders:
                self.categorical_encoders[categorical_column] = self.__get_categorical_encoder()

        self.continuous_imputers = dict()
        self.continuous_encoders : dict[str, sklearn.preprocessing.StandardScaler] = dict()
        
        # Normal encoding
        for continuous_column in self.continuous_columns + self.static_continuous_columns:
            if continuous_column not in self.continuous_imputers:
                self.continuous_imputers[continuous_column] = self.__get_continuous_imputer()
                self.continuous_encoders[continuous_column] = self.__get_continuous_encoder()
        
        for continuous_positive_column in self.continuous_positive_columns:
            self.continuous_imputers[continuous_positive_column] = self.__get_continuous_positive_imputer()
            self.continuous_encoders[continuous_positive_column] = self.__get_continuous_positive_encoder()
    
    def train_imputers_encoders(self):
        # categorical encoders: fit on 2D numpy arrays with dtype=object
        for col, categorical_encoder in self.categorical_encoders.items():
            column_data = self.event_log[[col]].astype(object).to_numpy()  # shape (n,1)
            categorical_encoder.fit(column_data)

        # continuous encoders / imputers: fit on 2D numpy arrays (n_samples, 1)
        for col, continuous_encoder in self.continuous_encoders.items():
            continuous_imputer = self.continuous_imputers[col]
            column_data = self.event_log[[col]].to_numpy()  # DataFrame -> ndarray (n,1)
            column_data = continuous_imputer.fit_transform(column_data)  # still (n,1)
            continuous_encoder.fit(column_data)  # StandardScaler or custom transformer expects 2D

    def _single_encode_categorical_column(self,
                                          df_case : pd.DataFrame,
                                          col : str) -> torch.Tensor:
        case_values = np.array(df_case[[col]], dtype=object)
        case_values_enc = self.categorical_encoders[col].transform(case_values) + 1  # shape (n,1)
        # Pad encodings - clearer prefix loop (prefix_len from min_suffix_size .. len)
        case_values_enc_pad = self.pad_to_window_size(case_values_enc)
        return torch.tensor(np.array(case_values_enc_pad, dtype=int), dtype=torch.long).squeeze(-1).unsqueeze(0)
    
    def _single_encode_continuous_column(self,
                                         df_case : pd.DataFrame,
                                         col : str) -> torch.Tensor:
        case_values = df_case[[col]].values  # shape (n,1)
        case_values_imputed = self.continuous_imputers[col].transform(case_values)
        case_values_enc = self.continuous_encoders[col].transform(case_values_imputed)
        case_values_enc_pad = self.pad_to_window_size(case_values_enc)
        return torch.tensor(np.array(case_values_enc_pad, dtype=float), dtype=torch.float).squeeze(-1).unsqueeze(0)

    def encode_case(self,
                    df_case : pd.DataFrame) -> tuple[tuple[torch.Tensor], tuple[torch.Tensor]]:
        categorical_tensors = []
        continuous_tensors = []
        for col in self.categorical_columns:
            cat_columns = self._single_encode_categorical_column(df_case, col)
            categorical_tensors.append(cat_columns)
        for col in self.continuous_columns + self.continuous_positive_columns:
            cont_columns = self._single_encode_continuous_column(df_case, col)
            continuous_tensors.append(cont_columns)
        return tuple(categorical_tensors), tuple(continuous_tensors)

    def encode_df(self, df) -> tuple[dict[str, object],
                                     tuple[list[tuple[str, int, dict[str, int]]]],
                                     tuple[list[tuple[str, int, dict[str, int]]]]]:
        categorical_tensors = []
        all_categories = [[], []]
        static_categories = [[], []]
        eos_padding_tensor = None
        zero_padding_tensor = None
        case_ids = None

        for col in tqdm(self.categorical_columns, desc='categorical tensors'):
            if col == self.concept_name:
                case_ids, enc_column, eos_padding_tensor, zero_padding_tensor, categories, max_classes = self.encode_categorical_column(df, col, return_case_ids_and_eos_paddings=True)
            else:
                enc_column, categories, max_classes = self.encode_categorical_column(df, col)
            categorical_tensors.append(enc_column)
            all_categories[0].append((col, max_classes, categories))

        if case_ids is None or eos_padding_tensor is None or zero_padding_tensor is None:
            raise ValueError("Concept column must be part of the categorical_columns to compute padding metadata.")

        continuous_tensors = []
        for col in tqdm(self.continuous_columns + self.continuous_positive_columns, desc='continouous tensors'):
            continuous_tensors.append(self.encode_continuous_column(df, col))
            all_categories[1].append((col, 1, dict()))

        for col in self.static_categorical_columns:
            filtered_categories = [category for category in self.categorical_encoders[col].categories_[0] if category != 'EOS']
            categories = {category: idx + 1 for idx, category in enumerate(filtered_categories)}
            max_classes = len(filtered_categories) + 1
            static_categories[0].append((col, max_classes, categories))
        for col in self.static_continuous_columns:
            static_categories[1].append((col, 1, dict()))

        static_cat_tensor, static_cont_tensor = self._encode_static_attributes(df, case_ids)

        tensor_bundle = {
            'categorical': categorical_tensors,
            'continuous': continuous_tensors,
            'eos_padding': eos_padding_tensor,
            'zero_padding': zero_padding_tensor,
            'case_ids': tuple(case_ids),
            'static_categorical': static_cat_tensor,
            'static_continuous': static_cont_tensor
        }

        return tensor_bundle, tuple(all_categories), tuple(static_categories)

    # Corrected verison:
    def encode_categorical_column(self, df, col, return_case_ids_and_eos_paddings=False):
        grouped = df.groupby(self.case_name)
        windows = []
        eos_masks = []
        zero_masks = []
        categories = {category: idx + 1 for idx, category in enumerate(self.categorical_encoders[col].categories_[0])}
        eos_token_id = categories.get('EOS', 0)
        
        case_ids = []
        for case_id, group in tqdm(grouped, desc=col, leave=False):
            case_values = np.array(group[[col]], dtype=object)
            case_values_enc = self.categorical_encoders[col].transform(case_values) + 1  # shape (n,1)
            padded_encodings = []
            for prefix_len in range(self.min_suffix_size + 1, len(case_values_enc) + 1):
                padded_slice = self.pad_to_window_size(case_values_enc[:prefix_len])
                padded_encodings.append(padded_slice)
                if return_case_ids_and_eos_paddings:
                    case_ids.append(case_id)
                    flattened = np.array(padded_slice, dtype=int).squeeze(-1)
                    eos_mask = np.ones_like(flattened, dtype=float)
                    if eos_token_id and eos_token_id > 0:
                        eos_positions = np.flatnonzero(flattened == eos_token_id)
                        if eos_positions.size > 0:
                            first_eos_idx = int(eos_positions[0])
                            eos_mask[first_eos_idx + 1:] = 0.0
                    zero_mask = np.zeros_like(flattened, dtype=float)
                    non_zero_positions = np.flatnonzero(flattened != 0)
                    if non_zero_positions.size > 0:
                        first_valid_idx = int(non_zero_positions[0])
                        zero_mask[first_valid_idx:] = 1.0
                    eos_masks.append(eos_mask.tolist())
                    zero_masks.append(zero_mask.tolist())
            windows.extend(padded_encodings)

        if len(windows) == 0:
            # avoid creating empty numpy array with ambiguous dtype
            padded_array = np.zeros((0, self.window_size), dtype=int)
        else:
            padded_array = np.array(windows, dtype=int)
        t = torch.tensor(padded_array, dtype=torch.long)

        max_classes = len(self.categorical_encoders[col].categories_[0]) + 1
        if return_case_ids_and_eos_paddings:
            if len(eos_masks) == 0:
                eos_padded_array = np.zeros((0, self.window_size), dtype=float)
                zero_padded_array = np.zeros((0, self.window_size), dtype=float)
            else:
                eos_padded_array = np.array(eos_masks, dtype=float)
                zero_padded_array = np.array(zero_masks, dtype=float)
            eos_padded_tensor = torch.tensor(eos_padded_array, dtype=torch.float32)
            zero_padded_tensor = torch.tensor(zero_padded_array, dtype=torch.float32)
            return case_ids, t.squeeze(-1), eos_padded_tensor, zero_padded_tensor, categories, max_classes
        else:
            return t.squeeze(-1), categories, max_classes
    
    def encode_continuous_column(self, df, col):
        grouped = df.groupby(self.case_name)
        windows = []
        for case_id, group in tqdm(grouped, desc=col, leave=False):
            case_values = group[[col]].values  # shape (n,1)
            case_values_imputed = self.continuous_imputers[col].transform(case_values)
            case_values_enc = self.continuous_encoders[col].transform(case_values_imputed)
            padded_encodings = []
            
            # check
            for prefix_len in range(self.min_suffix_size + 1, len(case_values_enc) + 1):
                padded_encodings.append(self.pad_to_window_size(case_values_enc[:prefix_len]))
            windows.extend(padded_encodings)
        
        if len(windows) == 0:
            padded_array = np.zeros((0, self.window_size), dtype=float)
        else:
            padded_array = np.array(windows, dtype=float)
        t = torch.tensor(padded_array, dtype=torch.float32)
        return t.squeeze(-1)
    
    def pad_to_window_size(self, previous_values):
        """
        previous_values: array-like with shape (k, 1)
        returns list of shape (window_size, 1)
        """
        prev_list = np.asarray(previous_values).tolist()
        if len(prev_list) > self.window_size:
            return prev_list[-self.window_size:]
        else:
            pad_count = self.window_size - len(prev_list)
            # use 0.0 for continuous; for categorical it will be cast to int later when dtype=int
            return [[0.0]] * pad_count + prev_list

    def _encode_static_attributes(self, df: pd.DataFrame, case_ids: list[object]) -> tuple[torch.Tensor, torch.Tensor]:
        ordered_case_ids = list(case_ids)
        num_samples = len(ordered_case_ids)
        if not num_samples:
            return (torch.zeros((0, len(self.static_categorical_columns)), dtype=torch.long),
                    torch.zeros((0, len(self.static_continuous_columns)), dtype=torch.float32))

        if not self.static_categorical_columns and not self.static_continuous_columns:
            return (torch.zeros((num_samples, 0), dtype=torch.long),
                    torch.zeros((num_samples, 0), dtype=torch.float32))

        case_static_values = self._collect_static_case_values(df)

        if self.static_categorical_columns:
            cat_rows = []
            for case_id in tqdm(ordered_case_ids, desc="static categorical", leave=False):
                row = []
                case_record = case_static_values.get(case_id, {})
                for col in self.static_categorical_columns:
                    value = case_record.get(col, np.nan)
                    value_arr = np.array([[value]], dtype=object)
                    encoded_value = self.categorical_encoders[col].transform(value_arr) + 1
                    row.append(int(encoded_value.squeeze()))
                cat_rows.append(row)
            static_cat_tensor = torch.tensor(cat_rows, dtype=torch.long)
        else:
            static_cat_tensor = torch.zeros((num_samples, 0), dtype=torch.long)

        if self.static_continuous_columns:
            cont_rows = []
            for case_id in tqdm(ordered_case_ids, desc="static continuous", leave=False):
                row = []
                case_record = case_static_values.get(case_id, {})
                for col in self.static_continuous_columns:
                    value = case_record.get(col, np.nan)
                    value_arr = np.array([[value]], dtype=float)
                    imputed = self.continuous_imputers[col].transform(value_arr)
                    encoded_value = self.continuous_encoders[col].transform(imputed)
                    row.append(float(encoded_value.squeeze()))
                cont_rows.append(row)
            static_cont_tensor = torch.tensor(cont_rows, dtype=torch.float32)
        else:
            static_cont_tensor = torch.zeros((num_samples, 0), dtype=torch.float32)

        return static_cat_tensor, static_cont_tensor

    def _collect_static_case_values(self, df: pd.DataFrame) -> dict[str, dict[str, object]]:
        grouped = df.groupby(self.case_name, sort=False)
        case_values: dict[str, dict[str, object]] = {}
        for case_id, group in grouped:
            record: dict[str, object] = {}
            for col in self.static_categorical_columns:
                if col in group.columns:
                    record[col] = self.__extract_static_value(group[col])
                else:
                    record[col] = np.nan
            for col in self.static_continuous_columns:
                if col in group.columns:
                    record[col] = self.__extract_static_value(group[col])
                else:
                    record[col] = np.nan
            case_values[case_id] = record
        return case_values

    @staticmethod
    def __extract_static_value(series: pd.Series) -> object:
        cleaned = series.dropna()
        if cleaned.empty:
            return np.nan
        return cleaned.iloc[0]

    def decode_event(self, event_tuple : tuple):
        cat, cont, *_, case_id = event_tuple
        decoded_event = dict()
        for i, col in enumerate(self.categorical_columns):
            enc_col = cat[i].unsqueeze(-1).numpy()
            if col in self.categorical_encoders:
                categories = self.categorical_encoders[col].categories_[0]
                dec_col = np.array([categories[idx - 1] if idx > 0 and idx <= len(categories) else np.nan for idx in enc_col.flatten()])
            else:
                dec_col = enc_col
            decoded_event[col] = dec_col.tolist()
        for i, col in enumerate(self.continuous_columns + self.continuous_positive_columns):
            enc_col = cont[i].unsqueeze(-1).numpy()
            if col in self.continuous_encoders:
                dec_col = self.continuous_encoders[col].inverse_transform(enc_col)
            else:
                dec_col = enc_col
            decoded_event[col] = dec_col.flatten().tolist()
        decoded_event[self.case_name] = [case_id] * len(decoded_event[self.categorical_columns[0]])
        return pd.DataFrame(decoded_event)
    
    def __get_continuous_imputer(self):
        return SimpleImputer(strategy='mean')

    def __get_categorical_encoder(self):
        return sklearn.preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                                    unknown_value=-1,
                                                    encoded_missing_value=-1)

    def __get_continuous_encoder(self):
        return sklearn.preprocessing.StandardScaler()

    def __get_continuous_positive_imputer(self):
        return SimpleImputer(strategy='mean')
    
    def __get_continuous_positive_encoder(self):
        standardizer = PositiveStandardizer_normed()
        return standardizer 

# create event log
class EventLogLoader:
    def __init__(self, event_log_location, event_log_properties, prefix_df: Optional[PrefixesDataFrameLoader]=None):
        if prefix_df:
            self.event_log = prefix_df.csv2event_log
            self.train_df = prefix_df.train_df.copy()
            self.val_df = prefix_df.val_df.copy()
            self.test_df = prefix_df.test_df.copy()
        else:
            self.event_log = CSV2EventLog(event_log_location, **event_log_properties) 
            splitter = EventLogSplitter(**event_log_properties)
            self.train_df, self.val_df, self.test_df = splitter.split(self.event_log)
        
        self.encoder_decoder = TensorEncoderDecoder(self.train_df, **event_log_properties)
        # Data are transformed
        self.encoder_decoder.train_imputers_encoders()
                    
    def get_dataset(self, type : str):
        if type == 'train':
            df = self.train_df
        elif type == 'val':
            df = self.val_df
        elif type == 'test':
            df = self.test_df
        else:
            raise ValueError("type must be one of 'train', 'val', or 'test'")
        encoded_data, all_categories, all_static_categories = self.encoder_decoder.encode_df(df)
        return EventLogDataset(encoded_data, all_categories, all_static_categories, self.encoder_decoder)

# return object: returns tensors and all further information
class EventLogDataset(Dataset):
    def __init__(self, tensor_bundle : dict[str, object], all_categories : tuple[list[tuple[str, int, dict[str, int]]]], all_static_categories : tuple[list[tuple[str, int, dict[str, int]]]], encoder_decoder : TensorEncoderDecoder):
        self.tensor_bundle = tensor_bundle
        
        self.case_ids : list[object] = list(tensor_bundle['case_ids'])
        
        self.categorical_tensors : list[torch.Tensor] = tensor_bundle['categorical']
        self.continuous_tensors : list[torch.Tensor] = tensor_bundle['continuous']
        
        self.static_categorical_tensor : torch.Tensor = tensor_bundle['static_categorical']
        self.static_continuous_tensor : torch.Tensor = tensor_bundle['static_continuous']
        
        self.eos_padding : torch.Tensor = tensor_bundle['eos_padding']
        self.zero_padding : torch.Tensor = tensor_bundle['zero_padding']
        
        num_rows = self.eos_padding.shape[0]
        self.prefixes_petri_net_marking : list[Optional[object]] = [None] * num_rows

        self.all_categories : tuple[list[tuple[str, int, dict[str, int]]]] = all_categories
        self.all_static_categories : tuple[list[tuple[str, int, dict[str, int]]]] = all_static_categories
        self.encoder_decoder : TensorEncoderDecoder = encoder_decoder
        self.min_suffix_size : Optional[int] = getattr(encoder_decoder, 'min_suffix_size', None)

    def __len__(self):
        return self.eos_padding.shape[0]

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        
        categorical_items = [tensor[idx] for tensor in self.categorical_tensors]
        continuous_items = [tensor[idx] for tensor in self.continuous_tensors]
        
        eos_mask = self.eos_padding[idx]
        zero_mask = self.zero_padding[idx]
        
        static_cat = self.static_categorical_tensor[idx]
        static_cont = self.static_continuous_tensor[idx]
        
        prefixes_petri_net_marking = self.prefixes_petri_net_marking[idx]
        
        return (case_id,
                tuple(categorical_items),
                tuple(continuous_items),
                eos_mask,
                zero_mask,
                static_cat,
                static_cont,
                prefixes_petri_net_marking)
    
    # Add the petri net markings to the dataset
    def set_prefix_markings(self, markings, indices: Optional[Iterable[int]] = None) -> None:
        markings_list = list(markings)

        if indices is None:
            if len(markings_list) != len(self.prefixes_petri_net_marking):
                raise ValueError("Number of markings must match dataset length when indices are omitted")
            target_indices = range(len(self.prefixes_petri_net_marking))
        else:
            target_indices = list(indices)
            if len(target_indices) != len(markings_list):
                raise ValueError("indices and markings must reference the same number of rows")

        for idx, marking in zip(target_indices, markings_list):
            if isinstance(marking, torch.Tensor):
                marking = marking.detach().cpu().tolist()
            elif isinstance(marking, np.ndarray):
                marking = marking.tolist()
            self.prefixes_petri_net_marking[idx] = marking


# class for pre-processing the data for perturnbation:
class EventLogPerturbationPreprocess:
    def __init__(self, event_log_location, event_log_properties):
        self.event_log_properties = event_log_properties
        
        self.event_log = CSV2EventLog(event_log_location, **event_log_properties) 
        splitter = EventLogSplitter(**event_log_properties)
        self.train_df, self.val_df, self.test_df = splitter.split(self.event_log)

    def get_all_datasets(self):
        return self.train_df, self.val_df, self.test_df

    def extract_feature_info(self):
        categories_info = {}
        ranges_info = {}
        
        # Get all unique categories for each categorical column
        categorical_columns = self.event_log_properties.get('categorical_columns', [])
        for col in categorical_columns:
            if col in self.event_log.df.columns:
                # Get all unique values (excluding NaN)
                unique_values = self.event_log.df[col].dropna().unique()
                # Convert to list and sort for consistency
                categories_info[col] = sorted([str(val) for val in unique_values if pd.notna(val)])
            else:
                categories_info[col] = []
        
        # Get min/max ranges for each continuous column
        continuous_columns = self.event_log_properties .get('continuous_columns', []) + self.event_log_properties .get('continuous_positive_columns', [])
        for col in continuous_columns:
            if col in self.event_log.df.columns:
                # Get non-null values
                col_data = self.event_log.df[col].dropna()
                if len(col_data) > 0:
                    ranges_info[col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std())
                    }
                else:
                    ranges_info[col] = {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
            else:
                ranges_info[col] = {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
        return {'categorical': categories_info, 'continuous': ranges_info}