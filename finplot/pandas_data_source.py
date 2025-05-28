import math

from collections import OrderedDict

import numpy as np
import pandas as pd

from .constants import (
    FinPlotConstants
)

from .global_state import (
    g_fin_plot_global_state
)

from .internal_utils import (
    _has_timecol,
    _pdtime2epoch,
    _is_standalone,
    _xminmax
)


class PandasDataSource(object):
    """Candle sticks: create with five columns: time, open, close, hi, lo - in that order.
       Volume bars: create with three columns: time, open, close, volume - in that order.
       For all other types, time needs to be first, usually followed by one or more Y-columns."""
    def __init__(self, df):
        if type(df.index) == pd.DatetimeIndex or df.index[-1] > 1e8 or '.RangeIndex' not in str(type(df.index)):
            df = df.reset_index()
        self.df = df.copy()
        # manage time column
        if _has_timecol(self.df):
            timecol = self.df.columns[0]

            dtype = (
                str(
                    (
                        df[
                            timecol
                        ]
                    ).dtype
                )
            )

            isnum = (
                (
                    'int' in dtype or
                    'float' in dtype
                ) and

                (
                    (
                        df[
                            timecol
                        ]
                    ).iloc[
                        -1
                    ] <

                    1e7
                )
            )

            if not isnum:
                (
                    self.df[
                        timecol
                    ]
                ) = (
                    _pdtime2epoch(
                        df[
                            timecol
                        ]
                    )
                )

            self.standalone = (
                _is_standalone(
                    self.df[
                        timecol
                    ]
                )
            )

            self.col_data_offset = 1  # no. of preceeding columns for other plots and time column
        else:
            self.standalone = False
            self.col_data_offset = 0  # no. of preceeding columns for other plots and time column

        # setup data for joining data sources and zooming
        self.scale_cols = [
            i

            for i in range(
                self.col_data_offset,
                len(
                    self.df.columns
                )
            )

            if (
                    (
                        self.df.iloc[
                            :,
                            i
                        ]
                    ).dtype !=

                    object
            )
        ]
        self.cache_hilo = OrderedDict()
        self.renames = {}
        newcols = []

        for col in self.df.columns:
            oldcol = col
            while col in newcols:
                col = str(col)+'+'
            newcols.append(col)
            if oldcol != col:
                self.renames[oldcol] = col
        self.df.columns = newcols
        self.pre_update = lambda df: df
        self.post_update = lambda df: df
        self._period = None
        self._smooth_time = None
        self.is_sparse = self.df[self.df.columns[self.col_data_offset]].isnull().sum().max() > len(self.df)//2

    @property
    def period_ns(self):
        if len(self.df) <= 1:
            return 1
        if not self._period:
            self._period = self.calc_period_ns()
        return self._period

    def calc_period_ns(self, n=100, delta=lambda dt: int(dt.median())):
        timecol = self.df.columns[0]
        dtimes = self.df[timecol].iloc[0:n].diff()
        dtimes = dtimes[dtimes!=0]
        return delta(dtimes) if len(dtimes)>1 else 1

    @property
    def index(self):
        return self.df.index

    @property
    def x(self):
        timecol = self.df.columns[0]
        return self.df[timecol]

    @property
    def y(self):
        col = self.df.columns[self.col_data_offset]
        return self.df[col]

    @property
    def z(self):
        col = self.df.columns[self.col_data_offset+1]
        return self.df[col]

    @property
    def xlen(self):
        return len(self.df)

    def calc_significant_decimals(self, full):
        def float_round(f):
            return float('%.3e'%f) # 0.00999748 -> 0.01
        def remainder_ok(a, b):
            c = a % b
            if c / b > 0.98: # remainder almost same as denominator
                c = abs(c-b)
            return c < b*0.6 # half is fine
        def calc_sd(ser):
            ser = ser.iloc[:1000]
            absdiff = ser.diff().abs()
            absdiff[absdiff<1e-30] = np.float32(1e30)
            smallest_diff = absdiff.min()
            if smallest_diff > 1e29: # just 0s?
                return 0
            smallest_diff = float_round(smallest_diff)
            absser = ser.iloc[:100].abs()
            for _ in range(2): # check if we have a remainder that is a better epsilon
                remainder = [math.fmod(v,smallest_diff) for v in absser]
                remainder = [v for v in remainder if v>smallest_diff/20]
                if not remainder:
                    break
                smallest_diff_r = min(remainder)
                if smallest_diff*0.05 < smallest_diff_r < smallest_diff * 0.7 and remainder_ok(smallest_diff, smallest_diff_r):
                    smallest_diff = smallest_diff_r
                else:
                    break
            return smallest_diff
        def calc_dec(ser, smallest_diff):
            if not full: # line plots usually have extreme resolution
                absmax = ser.iloc[:300].abs().max()
                s = '%.3e' % absmax
            else: # candles
                s = '%.2e' % smallest_diff
            base,_,exp = s.partition('e')
            base = base.rstrip('0')
            exp = -int(exp)
            max_base_decimals = min(5, -exp+2) if exp < 0 else 3
            base_decimals = max(0, min(max_base_decimals, len(base)-2))
            decimals = exp + base_decimals
            decimals = max(0, min(FinPlotConstants.max_decimals, decimals))
            if not full: # apply grid for line plots only
                smallest_diff = max(10**(-decimals), smallest_diff)
            return decimals, smallest_diff
        # first calculate EPS for series 0&1, then do decimals
        sds = [calc_sd(self.y)] # might be all zeros for bar charts
        if len(self.scale_cols) > 1:
            sds.append(calc_sd(self.z)) # if first is open, this might be close
        sds = [sd for sd in sds if sd>0]
        big_diff = max(sds)
        smallest_diff = min([sd for sd in sds if sd>big_diff/100]) # filter out extremely small epsilons
        ser = self.z if len(self.scale_cols) > 1 else self.y
        return calc_dec(ser, smallest_diff)

    def update_init_x(
            self,

            init_steps
    ):
        (
            self.init_x0,
            self.init_x1
        ) = (
            _xminmax(
                self,

                x_indexed=(
                    True
                ),

                init_steps=(
                    init_steps
                )
            )
        )

    def closest_time(self, x):
        timecol = self.df.columns[0]
        return self.df.loc[int(x), timecol]

    def timebased(self):
        return self.df.iloc[-1,0] > 1e7

    def is_smooth_time(self):
        if self._smooth_time is None:
            # less than 1% time delta is smooth
            self._smooth_time = self.timebased() and (np.abs(np.diff(self.x.values[1:100])[1:]//(self.period_ns//1000)-1000) < 10).all()
        return self._smooth_time

    def addcols(self, datasrc):
        new_scale_cols = [c+len(self.df.columns)-datasrc.col_data_offset for c in datasrc.scale_cols]
        self.scale_cols += new_scale_cols
        orig_col_data_cnt = len(self.df.columns)
        if _has_timecol(datasrc.df):
            timecol = self.df.columns[0]
            df = self.df.set_index(timecol)
            timecol = timecol if timecol in datasrc.df.columns else datasrc.df.columns[0]
            newcols = datasrc.df.set_index(timecol)
        else:
            df = self.df
            newcols = datasrc.df
        cols = list(newcols.columns)
        for i,col in enumerate(cols):
            old_col = col
            while col in self.df.columns:
                cols[i] = col = str(col)+'+'
            if old_col != col:
                datasrc.renames[old_col] = col
        newcols.columns = cols
        self.df = df.join(newcols, how='outer')
        if _has_timecol(datasrc.df):
            self.df.reset_index(inplace=True)
        datasrc.df = self.df # they are the same now
        datasrc.init_x0 = self.init_x0
        datasrc.init_x1 = self.init_x1
        datasrc.col_data_offset = orig_col_data_cnt
        datasrc.scale_cols = new_scale_cols
        self.cache_hilo = OrderedDict()
        self._period = self._smooth_time = None
        datasrc._period = datasrc._smooth_time = None
        ldf2 = len(self.df) // 2
        self.is_sparse = self.is_sparse or self.df[self.df.columns[self.col_data_offset]].isnull().sum().max() > ldf2
        datasrc.is_sparse = datasrc.is_sparse or datasrc.df[datasrc.df.columns[datasrc.col_data_offset]].isnull().sum().max() > ldf2

    def update(self, datasrc):
        df = self.pre_update(self.df)
        orig_cols = list(df.columns)
        timecol, orig_cols = orig_cols[0],orig_cols[1:]

        df = (
            df.set_index(
                timecol
            )
        )

        input_df = (
            datasrc.df.set_index(
                datasrc.df.columns[0]
            )
        )

        input_df.columns = [self.renames.get(col, col) for col in input_df.columns]
        # pad index if the input data is a sub-set
        if (
                len(input_df) > 0 and
                len(df) > 0 and

                (
                    len(df) != len(input_df) or
                    input_df.index[-1] != df.index[-1]
                )
        ):
            output_df = (
                input_df.copy()
            )
        else:
            output_df = input_df
        for col in df.columns:
            if col not in output_df.columns:
                output_df[col] = df[col]
        # if neccessary, cut out unwanted data
        if len(input_df) > 0 and len(df) > 0:
            start_idx = end_idx = None
            if input_df.index[0] > df.index[0]:
                start_idx = 0
            if input_df.index[-1] < df.index[-1]:
                end_idx = -1
            if start_idx is not None or end_idx is not None:
                end_idx = None if end_idx == -1 else end_idx
                output_df = output_df.loc[input_df.index[start_idx:end_idx], :]
        output_df = self.post_update(output_df)
        output_df = output_df.reset_index()
        self.df = output_df[[output_df.columns[0]]+orig_cols] if orig_cols else output_df

        self.init_x1 = (  # TODO
            self.xlen +
            g_fin_plot_global_state.get_right_margin_candles() -
            FinPlotConstants.side_margin
        )

        self.cache_hilo = OrderedDict()
        self._period = self._smooth_time = None

    def set_df(self, df):
        self.df = df
        self.cache_hilo = OrderedDict()
        self._period = self._smooth_time = None

    def hilo(self, x0, x1):
        """Return five values in time range: t0, t1, highest, lowest, number of rows."""
        if x0 == x1:
            x0 = x1 = int(x1)
        else:
            x0,x1 = int(x0+0.5),int(x1)
        query = '%i,%i' % (x0,x1)
        if query not in self.cache_hilo:
            v = self.cache_hilo[query] = self._hilo(x0, x1)
        else:
            # re-insert to raise prio
            v = self.cache_hilo[query] = self.cache_hilo.pop(query)
        if len(self.cache_hilo) > 100: # drop if too many
            del self.cache_hilo[next(iter(self.cache_hilo))]
        return v

    def _hilo(self, x0, x1):
        df = self.df.loc[x0:x1, :]
        if not len(df):
            return 0,0,0,0,0
        timecol = df.columns[0]
        t0 = df[timecol].iloc[0]
        t1 = df[timecol].iloc[-1]
        valcols = df.columns[self.scale_cols]
        hi = df[valcols].max().max()
        lo = df[valcols].min().min()
        return t0,t1,hi,lo,len(df)

    def rows(self, colcnt, x0, x1, yscale, lod=True, resamp=None):
        df = self.df.loc[x0:x1, :]
        if self.is_sparse:
            df = df.loc[df.iloc[:,self.col_data_offset].notna(), :]
        origlen = len(df)
        return self._rows(df, colcnt, yscale=yscale, lod=lod, resamp=resamp), origlen

    def _rows(self, df, colcnt, yscale, lod, resamp):
        colcnt -= 1 # time is always implied
        colidxs = [0] + list(range(self.col_data_offset, self.col_data_offset+colcnt))
        if lod and len(df) > FinPlotConstants.lod_candles:
            if resamp:
                df = self._resample(df, colcnt, resamp)
                colidxs = None
            else:
                df = df.iloc[::len(df)//FinPlotConstants.lod_candles]
        dfr = df.iloc[:,colidxs] if colidxs else df
        if yscale.scaletype == 'log' or yscale.scalef != 1:
            dfr = dfr.copy()
            for i in range(1, colcnt+1):
                colname = dfr.columns[i]
                if dfr[colname].dtype != object:
                    dfr[colname] = yscale.invxform(dfr.iloc[:,i])
        return dfr

    def _resample(
            self,

            df,
            colcnt,
            resamp
    ):
        cdo = self.col_data_offset
        sample_rate = len(df) * 5 // FinPlotConstants.lod_candles
        offset = len(df) % sample_rate
        dfd = df[[df.columns[0]]+[df.columns[cdo]]].iloc[offset::sample_rate]
        c = df[df.columns[cdo+1]].iloc[offset+sample_rate-1::sample_rate]
        c.index -= sample_rate - 1

        (
            dfd[
                df.columns[
                    cdo + 1
                ]
            ]
        ) = c

        if resamp == 'hilo':
            (
                dfd[
                    df.columns[
                        cdo +
                        2
                    ]
                ]
            ) = (
                df[
                    df.columns[
                        cdo +
                        2
                    ]
                ].rolling(
                    sample_rate
                ).max().shift(
                    -sample_rate +
                    1
                )
            )

            (
                dfd[
                    df.columns[
                        cdo +
                        3
                    ]
                ]
            ) = (
                df[
                    df.columns[
                        cdo +
                        3
                    ]
                ].rolling(
                    sample_rate
                ).min().shift(
                    -sample_rate +
                    1
                )
            )
        else:
            (
                dfd[
                    df.columns[
                        cdo +
                        2
                    ]
                ]
            ) = (
                df[
                    df.columns[
                        cdo +
                        2
                    ]
                ].rolling(
                    sample_rate
                ).sum().shift(
                    -sample_rate +
                    1
                )
            )

            (
                dfd[
                    df.columns[
                        cdo +
                        3
                    ]
                ]
            ) = (
                df[
                    df.columns[
                        cdo +
                        3
                    ]
                ].rolling(
                    sample_rate
                ).sum().shift(
                    -sample_rate +
                    1
                )
            )

        # append trailing columns
        trailing_colidx = cdo + 4

        for i in range(trailing_colidx, colcnt):
            col = df.columns[i]
            dfd[col] = df[col].iloc[offset::sample_rate]

        return dfd

    def __eq__(self, other):
        return id(self) == id(other) or id(self.df) == id(other.df)

    def __hash__(self):
        return id(self)
