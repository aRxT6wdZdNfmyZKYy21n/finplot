# TODO: decomposite

from collections import OrderedDict

from functools import partial, partialmethod
from finplot.constants import *  # TODO
from finplot.yscale import YScale
from math import fmod
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from finplot.utils import (
    _has_timecol,
    _pdtime2epoch,
    _is_standalone,
    _xminmax,
    _savewindata,
    _clear_timers,
    _clamp_xy,
    _x2local_t,
    _round_to_significant,
    _draw_line_segment_text,
    _clamp_point,
    _roihandle_move_snap,
    _mouse_moved,
    add_vertical_band,
    _set_clamp_pos,
    _create_poly_line,
    _mouse_clicked,
    _key_pressed,
    _pdtime2index,
    _round,
)


class PandasDataSource:
    '''Candle sticks: create with five columns: time, open, close, hi, lo - in that order.
       Volume bars: create with three columns: time, open, close, volume - in that order.
       For all other types, time needs to be first, usually followed by one or more Y-columns.'''
    def __init__(self, df):
        if type(df.index) == pd.DatetimeIndex or df.index[-1]>1e8 or '.RangeIndex' not in str(type(df.index)):
            df = df.reset_index()
        self.df = df.copy()
        # manage time column
        if _has_timecol(self.df):
            timecol = self.df.columns[0]
            dtype = str(df[timecol].dtype)
            isnum = ('int' in dtype or 'float' in dtype) and df[timecol].iloc[-1] < 1e7
            if not isnum:
                self.df[timecol] = _pdtime2epoch(df[timecol])
            self.standalone = _is_standalone(self.df[timecol])
            self.col_data_offset = 1 # no. of preceeding columns for other plots and time column
        else:
            self.standalone = False
            self.col_data_offset = 0 # no. of preceeding columns for other plots and time column
        # setup data for joining data sources and zooming
        self.scale_cols = [i for i in range(self.col_data_offset,len(self.df.columns)) if self.df.iloc[:,i].dtype!=object]
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
                remainder = [fmod(v,smallest_diff) for v in absser]
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
            decimals = max(0, min(max_decimals, decimals))
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

    def update_init_x(self, init_steps):
        self.init_x0, self.init_x1 = _xminmax(self, x_indexed=True, init_steps=init_steps)

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
        timecol,orig_cols = orig_cols[0],orig_cols[1:]
        df = df.set_index(timecol)
        input_df = datasrc.df.set_index(datasrc.df.columns[0])
        input_df.columns = [self.renames.get(col, col) for col in input_df.columns]
        # pad index if the input data is a sub-set
        if len(input_df) > 0 and len(df) > 0 and (len(df) != len(input_df) or input_df.index[-1] != df.index[-1]):
            output_df = pd.merge(input_df, df[[]], how='outer', left_index=True, right_index=True)
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
        self.init_x1 = self.xlen + right_margin_candles - side_margin
        self.cache_hilo = OrderedDict()
        self._period = self._smooth_time = None

    def set_df(self, df):
        self.df = df
        self.cache_hilo = OrderedDict()
        self._period = self._smooth_time = None

    def hilo(self, x0, x1):
        '''Return five values in time range: t0, t1, highest, lowest, number of rows.'''
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
        if lod and len(df) > lod_candles:
            if resamp:
                df = self._resample(df, colcnt, resamp)
                colidxs = None
            else:
                df = df.iloc[::len(df)//lod_candles]
        dfr = df.iloc[:,colidxs] if colidxs else df
        if yscale.scaletype == 'log' or yscale.scalef != 1:
            dfr = dfr.copy()
            for i in range(1, colcnt+1):
                colname = dfr.columns[i]
                if dfr[colname].dtype != object:
                    dfr[colname] = yscale.invxform(dfr.iloc[:,i])
        return dfr

    def _resample(self, df, colcnt, resamp):
        cdo = self.col_data_offset
        sample_rate = len(df) * 5 // lod_candles
        offset = len(df) % sample_rate
        dfd = df[[df.columns[0]]+[df.columns[cdo]]].iloc[offset::sample_rate]
        c = df[df.columns[cdo+1]].iloc[offset+sample_rate-1::sample_rate]
        c.index -= sample_rate - 1
        dfd[df.columns[cdo+1]] = c
        if resamp == 'hilo':
            dfd[df.columns[cdo+2]] = df[df.columns[cdo+2]].rolling(sample_rate).max().shift(-sample_rate+1)
            dfd[df.columns[cdo+3]] = df[df.columns[cdo+3]].rolling(sample_rate).min().shift(-sample_rate+1)
        else:
            dfd[df.columns[cdo+2]] = df[df.columns[cdo+2]].rolling(sample_rate).sum().shift(-sample_rate+1)
            dfd[df.columns[cdo+3]] = df[df.columns[cdo+3]].rolling(sample_rate).sum().shift(-sample_rate+1)
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


class FinWindow(pg.GraphicsLayoutWidget):
    def __init__(self, title, **kwargs):
        global winx, winy
        self.title = title
        pg.mkQApp()
        super().__init__(**kwargs)
        self.setWindowTitle(title)
        self.setGeometry(winx, winy, winw, winh)
        winx = (winx+win_recreate_delta) % 800
        winy = (winy+win_recreate_delta) % 500
        self.centralWidget.installEventFilter(self)
        self.ci.setContentsMargins(0, 0, 0, 0)
        self.ci.setSpacing(-1)
        self.closing = False

    @property
    def axs(self):
        return [ax for ax in self.ci.items if isinstance(ax, pg.PlotItem)]

    def autoRangeEnabled(self):
        return [True, True]

    def close(self):
        self.closing = True
        _savewindata(self)
        _clear_timers()
        return super().close()

    def eventFilter(self, obj, ev):
        if ev.type()== QtCore.QEvent.Type.WindowDeactivate:
            _savewindata(self)
        return False

    def resizeEvent(self, ev):
        '''We resize and set the top Y axis larger according to the axis_height_factor.
           No point in trying to use the "row stretch factor" in Qt which is broken
           beyond repair.'''
        if ev and not self.closing:
            axs = self.axs
            new_win_height = ev.size().height()
            old_win_height = ev.oldSize().height() if ev.oldSize().height() > 0 else new_win_height
            client_borders = old_win_height - sum(ax.vb.size().height() for ax in axs)
            client_borders = min(max(client_borders, 0), 30) # hrm
            new_height = new_win_height - client_borders
            for i,ax in enumerate(axs):
                j = axis_height_factor.get(i, 1)
                f = j / (len(axs)+sum(axis_height_factor.values())-len(axis_height_factor))
                ax.setMinimumSize(100 if j>1 else 50, new_height*f)
        return super().resizeEvent(ev)

    def leaveEvent(self, ev):
        if not self.closing:
            super().leaveEvent(ev)


class FinCrossHair:
    def __init__(self, ax, color):
        self.ax = ax
        self.x = 0
        self.y = 0
        self.clamp_x = 0
        self.clamp_y = 0
        self.infos = []
        pen = pg.mkPen(color=color, style=QtCore.Qt.PenStyle.CustomDashLine, dash=[7, 7])
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self.xtext = pg.TextItem(color=color, anchor=(0,1))
        self.ytext = pg.TextItem(color=color, anchor=(0,0))
        self.vline.setZValue(50)
        self.hline.setZValue(50)
        self.xtext.setZValue(50)
        self.ytext.setZValue(50)
        self.show()

    def update(self, point=None):
        if point is not None:
            self.x,self.y = x,y = point.x(),point.y()
        else:
            x,y = self.x,self.y
        x,y = _clamp_xy(self.ax, x,y)
        if x == self.clamp_x and y == self.clamp_y:
            return
        self.clamp_x,self.clamp_y = x,y
        self.vline.setPos(x)
        self.hline.setPos(y)
        self.xtext.setPos(x, y)
        self.ytext.setPos(x, y)
        rng = self.ax.vb.y_max - self.ax.vb.y_min
        rngmax = abs(self.ax.vb.y_min) + rng # any approximation is fine
        sd,se = (self.ax.significant_decimals,self.ax.significant_eps) if clamp_grid else (significant_decimals,significant_eps)
        timebased = False
        if self.ax.vb.x_indexed:
            xtext,timebased = _x2local_t(self.ax.vb.datasrc, x)
        else:
            xtext = _round_to_significant(rng, rngmax, x, sd, se)
        linear_y = y
        y = self.ax.vb.yscale.xform(y)
        ytext = _round_to_significant(rng, rngmax, y, sd, se)
        if not timebased:
            if xtext:
                xtext = 'x ' + xtext
            ytext = 'y ' + ytext
        screen_pos = self.ax.mapFromView(pg.Point(x, linear_y))
        far_right = self.ax.boundingRect().right() - crosshair_right_margin
        far_bottom = self.ax.boundingRect().bottom() - crosshair_bottom_margin
        close2right = screen_pos.x() > far_right
        close2bottom = screen_pos.y() > far_bottom
        try:
            for info in self.infos:
                xtext,ytext = info(x,y,xtext,ytext)
        except Exception as e:
            print('Crosshair error:', type(e), e)
        space = '      '
        if close2right:
            xtext = xtext + space
            ytext = ytext + space
            xanchor = [1,1]
            yanchor = [1,0]
        else:
            xtext = space + xtext
            ytext = space + ytext
            xanchor = [0,1]
            yanchor = [0,0]
        if close2bottom:
            yanchor = [1,1]
            if close2right:
                xanchor = [1,2]
            else:
                ytext = ytext + space
        self.xtext.setAnchor(xanchor)
        self.ytext.setAnchor(yanchor)
        self.xtext.setText(xtext)
        self.ytext.setText(ytext)

    def show(self):
        self.ax.addItem(self.vline, ignoreBounds=True)
        self.ax.addItem(self.hline, ignoreBounds=True)
        self.ax.addItem(self.xtext, ignoreBounds=True)
        self.ax.addItem(self.ytext, ignoreBounds=True)

    def hide(self):
        self.ax.removeItem(self.xtext)
        self.ax.removeItem(self.ytext)
        self.ax.removeItem(self.vline)
        self.ax.removeItem(self.hline)


class FinPolyLine(pg.PolyLineROI):
    def __init__(self, vb, *args, **kwargs):
        self.vb = vb # init before parent constructor
        self.texts = []
        super().__init__(*args, **kwargs)

    def addSegment(self, h1, h2, index=None):
        super().addSegment(h1, h2, index)
        text = pg.TextItem(color=draw_line_color)
        text.setZValue(50)
        text.segment = self.segments[-1 if index is None else index]
        if index is None:
            self.texts.append(text)
        else:
            self.texts.insert(index, text)
        self.update_text(text)
        self.vb.addItem(text, ignoreBounds=True)

    def removeSegment(self, seg):
        super().removeSegment(seg)
        for text in list(self.texts):
            if text.segment == seg:
                self.vb.removeItem(text)
                self.texts.remove(text)

    def update_text(self, text):
        h0 = text.segment.handles[0]['item']
        h1 = text.segment.handles[1]['item']
        diff = h1.pos() - h0.pos()
        if diff.y() < 0:
            text.setAnchor((0.5,0))
        else:
            text.setAnchor((0.5,1))
        text.setPos(h1.pos())
        text.setText(_draw_line_segment_text(self, text.segment, h0.pos(), h1.pos()))

    def update_texts(self):
        for text in self.texts:
            self.update_text(text)

    def movePoint(self, handle, pos, modifiers=QtCore.Qt.KeyboardModifier, finish=True, coords='parent'):
        super().movePoint(handle, pos, modifiers, finish, coords)
        self.update_texts()

    def segmentClicked(self, segment, ev=None, pos=None):
        pos = segment.mapToParent(ev.pos())
        pos = _clamp_point(self.vb.parent(), pos)
        super().segmentClicked(segment, pos=pos)
        self.update_texts()

    def addHandle(self, info, index=None):
        handle = super().addHandle(info, index)
        handle.movePoint = partial(_roihandle_move_snap, self.vb, handle.movePoint)
        return handle


class FinLine(pg.GraphicsObject):
    def __init__(self, points, pen):
        super().__init__()
        self.points = points
        self.pen = pen

    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawPath(self.shape())

    def shape(self):
        p = QtGui.QPainterPath()
        p.moveTo(*self.points[0])
        p.lineTo(*self.points[1])
        return p

    def boundingRect(self):
        return self.shape().boundingRect()


class FinEllipse(pg.EllipseROI):
    def addRotateHandle(self, *args, **kwargs):
        pass


class FinRect(pg.RectROI):
    def __init__(self, ax, brush, *args, **kwargs):
        self.ax = ax
        self.brush = brush
        super().__init__(*args, **kwargs)

    def paint(self, p, *args):
        r = QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()
        p.setPen(self.currentPen)
        p.setBrush(self.brush)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)

    def addScaleHandle(self, *args, **kwargs):
        if self.resizable:
            super().addScaleHandle(*args, **kwargs)


class FinViewBox(pg.ViewBox):
    def __init__(self, win, init_steps=300, yscale=YScale('linear', 1), v_zoom_scale=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.win = win
        self.init_steps = init_steps
        self.yscale = yscale
        self.v_zoom_scale = v_zoom_scale
        self.master_viewbox = None
        self.rois = []
        self.vband = None
        self.win._isMouseLeftDrag = False
        self.zoom_listeners = set()
        self.reset()

    def reset(self):
        self.v_zoom_baseline = 0.5
        self.v_autozoom = True
        self.max_zoom_points_f = 1
        self.y_max = 1000
        self.y_min = 0
        self.y_positive = True
        self.x_indexed = True
        self.force_range_update = 0
        while self.rois:
            self.remove_last_roi()
        self.draw_line = None
        self.drawing = False
        self.standalones = set()
        self.updating_linked = False
        self.set_datasrc(None)
        self.setMouseEnabled(x=True, y=False)
        self.setRange(QtCore.QRectF(pg.Point(0, 0), pg.Point(1, 1)))

    def set_datasrc(self, datasrc):
        self.datasrc = datasrc
        if not self.datasrc:
            return
        datasrc.update_init_x(self.init_steps)

    def pre_process_data(self):
        if self.datasrc and self.datasrc.scale_cols:
            df = self.datasrc.df.iloc[:, self.datasrc.scale_cols]
            self.y_max = df.max().max()
            self.y_min = df.min().min()
            if self.y_min <= 0:
                self.y_positive = False
            if self.y_min < 0:
                self.v_zoom_baseline = 0.5

    @property
    def datasrc_or_standalone(self):
        ds = self.datasrc
        if not ds and self.standalones:
            ds = next(iter(self.standalones))
        return ds

    def wheelEvent(self, ev, axis=None):
        if self.master_viewbox:
            return self.master_viewbox.wheelEvent(ev, axis=axis)
        if ev.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            scale_fact = 1
            self.v_zoom_scale /= 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        else:
            scale_fact = 1.02 ** (ev.delta() * self.state['wheelScaleFactor'])
        vr = self.targetRect()
        center = self.mapToView(ev.scenePos())
        pct_x = (center.x()-vr.left()) / vr.width()
        if pct_x < 0.05: # zoom to far left => all the way left
            center = pg.Point(vr.left(), center.y())
        elif pct_x > 0.95: # zoom to far right => all the way right
            center = pg.Point(vr.right(), center.y())
        self.zoom_rect(vr, scale_fact, center)
        # update crosshair
        _mouse_moved(self.win, self, None)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        axis = 0 # don't constrain drag direction
        if self.master_viewbox:
            return self.master_viewbox.mouseDragEvent(ev, axis=axis)
        if not self.datasrc:
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.mouseLeftDrag(ev, axis)
        elif ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.mouseMiddleDrag(ev, axis)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.mouseRightDrag(ev, axis)
        else:
            super().mouseDragEvent(ev, axis)

    def mouseLeftDrag(self, ev, axis):
        '''
            LButton drag pans.
            Shift+LButton drag draws vertical bars ("selections").
            Ctrl+LButton drag draw lines.
        '''
        pan_drag = ev.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier
        select_band_drag = not self.drawing and (self.vband is not None or ev.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier)
        draw_drag = self.vband is None and (self.drawing or ev.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier)

        if pan_drag:
            super().mouseDragEvent(ev, axis)
            if ev.isFinish():
                self.win._isMouseLeftDrag = False
            else:
                self.win._isMouseLeftDrag = True
            if ev.isFinish() or draw_drag or select_band_drag:
                self.refresh_all_y_zoom()

        if select_band_drag:
            p = self.mapToView(ev.pos())
            p = _clamp_point(self.parent(), p)
            if self.vband is None:
                p0 = self.mapToView(ev.buttonDownPos())
                p0 = _clamp_point(self.parent(), p0)
                x = self.datasrc.x
                x0, x1 = x[int(p0.x())], x[min(len(x)-1, int(p.x())+1)]
                self.vband = add_vertical_band(x0, x1, color=draw_band_color, ax=self.parent())
                self.vband.setMovable(True)
                _set_clamp_pos(self.vband.lines[0])
                _set_clamp_pos(self.vband.lines[1])
            else:
                rgn = (self.vband.lines[0].value(), int(p.x()))
                self.vband.setRegion(rgn)
            if ev.isFinish():
                self.rois += [self.vband]
                self.vband = None

        if draw_drag:
            if self.draw_line and not self.drawing:
                self.set_draw_line_color(draw_done_color)
            p1 = self.mapToView(ev.pos())
            p1 = _clamp_point(self.parent(), p1)
            if not self.drawing:
                # add new line
                p0 = self.mapToView(ev.lastPos())
                p0 = _clamp_point(self.parent(), p0)
                self.draw_line = _create_poly_line(self, [p0, p1], closed=False, pen=pg.mkPen(draw_line_color), movable=False)
                self.draw_line.setZValue(40)
                self.rois.append(self.draw_line)
                self.addItem(self.draw_line)
                self.drawing = True
            else:
                # draw placed point at end of poly-line
                self.draw_line.movePoint(-1, p1)
            if ev.isFinish():
                self.drawing = False

        ev.accept()

    def mouseMiddleDrag(self, ev, axis):
        '''Ctrl+MButton draw ellipses.'''
        if ev.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier:
            return super().mouseDragEvent(ev, axis)
        p1 = self.mapToView(ev.pos())
        p1 = _clamp_point(self.parent(), p1)
        def nonzerosize(a, b):
            c = b-a
            return pg.Point(abs(c.x()) or 1, abs(c.y()) or 1e-3)
        if not self.drawing:
            # add new ellipse
            p0 = self.mapToView(ev.lastPos())
            p0 = _clamp_point(self.parent(), p0)
            s = nonzerosize(p0, p1)
            p0 = QtCore.QPointF(p0.x()-s.x()/2, p0.y()-s.y()/2)
            self.draw_ellipse = FinEllipse(p0, s, pen=pg.mkPen(draw_line_color), movable=True)
            self.draw_ellipse.setZValue(80)
            self.rois.append(self.draw_ellipse)
            self.addItem(self.draw_ellipse)
            self.drawing = True
        else:
            c = self.draw_ellipse.pos() + self.draw_ellipse.size()*0.5
            s = nonzerosize(c, p1)
            self.draw_ellipse.setSize(s*2, update=False)
            self.draw_ellipse.setPos(c-s)
        if ev.isFinish():
            self.drawing = False
        ev.accept()

    def mouseRightDrag(self, ev, axis):
        '''RButton drag is box zoom. At least for now.'''
        ev.accept()
        if not ev.isFinish():
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        else:
            self.rbScaleBox.hide()
            ax = QtCore.QRectF(pg.Point(ev.buttonDownPos(ev.button())), pg.Point(ev.pos()))
            ax = self.childGroup.mapRectFromParent(ax)
            if ax.width() < 2: # zooming this narrow is probably a mistake
                ax.adjust(-1, 0, +1, 0)
            self.showAxRect(ax)
            self.axHistoryPointer += 1
            self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]

    def mouseClickEvent(self, ev):
        if self.master_viewbox:
            return self.master_viewbox.mouseClickEvent(ev)
        if _mouse_clicked(self, ev):
            ev.accept()
            return
        if ev.button() != QtCore.Qt.MouseButton.LeftButton or ev.modifiers() != QtCore.Qt.KeyboardModifier.ControlModifier or not self.draw_line:
            return super().mouseClickEvent(ev)
        # add another segment to the currently drawn line
        p = self.mapClickToView(ev.pos())
        p = _clamp_point(self.parent(), p)
        self.append_draw_segment(p)
        self.drawing = False
        ev.accept()

    def mapClickToView(self, pos):
        '''mapToView() does not do grids properly in embedded widgets. Strangely, only affect clicks, not drags.'''
        if self.win.parent() is not None:
            ax = self.parent()
            if ax.getAxis('right').grid:
                pos.setX(pos.x() + self.width())
            elif ax.getAxis('bottom').grid:
                pos.setY(pos.y() + self.height())
        return super().mapToView(pos)

    def keyPressEvent(self, ev):
        if self.master_viewbox:
            return self.master_viewbox.keyPressEvent(ev)
        if _key_pressed(self, ev):
            ev.accept()
            return
        super().keyPressEvent(ev)

    def linkedViewChanged(self, view, axis):
        if not self.datasrc or self.updating_linked:
            return
        if view and self.datasrc and view.datasrc:
            self.updating_linked = True
            tr = self.targetRect()
            vr = view.targetRect()
            is_dirty = view.force_range_update > 0
            is_same_scale = self.datasrc.xlen == view.datasrc.xlen
            if is_same_scale: # stable zoom based on index
                if is_dirty or abs(vr.left()-tr.left()) >= 1 or abs(vr.right()-tr.right()) >= 1:
                    if is_dirty:
                        view.force_range_update -= 1
                    self.update_y_zoom(vr.left(), vr.right())
            else: # sloppy one based on time stamps
                tt0,tt1,_,_,_ = self.datasrc.hilo(tr.left(), tr.right())
                vt0,vt1,_,_,_ = view.datasrc.hilo(vr.left(), vr.right())
                period2 = self.datasrc.period_ns * 0.5
                if is_dirty or abs(vt0-tt0) >= period2 or abs(vt1-tt1) >= period2:
                    if is_dirty:
                        view.force_range_update -= 1
                    if self.parent():
                        x0,x1 = _pdtime2index(self.parent(), pd.Series([vt0,vt1]), any_end=True)
                        self.update_y_zoom(x0, x1)
            self.updating_linked = False

    def zoom_rect(self, vr, scale_fact, center):
        if not self.datasrc:
            return
        x0 = center.x() + (vr.left()-center.x()) * scale_fact
        x1 = center.x() + (vr.right()-center.x()) * scale_fact
        self.update_y_zoom(x0, x1)

    def pan_x(self, steps=None, percent=None):
        if self.datasrc is None:
            return
        if steps is None:
            steps = int(percent/100*self.targetRect().width())
        tr = self.targetRect()
        x1 = tr.right() + steps
        startx = -side_margin
        endx = self.datasrc.xlen + right_margin_candles - side_margin
        if x1 > endx:
            x1 = endx
        x0 = x1 - tr.width()
        if x0 < startx:
            x0 = startx
            x1 = x0 + tr.width()
        self.update_y_zoom(x0, x1)

    def refresh_all_y_zoom(self):
        '''This updates Y zoom on all views, such as when a mouse drag is completed.'''
        main_vb = self
        if self.linkedView(0):
            self.force_range_update = 1 # main need to update only once to us
            main_vb = list(self.win.axs)[0].vb
        main_vb.force_range_update = len(self.win.axs)-1 # update main as many times as there are other rows
        self.update_y_zoom()
        # refresh crosshair when done
        _mouse_moved(self.win, self, None)

    def update_y_zoom(self, x0=None, x1=None):
        datasrc = self.datasrc_or_standalone
        if datasrc is None:
            return
        if x0 is None or x1 is None:
            tr = self.targetRect()
            x0 = tr.left()
            x1 = tr.right()
            if x1-x0 <= 1:
                return
        # make edges rigid
        xl = max(_round(x0-side_margin)+side_margin, -side_margin)
        xr = min(_round(x1-side_margin)+side_margin, datasrc.xlen+right_margin_candles-side_margin)
        dxl = xl-x0
        dxr = xr-x1
        if dxl > 0:
            x1 += dxl
        if dxr < 0:
            x0 += dxr
        x0 = max(_round(x0-side_margin)+side_margin, -side_margin)
        x1 = min(_round(x1-side_margin)+side_margin, datasrc.xlen+right_margin_candles-side_margin)
        # fetch hi-lo and set range
        _,_,hi,lo,cnt = datasrc.hilo(x0, x1)
        vr = self.viewRect()
        minlen = int((max_zoom_points-0.5) * self.max_zoom_points_f + 0.51)
        if (x1-x0) < vr.width() and cnt < minlen:
            return
        if not self.v_autozoom:
            hi = vr.bottom()
            lo = vr.top()
        if self.yscale.scaletype == 'log':
            if lo < 0:
                lo = 0.05 * self.yscale.scalef # strange QT log scale rendering, which I'm unable to compensate for
            else:
                lo = max(1e-100, lo)
            rng = (hi / lo) ** (1/self.v_zoom_scale)
            rng = min(rng, 1e200) # avoid float overflow
            base = (hi*lo) ** self.v_zoom_baseline
            y0 = base / rng**self.v_zoom_baseline
            y1 = base * rng**(1-self.v_zoom_baseline)
        else:
            rng = (hi-lo) / self.v_zoom_scale
            rng = max(rng, 2e-7) # some very weird bug where high/low exponents stops rendering
            base = (hi+lo) * self.v_zoom_baseline
            y0 = base - rng*self.v_zoom_baseline
            y1 = base + rng*(1-self.v_zoom_baseline)
        if not self.x_indexed:
            x0,x1 = _xminmax(datasrc, x_indexed=False, extra_margin=0)
        return self.set_range(x0, y0, x1, y1)

    def set_range(self, x0, y0, x1, y1):
        if x0 is None or x1 is None:
            tr = self.targetRect()
            x0 = tr.left()
            x1 = tr.right()
        if np.isnan(y0) or np.isnan(y1):
            return
        _y0 = self.yscale.invxform(y0, verify=True)
        _y1 = self.yscale.invxform(y1, verify=True)
        self.setRange(QtCore.QRectF(pg.Point(x0, _y0), pg.Point(x1, _y1)), padding=0)
        self.zoom_changed()
        return True

    def remove_last_roi(self):
        if self.rois:
            if not isinstance(self.rois[-1], pg.PolyLineROI):
                self.removeItem(self.rois[-1])
                self.rois = self.rois[:-1]
            else:
                h = self.rois[-1].handles[-1]['item']
                self.rois[-1].removeHandle(h)
                if not self.rois[-1].segments:
                    self.removeItem(self.rois[-1])
                    self.rois = self.rois[:-1]
                    self.draw_line = None
            if self.rois:
                if isinstance(self.rois[-1], pg.PolyLineROI):
                    self.draw_line = self.rois[-1]
                    self.set_draw_line_color(draw_line_color)
            return True

    def append_draw_segment(self, p):
        h0 = self.draw_line.handles[-1]['item']
        h1 = self.draw_line.addFreeHandle(p)
        self.draw_line.addSegment(h0, h1)
        self.drawing = True

    def set_draw_line_color(self, color):
        if self.draw_line:
            pen = pg.mkPen(color)
            for segment in self.draw_line.segments:
                segment.currentPen = segment.pen = pen
                segment.update()

    def suggestPadding(self, axis):
        return 0

    def zoom_changed(self):
        for zl in self.zoom_listeners:
            zl(self)
