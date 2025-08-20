# -*- coding: utf-8 -*-
'''
Financial data plotter with better defaults, api, behavior and performance than
mpl_finance and plotly.

Lines up your time-series with a shared X-axis; ideal for volume, RSI, etc.

Zoom does something similar to what you'd normally expect for financial data,
where the Y-axis is auto-scaled to highest high and lowest low in the active
region.
'''

from finplot.constants import *  # TODO
from finplot.item import (
    EpochAxisItem,
    YAxisItem,
    FinLegendItem,
    FinPlotItem,
    CandlestickItem,
    HeatmapItem,
    HorizontalTimeVolumeItem,
    ScatterLabelItem,
)
from finplot.live import Live
from finplot.other import *
from finplot.yscale import YScale
from finplot.utils import _wheel_event_wrapper
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import QtCore


try:
    qtver = '%d.%d' % (QtCore.QT_VERSION//256//256, QtCore.QT_VERSION//256%256)
    if qtver not in ('5.9', '5.13') and [int(i) for i in pg.__version__.split('.')] <= [0,11,0]:
        print('WARNING: your version of Qt may not plot curves containing NaNs and is not recommended.')
        print('See https://github.com/pyqtgraph/pyqtgraph/issues/1057')
except:
    pass

# default to black-on-white
pg.widgets.GraphicsView.GraphicsView.wheelEvent = partialmethod(_wheel_event_wrapper, pg.widgets.GraphicsView.GraphicsView.wheelEvent)
# use finplot instead of matplotlib
pd.set_option('plotting.backend', 'finplot.pdplot')
# pick up win resolution
try:
    import ctypes
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    lod_candles = int(user32.GetSystemMetrics(0) * 1.6)
    candle_shadow_width = int(user32.GetSystemMetrics(0) // 2100 + 1) # 2560 and resolutions above -> wider shadows
except:
    pass
