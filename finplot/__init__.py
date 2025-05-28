# -*- coding: utf-8 -*-
"""
Financial data plotter with better defaults, api, behavior and performance than
mpl_finance and plotly.

Lines up your time-series with a shared X-axis; ideal for volume, RSI, etc.

Zoom does something similar to what you'd normally expect for financial data,
where the Y-axis is auto-scaled to highest high and lowest low in the active
region.
"""


__all__ = (
    'add_horizontal_band',
    'add_rect',
    'candlestick_ochl',
    'create_plot',
    'live',
    'plot',
    'refresh',
    'remove_primitive',
    'show',
    'volume_ocv'
)


# from decimal import Decimal
from functools import partialmethod

import pandas as pd
import pyqtgraph as pg

from pyqtgraph import (
    QtCore
)

from ._version import __version__

from .finplot_utils import (
    add_horizontal_band,
    add_rect,
    candlestick_ochl,
    create_plot,
    fill_between,
    live,
    plot,
    refresh,
    remove_primitive,
    set_y_range,
    show,
    volume_ocv
)

from .internal_utils import (
    _wheel_event_wrapper
)

from .live import Live


try:
    qt_version = '%d.%d' % (QtCore.QT_VERSION // 256 // 256, QtCore.QT_VERSION // 256 % 256)
    if qt_version not in ('5.9', '5.13') and [int(i) for i in pg.__version__.split('.')] <= [0, 11, 0]:
        print('WARNING: your version of Qt may not plot curves containing NaNs and is not recommended.')
        print('See https://github.com/pyqtgraph/pyqtgraph/issues/1057')
except:
    pass


# default to black-on-white
pg.widgets.GraphicsView.GraphicsView.wheelEvent = (
    partialmethod(
        _wheel_event_wrapper,
        pg.widgets.GraphicsView.GraphicsView.wheelEvent
    )
)

# use finplot instead of matplotlib
pd.set_option(
    'plotting.backend',
    'finplot.pdplot'
)
