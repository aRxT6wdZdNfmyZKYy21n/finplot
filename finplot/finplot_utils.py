from functools import (
    partial
)

import numpy as np
import pandas as pd
import pyqtgraph as pg

from pyqtgraph import (
    QtCore,
    QtGui
)

from .constants import (
    FinPlotConstants
)

from .fin_cross_hair import (
    FinCrossHair
)

from .fin_line import (
    FinLine
)

from .fin_rect import (
    FinRect
)

from .fin_view_box import (
    FinViewBox
)

from .fin_window import (
    FinWindow
)

from .fin_poly_line import (
    FinPolyLine
)

from .global_state import (
    g_fin_plot_global_state
)

from .internal_utils import (
    brighten,
    _adjust_bar_datasrc,
    _adjust_horiz_datasrc,
    _adjust_renko_datasrc,
    _adjust_renko_log_datasrc,
    _adjust_volume_datasrc,
    _ax_decouple,
    _ax_disable_x_index,
    _ax_expand,
    _ax_invert_y,
    _ax_set_visible,
    _create_legend,
    _create_series,
    _get_color,
    _loadwindata,
    _makepen,
    _mcallback_click,
    _mcallback_pos,
    _mouse_moved,
    _paint_scatter,
    _pdtime2index,
    _preadjust_horiz_datasrc,
    _repaint_candles,
    _set_datasrc,
    _set_max_zoom,
    _update_gfx,
    _update_significants
)

from .item.axis.epoch import (
    EpochAxisItem
)

from .item.axis.y import (
    YAxisItem
)

from .item.candlestick import (
    CandlestickItem
)

from .item.heatmap import (
    HeatmapItem
)

from .item.horizontal_time_volume import (
    HorizontalTimeVolumeItem
)

from .item.scatter_label import (
    ScatterLabelItem
)

from .live import Live

from .pandas_data_source import (
    PandasDataSource
)

from .y_scale import (
    YScale
)


def _add_timestamp_plot(master, prev_ax, viewbox, index, yscale):
    native_win = isinstance(master, pg.GraphicsLayoutWidget)
    if native_win and prev_ax is not None:
        prev_ax.set_visible(xaxis=False)  # hide the whole previous axis
    axes = {'bottom': _create_axis(pos='x', vb=viewbox, orientation='bottom'),
            'right': _create_axis(pos='y', vb=viewbox, orientation='right')}
    if native_win:
        ax = pg.PlotItem(viewBox=viewbox, axisItems=axes, name='plot-%i' % index, enableMenu=False)
    else:
        axw = pg.PlotWidget(viewBox=viewbox, axisItems=axes, name='plot-%i' % index, enableMenu=False)
        ax = axw.plotItem
        ax.ax_widget = axw
    ax.setClipToView(True)
    ax.setDownsampling(auto=True, mode='subsample')
    ax.hideAxis('left')

    if FinPlotConstants.y_label_width:
        ax.axes['right']['item'].setWidth(  # this is to put all graphs on equal footing when texts vary from 0.4 to 2000000
            FinPlotConstants.y_label_width
        )

    ax.axes['right']['item'].setStyle(
        tickLength=-5)  # some bug, totally unexplicable (why setting the default value again would fix repaint width as axis scale down)
    ax.axes['right']['item'].setZValue(30)  # put axis in front instead of behind data
    ax.axes['bottom']['item'].setZValue(30)
    ax.setLogMode(y=(yscale.scaletype == 'log'))
    ax.significant_forced = False

    ax.significant_decimals = (
        FinPlotConstants.significant_decimals
    )

    ax.significant_eps = (
        FinPlotConstants.significant_eps
    )

    ax.inverted = False
    ax.axos = []

    ax.crosshair = (
        FinCrossHair(
            ax,
            color=FinPlotConstants.cross_hair_color
        )
    )

    ax.hideButtons()
    ax.overlay = partial(_ax_overlay, ax)
    ax.set_visible = partial(_ax_set_visible, ax)
    ax.decouple = partial(_ax_decouple, ax)
    ax.disable_x_index = partial(_ax_disable_x_index, ax)
    ax.reset = partial(_ax_reset, ax)
    ax.invert_y = partial(_ax_invert_y, ax)
    ax.expand = partial(_ax_expand, ax)
    ax.prev_ax = prev_ax
    ax.win_index = index

    if index % 2:
        viewbox.setBackgroundColor(
            FinPlotConstants.odd_plot_background
        )

    viewbox.setParent(ax)
    return ax


def _ax_overlay(ax, scale=0.25, yaxis=False):
    """The scale parameter defines how "high up" on the initial plot this overlay will show.
       The yaxis parameter can be one of [False, 'linear', 'log']."""
    yscale = yaxis if yaxis else 'linear'
    viewbox = FinViewBox(ax.vb.win, init_steps=ax.vb.init_steps, yscale=YScale(yscale, 1), enableMenu=False)
    viewbox.master_viewbox = ax.vb
    viewbox.setZValue(-5)
    viewbox.setBackgroundColor(ax.vb.state['background'])
    ax.vb.setBackgroundColor(None)
    viewbox.v_zoom_scale = scale
    if hasattr(ax, 'ax_widget'):
        ax.ax_widget.scene().addItem(viewbox)
    else:
        ax.vb.win.centralWidget.scene().addItem(viewbox)
    viewbox.setXLink(ax.vb)

    def updateView():
        viewbox.setGeometry(
            ax.vb.sceneBoundingRect()
        )

    axo = pg.PlotItem(enableMenu=False)
    axo.significant_forced = False

    axo.significant_decimals = (
        FinPlotConstants.significant_decimals
    )

    axo.significant_eps = (
        FinPlotConstants.significant_eps
    )

    axo.prev_ax = None
    axo.crosshair = None
    axo.decouple = partial(_ax_decouple, axo)
    axo.disable_x_index = partial(_ax_disable_x_index, axo)
    axo.reset = partial(_ax_reset, axo)
    axo.hideAxis('left')
    axo.hideAxis('right')
    axo.hideAxis('bottom')
    axo.hideButtons()
    viewbox.addItem(axo)
    axo.vb = viewbox

    if yaxis and isinstance(axo.vb.win, pg.GraphicsLayoutWidget):
        axi = _create_axis(pos='y', vb=axo.vb, orientation='left')
        axo.setAxisItems({'left': axi})
        axo.vb.win.addItem(axi, row=0, col=0)

    ax.vb.sigResized.connect(
        updateView
    )

    g_fin_plot_global_state.add_overlay_axo(
        axo
    )

    ax.axos.append(
        axo
    )

    updateView()

    return axo


def _ax_reset(ax):
    if ax.crosshair is not None:
        ax.crosshair.hide()
    for item in list(ax.items):
        if any(isinstance(item, c) for c in [FinLine, FinPolyLine, pg.TextItem]):
            try:
                remove_primitive(item)
            except:
                pass
        else:
            ax.removeItem(item)
        if ax.vb.master_viewbox and hasattr(item, 'name') and item.name():
            legend = ax.vb.master_viewbox.parent().legend
            if legend:
                legend.removeItem(item)
    if ax.legend:
        ax.legend.opts['offset'] = None
        ax.legend.setParentItem(None)
        ax.legend = None
    ax.vb.reset()
    ax.vb.set_datasrc(None)
    if ax.crosshair is not None:
        ax.crosshair.show()
    ax.significant_forced = False


def _create_axis(pos, **kwargs):
    if pos == 'x':
        return (
            EpochAxisItem(
                **kwargs
            )
        )
    elif pos == 'y':
        return (
            YAxisItem(
                **kwargs
            )
        )


def _create_datasrc(ax, *args, ncols=-1, allow_scaling=True):
    def do_create(args):
        if len(args) == 1 and type(args[0]) == PandasDataSource:
            return args[0]
        if len(args) == 1 and type(args[0]) in (list, tuple):
            args = [np.array(args[0])]
        if len(args) == 1 and type(args[0]) == np.ndarray:
            args = [pd.DataFrame(args[0].T)]
        if len(args) == 1 and type(args[0]) == pd.DataFrame:
            return PandasDataSource(args[0])
        args = [_create_series(a) for a in args]
        return PandasDataSource(pd.concat(args, axis=1))

    iargs = [a for a in args if a is not None]
    datasrc = do_create(iargs)
    # check if time column missing
    if len(datasrc.df.columns) in (1, ncols - 1):
        # assume time data has already been added before
        for a in ax.vb.win.axs:
            if a.vb.datasrc and len(a.vb.datasrc.df.columns) >= 2:
                col = a.vb.datasrc.df.columns[0]
                if col in datasrc.df.columns:
                    # ensure column names are unique
                    datasrc.df.columns = a.vb.datasrc.df.columns[1:len(datasrc.df.columns) + 1]
                col = a.vb.datasrc.df.columns[0]
                datasrc.df.insert(0, col, a.vb.datasrc.df[col])
                datasrc = PandasDataSource(datasrc.df)
                break
        if len(datasrc.df.columns) in (1, ncols - 1):
            if ncols > 1:
                print(f"WARNING: this type of plot wants %i columns/args, but you've only supplied %i" % (
                ncols, len(datasrc.df.columns)))
                print(' - Assuming time column is missing and using index instead.')
            datasrc = PandasDataSource(datasrc.df.reset_index())
    elif len(iargs) >= 2 and len(datasrc.df.columns) == len(iargs) + 1 and len(iargs) == len(args):
        try:
            if '.Int' in str(type(iargs[0].index)):
                print('WARNING: performance penalty and crash may occur when using int64 instead of range indices.')
                if (iargs[0].index == range(len(iargs[0]))).all():
                    print(' - Fix by .reset_index(drop=True)')
                    return _create_datasrc(ax, datasrc.df[datasrc.df.columns[1:]], ncols=ncols)
        except:
            print('WARNING: input data source may cause performance penalty and crash.')

    assert len(datasrc.df.columns) >= ncols, 'ERROR: too few columns/args supplied for this plot'

    if datasrc.period_ns < 0:
        print('WARNING: input data source has time in descending order. Try sort_values() before calling.')

    # FIX: stupid QT bug causes rectangles larger than 2G to flicker, so scale rendering down some
    # FIX: PyQt 5.15.2 lines >1e6 are being clipped to 1e6 during the first render pass, so scale down if >1e6
    if ax.vb.yscale.scalef == 1 and allow_scaling and datasrc.df.iloc[:, 1:].max(numeric_only=True).max() > 1e6:
        ax.vb.yscale.set_scale(int(1e6))
    return datasrc


def _create_plot(
        ax=None,
        **kwargs
):
    if ax:
        return ax

    last_ax = (
        g_fin_plot_global_state.get_last_ax()
    )

    if last_ax:
        return last_ax

    return create_plot(
        **kwargs
    )


def _improve_significants(ax):
    """Force update of the EPS if we both have no bars/candles AND a log scale.
       This is intended to fix the lower part of the grid on line plots on a log scale."""
    if ax.vb.yscale.scaletype == 'log':
        if not (
                any(
                    isinstance(
                        item,
                        CandlestickItem
                    )

                    for item in (
                        ax.items
                    )
                )
        ):
            _update_significants(
                ax,
                ax.vb.datasrc,
                force=True
            )


def _is_internal_windows_only() -> bool:
    return (
        all(
            isinstance(
                win,
                FinWindow
            )

            for win in (
                g_fin_plot_global_state.get_windows()
            )
        )
    )


def _update_data(preadjustfunc, adjustfunc, item, ds, gfx=True):
    if preadjustfunc:
        ds = preadjustfunc(ds)
    ds = _create_datasrc(item.ax, ds)
    if adjustfunc:
        adjustfunc(ds)
    cs = list(item.datasrc.df.columns[:1]) + list(item.datasrc.df.columns[item.datasrc.col_data_offset:])
    if len(cs) >= len(ds.df.columns):
        ds.df.columns = cs[:len(ds.df.columns)]
    item.datasrc.update(ds)
    _set_datasrc(item.ax, item.datasrc, addcols=False)
    if gfx:
        item.update_gfx()


def create_plot(
        title='Finance Plot',
        rows=1,
        init_zoom_periods=1e10,
        maximize=True,
        yscale='linear'
):
    pg.setConfigOptions(
        foreground=(
            FinPlotConstants.foreground
        ),

        background=(
            FinPlotConstants.background
        )
    )

    win = (
        FinWindow(
            title
        )
    )

    win.show_maximized = maximize
    ax0 = axs = create_plot_widget(master=win, rows=rows, init_zoom_periods=init_zoom_periods, yscale=yscale)
    axs = axs if type(axs) in (tuple, list) else [axs]

    for ax in axs:
        win.addItem(ax, col=1)
        win.nextRow()

    return ax0


def create_plot_widget(master, rows=1, init_zoom_periods=1e10, yscale='linear'):
    pg.setConfigOptions(
        foreground=(
            FinPlotConstants.foreground
        ),

        background=(
            FinPlotConstants.background
        )
    )

    if (
            master not in
            g_fin_plot_global_state.get_windows()
    ):
        g_fin_plot_global_state.add_window(
            master
        )

    axs = []
    prev_ax = None

    for n in range(rows):
        ysc = yscale[n] if type(yscale) in (list, tuple) else yscale
        ysc = YScale(ysc, 1) if type(ysc) == str else ysc

        viewbox = (
            FinViewBox(
                master,

                init_steps=(
                    init_zoom_periods
                ),

                yscale=ysc,

                v_zoom_scale=(
                    1 -
                    FinPlotConstants.y_pad
                ),

                enableMenu=False
            )
        )

        ax = (
            prev_ax
        ) = (
            _add_timestamp_plot(
                master=master,
                prev_ax=prev_ax,
                viewbox=viewbox,
                index=n,
                yscale=ysc
            )
        )

        if axs:
            ax.setXLink(
                axs[0].vb
            )
        else:
            viewbox.setFocus()

        axs += [ax]

    window_master_data = (
        g_fin_plot_global_state.get_window_master_data(
            master
        )
    )

    if (
            isinstance(
                master,
                pg.GraphicsLayoutWidget
            )
    ):
        first_axs_vb = (
            axs[0].vb
        )

        proxy = (
            pg.SignalProxy(
                master.scene().sigMouseMoved,
                rateLimit=144,

                slot=(
                    partial(
                        _mouse_moved,
                        master,
                        first_axs_vb
                    )
                )
            )
        )

        (
            window_master_data[
                first_axs_vb
            ]
        ) = (
            dict(
                proxymm=proxy,
                last_mouse_evs=None,
                last_mouse_y=0
            )
        )

        if (
                'default' not in
                window_master_data
        ):
            (
                window_master_data[
                    'default'
                ]
            ) = (
                window_master_data[
                    first_axs_vb
                ]
            )
    else:
        for ax in axs:
            ax_vb = (
                ax.vb
            )

            proxy = (
                pg.SignalProxy(
                    ax.ax_widget.scene().sigMouseMoved,
                    rateLimit=144,

                    slot=partial(
                        _mouse_moved,
                        master,
                        ax_vb
                    )
                )
            )

            (
                window_master_data[
                    ax_vb
                ]
            ) = (
                dict(
                    proxymm=proxy,
                    last_mouse_evs=None,
                    last_mouse_y=0
                )
            )

    g_fin_plot_global_state.set_window_master_data(
        master,
        window_master_data
    )

    g_fin_plot_global_state.set_last_ax(
        axs[
            0
        ]
    )

    return (
        axs[0]
        if len(axs) == 1
        else axs
    )


def close():
    for window in (
            g_fin_plot_global_state.get_windows()
    ):
        try:
            window.close()
        except Exception as e:
            print(
                'Window closing error:',
                type(e),
                e
            )

    g_fin_plot_global_state.clear_windows()
    g_fin_plot_global_state.clear_overlay_axs()
    g_fin_plot_global_state.clear_timers()
    g_fin_plot_global_state.clear_sound_effect_by_filename_map()
    g_fin_plot_global_state.clear_master_data_by_window_map()

    g_fin_plot_global_state.set_last_ax(
        None
    )


def price_colorfilter(item, datasrc, df):
    opencol = df.columns[1]
    closecol = df.columns[2]
    is_up = df[opencol] <= df[closecol] # open lower than close = goes up
    yield item.rowcolors('bull') + [df.loc[is_up, :]]
    yield item.rowcolors('bear') + [df.loc[~is_up, :]]


def volume_colorfilter(item, datasrc, df):
    opencol = df.columns[3]
    closecol = df.columns[4]
    is_up = df[opencol] <= df[closecol] # open lower than close = goes up
    yield item.rowcolors('bull') + [df.loc[is_up, :]]
    yield item.rowcolors('bear') + [df.loc[~is_up, :]]


def strength_colorfilter(item, datasrc, df):
    opencol = df.columns[1]
    closecol = df.columns[2]
    startcol = df.columns[3]
    endcol = df.columns[4]
    is_up = df[opencol] <= df[closecol] # open lower than close = goes up
    is_strong = df[startcol] <= df[endcol]
    yield item.rowcolors('bull') + [df.loc[is_up&is_strong, :]]
    yield item.rowcolors('weak_bull') + [df.loc[is_up&(~is_strong), :]]
    yield item.rowcolors('weak_bear') + [df.loc[(~is_up)&is_strong, :]]
    yield item.rowcolors('bear') + [df.loc[(~is_up)&(~is_strong), :]]


def volume_colorfilter_section(sections=[]):
    """The sections argument is a (starting_index, color_name) array."""
    def _colorfilter(sections, item, datasrc, df):
        if not sections:
            return volume_colorfilter(item, datasrc, df)
        for (i0,colname),(i1,_) in zip(sections, sections[1:]+[(None,'neutral')]):
            rows = df.iloc[i0:i1, :]
            yield item.rowcolors(colname) + [rows]
    return partial(_colorfilter, sections)


def candlestick_ochl(datasrc, draw_body=True, draw_shadow=True, candle_width=0.6, ax=None, colorfunc=price_colorfilter):
    ax = _create_plot(ax=ax, maximize=False)
    datasrc = _create_datasrc(ax, datasrc, ncols=5)
    datasrc.scale_cols = [3, 4]  # only hi+lo scales
    _set_datasrc(ax, datasrc)

    item = (
        CandlestickItem(
            ax=ax,
            datasrc=datasrc,
            draw_body=draw_body,
            draw_shadow=draw_shadow,
            candle_width=candle_width,
            colorfunc=colorfunc,
            resamp='hilo'
        )
    )

    _update_significants(ax, datasrc, force=True)
    item.update_data = partial(_update_data, None, None, item)
    item.update_gfx = partial(_update_gfx, item)
    ax.addItem(item)
    return item


def renko(x, y=None, bins=None, step=None, ax=None, colorfunc=price_colorfilter):
    ax = _create_plot(ax=ax, maximize=False)
    datasrc = _create_datasrc(ax, x, y, ncols=3)
    origdf = datasrc.df
    adj = _adjust_renko_log_datasrc if ax.vb.yscale.scaletype == 'log' else _adjust_renko_datasrc
    step_adjust_renko_datasrc = partial(adj, bins, step)
    step_adjust_renko_datasrc(datasrc)
    ax.decouple()

    item = (
        candlestick_ochl(
            datasrc,
            draw_shadow=False,
            candle_width=1,
            ax=ax,
            colorfunc=colorfunc
        )
    )

    (
        item.colors[
            'bull_body'
        ]
    ) = (
        item.colors[
            'bull_frame'
        ]
    )

    item.update_data = (
        partial(
            _update_data,
            None,
            step_adjust_renko_datasrc,
            item
        )
    )

    item.update_gfx = (
        partial(
            _update_gfx,
            item
        )
    )

    g_fin_plot_global_state.set_epoch_period(
        (
            origdf.iloc[1, 0] -
            origdf.iloc[0, 0]
        ) //

        (
            10 **
            9
        )
    )

    return item


def volume_ocv(datasrc, candle_width=0.8, ax=None, colorfunc=volume_colorfilter):
    ax = _create_plot(ax=ax, maximize=False)
    datasrc = _create_datasrc(ax, datasrc, ncols=4)
    _adjust_volume_datasrc(datasrc)
    _set_datasrc(ax, datasrc)
    item = CandlestickItem(ax=ax, datasrc=datasrc, draw_body=True, draw_shadow=False, candle_width=candle_width, colorfunc=colorfunc, resamp='sum')

    _update_significants(
        ax,
        datasrc,
        force=True
    )

    item.colors['bull_body'] = (
        item.colors['bull_frame']
    )

    if (
            colorfunc ==
            volume_colorfilter
    ):
        # assume normal volume plot

        item.colors.update({
            'bear_frame': (
                FinPlotConstants.volume_bear_color
            ),

            'bear_body': (
                FinPlotConstants.volume_bear_color
            ),

            'bull_body': (
                FinPlotConstants.volume_bull_body_color
            ),

            'bull_frame': (
                FinPlotConstants.volume_bull_color
            ),
        })

        ax.vb.v_zoom_baseline = 0
    else:
        item.colors.update({
            'weak_bull_frame': (
                brighten(
                    FinPlotConstants.volume_bull_color,
                    1.2
                )
            ),

            'weak_bull_body': (
                brighten(
                    FinPlotConstants.volume_bull_color,
                    1.2
                )
            )
        })

    item.update_data = (
        partial(
            _update_data,
            None,
            _adjust_volume_datasrc,
            item
        )
    )

    item.update_gfx = (
        partial(
            _update_gfx,
            item
        )
    )

    ax.addItem(
        item
    )

    item.setZValue(
        -20
    )

    return item


def horiz_time_volume(datasrc, ax=None, **kwargs):
    """Draws multiple fixed horizontal volumes. The input format is:
       [[time0, [(price0,volume0),(price1,volume1),...]], ...]

       This chart needs to be plot last, so it knows if it controls
       what time periods are shown, or if its using time already in
       place by another plot."""
    # update handling default if necessary

    max_zoom_points = (
        g_fin_plot_global_state.get_max_zoom_points()
    )

    if max_zoom_points > 15:
        max_zoom_points = 4

        g_fin_plot_global_state.set_max_zoom_points(
            max_zoom_points
        )

    right_margin_candles = (
        g_fin_plot_global_state.get_right_margin_candles()
    )

    if right_margin_candles > 3:
        right_margin_candles = 1

        g_fin_plot_global_state.set_right_margin_candles(
            right_margin_candles
        )

    ax = (
        _create_plot(
            ax=ax,
            maximize=False
        )
    )

    datasrc = _preadjust_horiz_datasrc(datasrc)
    datasrc = _create_datasrc(ax, datasrc, allow_scaling=False)
    _adjust_horiz_datasrc(datasrc)
    if ax.vb.datasrc is not None:
        datasrc.standalone = True # only force standalone if there is something on our charts already
    datasrc.scale_cols = [datasrc.col_data_offset, len(datasrc.df.columns)-2] # first and last price columns
    datasrc.pre_update = lambda df: df.loc[:, :df.columns[0]] # throw away previous data
    datasrc.post_update = lambda df: df.dropna(how='all') # kill all-NaNs
    _set_datasrc(ax, datasrc)
    item = HorizontalTimeVolumeItem(ax=ax, datasrc=datasrc, **kwargs)
    item.update_data = partial(_update_data, _preadjust_horiz_datasrc, _adjust_horiz_datasrc, item)
    item.update_gfx = partial(_update_gfx, item)
    item.setZValue(-10)
    ax.addItem(item)

    return item


def heatmap(datasrc, ax=None, **kwargs):
    """Expensive function. Only use on small data sets. See HeatmapItem for kwargs. Input datasrc
       has x (time) in index or first column, y (price) as column names, and intensity (color) as
       cell values."""
    ax = _create_plot(ax=ax, maximize=False)
    if ax.vb.v_zoom_scale >= 0.9:
        ax.vb.v_zoom_scale = 0.6
    datasrc = _create_datasrc(ax, datasrc)
    datasrc.scale_cols = [] # doesn't scale
    _set_datasrc(ax, datasrc)
    item = HeatmapItem(ax=ax, datasrc=datasrc, **kwargs)
    item.update_data = partial(_update_data, None, None, item)
    item.update_gfx = partial(_update_gfx, item)
    item.setZValue(-30)
    ax.addItem(item)
    if ax.vb.datasrc is not None and not ax.vb.datasrc.timebased(): # manual zoom update
        ax.setXLink(None)
        if ax.prev_ax:
            ax.prev_ax.set_visible(xaxis=True)
        df = ax.vb.datasrc.df
        prices = df.columns[ax.vb.datasrc.col_data_offset:item.col_data_end]
        delta_price = abs(prices[0] - prices[1])
        ax.vb.set_range(0, min(df.columns[1:]), len(df), max(df.columns[1:])+delta_price)
    return item


def bar(
        x,
        y=None,
        width=0.8,
        ax=None,
        colorfunc=strength_colorfilter,
        **kwargs
):
    """Bar plots are decoupled. Use volume_ocv() if you want a bar plot which relates to other time plots."""

    g_fin_plot_global_state.set_right_margin_candles(
        0
    )

    max_max_zoom_points = 8

    max_zoom_points = (
        g_fin_plot_global_state.get_max_zoom_points()
    )

    if (
            max_zoom_points >
            max_max_zoom_points
    ):
        max_zoom_points = (
            max_max_zoom_points
        )

        g_fin_plot_global_state.set_max_zoom_points(
            max_zoom_points
        )

    ax = (
        _create_plot(
            ax=ax,
            maximize=False
        )
    )

    ax.decouple()
    datasrc = _create_datasrc(ax, x, y, ncols=1)
    _adjust_bar_datasrc(datasrc, order_cols=False) # don't rearrange columns, done for us in volume_ocv()
    item = volume_ocv(datasrc, candle_width=width, ax=ax, colorfunc=colorfunc)
    item.update_data = partial(_update_data, None, _adjust_bar_datasrc, item)
    item.update_gfx = partial(_update_gfx, item)
    ax.vb.pre_process_data()
    if ax.vb.y_min >= 0:
        ax.vb.v_zoom_baseline = 0

    return item


def hist(x, bins, ax=None, **kwargs):
    hist_data = pd.cut(x, bins=bins).value_counts()
    data = [(i.mid,0,hist_data.loc[i],hist_data.loc[i]) for i in sorted(hist_data.index)]
    df = pd.DataFrame(data, columns=['x','_op_','_cl_','bin'])
    df.set_index('x', inplace=True)
    item = bar(df, ax=ax)
    del item.update_data
    return item


def plot(x, y=None, color=None, width=1, ax=None, style=None, legend=None, zoomscale=True, **kwargs):
    ax = _create_plot(ax=ax, maximize=False)
    used_color = _get_color(ax, style, color)
    datasrc = _create_datasrc(ax, x, y, ncols=1)
    if not zoomscale:
        datasrc.scale_cols = []
    _set_datasrc(ax, datasrc)
    if legend is not None:
        _create_legend(ax)
    x = datasrc.x if not ax.vb.x_indexed else datasrc.index
    y = datasrc.y / ax.vb.yscale.scalef

    if ax.vb.yscale.scaletype == 'log':
        y += FinPlotConstants.log_plot_offset

    if style is None or any(ch in style for ch in '-_.'):
        connect_dots = 'finite' # same as matplotlib; use datasrc.standalone=True if you want to keep separate intervals on a plot
        item = ax.plot(x, y, pen=_makepen(color=used_color, style=style, width=width), name=legend, connect=connect_dots)
        item.setZValue(5)
    else:
        symbol = {'v':'t', '^':'t1', '>':'t2', '<':'t3'}.get(style, style) # translate some similar styles
        yfilter = y.notnull()
        ser = y.loc[yfilter]
        x = x.loc[yfilter].values if hasattr(x, 'loc') else x[yfilter]
        item = ax.plot(x, ser.values, pen=None, symbol=symbol, symbolPen=None, symbolSize=7*width, symbolBrush=pg.mkBrush(used_color), name=legend)
        if width < 1:
            item.opts['antialias'] = True
        item.scatter._dopaint = item.scatter.paint
        item.scatter.paint = partial(_paint_scatter, item.scatter)
        # optimize (when having large number of points) by ignoring scatter click detection
        _dummy_mouse_click = lambda ev: 0
        item.scatter.mouseClickEvent = _dummy_mouse_click
        item.setZValue(10)
    item.opts['handed_color'] = color
    item.ax = ax
    item.datasrc = datasrc
    _update_significants(ax, datasrc, force=False)
    item.update_data = partial(_update_data, None, None, item)
    item.update_gfx = partial(_update_gfx, item)
    # add legend to main ax, not to overlay
    axm = ax.vb.master_viewbox.parent() if ax.vb.master_viewbox else ax
    if axm.legend is not None:
        if legend and axm != ax:
            axm.legend.addItem(
                item,
                name=legend
            )

        for _, label in axm.legend.items:
            if label.text == legend:
                label.setAttr(
                    'justify',
                    'left'
                )

                label.setText(
                    label.text,
                    color=FinPlotConstants.legend_text_color
                )
    return item


def labels(x, y=None, labels=None, color=None, ax=None, anchor=(0.5,1)):
    ax = _create_plot(ax=ax, maximize=False)
    used_color = _get_color(ax, '?', color)
    datasrc = _create_datasrc(ax, x, y, labels, ncols=3)
    datasrc.scale_cols = [] # don't use this for scaling
    _set_datasrc(ax, datasrc)
    item = ScatterLabelItem(ax=ax, datasrc=datasrc, color=used_color, anchor=anchor)
    _update_significants(ax, datasrc, force=False)
    item.update_data = partial(_update_data, None, None, item)
    item.update_gfx = partial(_update_gfx, item)
    ax.addItem(item)
    if ax.vb.v_zoom_scale > 0.9: # adjust to make hi/lo text fit
        ax.vb.v_zoom_scale = 0.9
    return item


def live(plots=1):
    if plots == 1:
        return Live()

    return [
        Live()

        for _ in (
            range(
                plots
            )
        )
    ]


def add_legend(text, ax=None):
    ax = (
        _create_plot(
            ax=ax,
            maximize=False
        )
    )

    _create_legend(
        ax
    )

    row = ax.legend.layout.rowCount()

    label = (
        pg.LabelItem(
            text,
            color=FinPlotConstants.legend_text_color,
            justify='left'
        )
    )

    ax.legend.layout.addItem(
        label,
        row,
        0,
        1,
        2
    )

    return label


def fill_between(
        plot0,
        plot1,
        color=None
) -> (
        pg.FillBetweenItem
):
    used_color = (
        brighten(
            _get_color(
                plot0.ax,
                None,
                color
            ),

            1.3
        )
    )

    item = (
        pg.FillBetweenItem(
            plot0,
            plot1,

            brush=(
                pg.mkBrush(
                    used_color
                )
            )
        )
    )

    item.ax = plot0.ax
    item.setZValue(-40)
    item.ax.addItem(item)

    # Ugly bug fix for PyQtGraph bug where downsampled/clipped plots are used in conjunction
    # with fill between. The reason is that the curves of the downsampled plots are only
    # calculated when shown, but not when added to the axis. We fix by saying the plot is
    # changed every time the zoom is changed - including initial load.

    def update_fill(vb):
        plot0.sigPlotChanged.emit(plot0)

    plot0.ax.vb.zoom_listeners.add(
        update_fill
    )

    return item


def set_x_pos(xmin, xmax, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    xidx0,xidx1 = _pdtime2index(ax, pd.Series([xmin, xmax]))
    ax.vb.update_y_zoom(xidx0, xidx1)
    _repaint_candles()


def set_y_range(ymin, ymax, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    ax.setLimits(yMin=ymin, yMax=ymax)
    ax.vb.v_autozoom = False
    ax.vb.set_range(None, ymin, None, ymax)


def set_y_scale(yscale='linear', ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    ax.setLogMode(y=(yscale=='log'))
    ax.vb.yscale = YScale(yscale, ax.vb.yscale.scalef)


def add_band(y0, y1, color=FinPlotConstants.band_color, ax=None):
    print('add_band() is deprecated, use add_horizontal_band() instead.')
    return add_horizontal_band(y0, y1, color, ax)


def add_horizontal_band(
        y0,
        y1,
        color=FinPlotConstants.band_color,
        ax=None
) -> (
        pg.LinearRegionItem
):
    ax = (
        _create_plot(
            ax=ax,
            maximize=False
        )
    )

    color = (
        _get_color(
            ax,
            None,
            color
        )
    )

    ix = ax.vb.yscale.invxform

    linear_region_item = (
        pg.LinearRegionItem(
            [
                ix(y0),
                ix(y1)
            ],

            orientation=(
                pg.LinearRegionItem.Horizontal
            ),

            brush=(
                pg.mkBrush(
                    color
                )
            ),

            movable=False
        )
    )

    linear_region_item_lines = (
        linear_region_item.lines
    )

    assert (
        len(
            linear_region_item_lines
        ) ==

        2
    ), (
        linear_region_item_lines
    )

    for linear_region_item_line in (
            linear_region_item_lines
    ):
        linear_region_item_line.setPen(
            pg.mkPen(
                None
            )
        )

    linear_region_item.setZValue(-50)
    linear_region_item.ax = ax

    ax.addItem(
        linear_region_item
    )

    return linear_region_item


def add_vertical_band(x0, x1, color=FinPlotConstants.band_color, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    x_pts = _pdtime2index(ax, pd.Series([x0, x1]))
    color = _get_color(ax, None, color)
    lr = pg.LinearRegionItem([x_pts[0],x_pts[1]], orientation=pg.LinearRegionItem.Vertical, brush=pg.mkBrush(color), movable=False)
    lr.lines[0].setPen(pg.mkPen(None))
    lr.lines[1].setPen(pg.mkPen(None))
    lr.setZValue(-50)
    lr.ax = ax
    ax.addItem(lr)
    return lr


def add_rect(
        start_price,
        start_time,
        end_price,
        end_time,

        ax=None,

        color=(
            FinPlotConstants.band_color
        ),

        is_interactive=False,

        pen=None
) -> FinRect:
    ax = (
        _create_plot(
            ax=ax,
            maximize=False
        )
    )

    x_pts = (
        _pdtime2index(
            ax,

            pd.Series((
                start_time,
                end_time
            ))
        )
    )

    ix = (
        ax.vb.yscale.invxform
    )

    if (
            start_price >
            end_price
    ):
        (
            start_price,
            end_price
        ) = (
            end_price,
            start_price
        )

    pos = (
        x_pts[0],

        ix(
            start_price
        )
    )

    size = (
       (
           x_pts[1] -
           pos[0]
       ),

       (
           ix(
               end_price
           ) -

           ix(
               start_price
           )
       )
    )

    rect = (
        FinRect(
            ax=ax,

            brush=(
                pg.mkBrush(
                    color
                )
            ),

            pen=pen,
            pos=pos,
            size=size,

            movable=(
                is_interactive
            ),

            resizable=(
                is_interactive
            ),

            rotatable=False
        )
    )

    rect.setZValue(-40)

    if is_interactive:
        ax.vb.rois.append(
            rect
        )

    rect.ax = ax

    ax.addItem(
        rect
    )

    return rect


def add_line(p0, p1, color=FinPlotConstants.draw_line_color, width=1, style=None, interactive=False, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    used_color = _get_color(ax, style, color)
    pen = _makepen(color=used_color, style=style, width=width)
    x_pts = _pdtime2index(ax, pd.Series([p0[0], p1[0]]))
    ix = ax.vb.yscale.invxform
    pts = [(x_pts[0], ix(p0[1])), (x_pts[1], ix(p1[1]))]
    if interactive:
        line = FinPolyLine(ax.vb, pts, closed=False, pen=pen, movable=False)
        ax.vb.rois.append(line)
    else:
        line = FinLine(pts, pen=pen)
    line.ax = ax
    ax.addItem(line)
    return line


def add_text(
        pos,
        s,
        color=FinPlotConstants.draw_line_color,
        anchor=(0, 0),
        ax=None
):
    ax = _create_plot(ax=ax, maximize=False)
    color = _get_color(ax, None, color)
    text = pg.TextItem(s, color=color, anchor=anchor)
    x = pos[0]
    if ax.vb.datasrc is not None:
        x = _pdtime2index(ax, pd.Series([pos[0]]))[0]
    y = ax.vb.yscale.invxform(pos[1])
    text.setPos(x, y)
    text.setZValue(50)
    text.ax = ax
    ax.addItem(text, ignoreBounds=True)
    return text


def remove_line(line):
    print('remove_line() is deprecated, use remove_primitive() instead')
    remove_primitive(line)


def remove_text(text):
    print('remove_text() is deprecated, use remove_primitive() instead')
    remove_primitive(text)


def remove_primitive(
        primitive
) -> None:
    ax = primitive.ax

    ax.removeItem(
        primitive
    )

    if primitive in ax.vb.rois:
        ax.vb.rois.remove(primitive)

    if hasattr(primitive, 'texts'):
        for txt in primitive.texts:
            ax.vb.removeItem(txt)


def set_mouse_callback(callback, ax=None, when='click'):
    """Callback when clicked like so: callback(x, y)."""
    ax = (
        ax
        if ax
        else g_fin_plot_global_state.get_last_ax()
    )

    master = ax.ax_widget if hasattr(ax, 'ax_widget') else ax.vb.win
    if when == 'hover':
        master.proxy_hover = pg.SignalProxy(master.scene().sigMouseMoved, rateLimit=15, slot=partial(_mcallback_pos, ax, callback))
    elif when in ('dclick', 'double-click'):
        master.proxy_dclick = pg.SignalProxy(master.scene().sigMouseClicked, slot=partial(_mcallback_click, ax, callback, 'dclick'))
    elif when in ('click', 'lclick'):
        master.proxy_click = pg.SignalProxy(master.scene().sigMouseClicked, slot=partial(_mcallback_click, ax, callback, 'lclick'))
    elif when in ('mclick',):
        master.proxy_click = pg.SignalProxy(master.scene().sigMouseClicked, slot=partial(_mcallback_click, ax, callback, 'mclick'))
    elif when in ('rclick',):
        master.proxy_click = pg.SignalProxy(master.scene().sigMouseClicked, slot=partial(_mcallback_click, ax, callback, 'rclick'))
    elif when in ('any',):
        master.proxy_click = pg.SignalProxy(master.scene().sigMouseClicked, slot=partial(_mcallback_click, ax, callback, 'any'))
    else:
        print(f'Warning: unknown click "{when}" sent to set_mouse_callback()')


def set_time_inspector(callback, ax=None, when='click'):
    print('Warning: set_time_inspector() is a misnomer from olden days. Please use set_mouse_callback() instead.')
    set_mouse_callback(callback, ax, when)


def add_crosshair_info(infofunc, ax=None):
    """Callback when crosshair updated like so: info(ax,x,y,xtext,ytext); the info()
       callback must return two values: xtext and ytext."""
    ax = _create_plot(ax=ax, maximize=False)
    ax.crosshair.infos.append(infofunc)


def timer_callback(update_func, seconds, single_shot=False):
    timer = QtCore.QTimer()

    timer.timeout.connect(
        update_func
    )

    if single_shot:
        timer.setSingleShot(True)

    timer.start(
        int(
            seconds *
            1000
        )
    )

    g_fin_plot_global_state.add_timer(
        timer
    )

    return timer


def autoviewrestore(enable=True):
    """Restore functionality saves view zoom coordinates when closing a window, and
       load them when creating the plot (with the same name) again."""

    g_fin_plot_global_state.set_viewrestore(
        enable
    )


def refresh():
    overlay_axs = (
        g_fin_plot_global_state.get_overlay_axs()
    )

    for window in (
            g_fin_plot_global_state.get_windows()
    ):
        axs = (
            window.axs +
            [
                ax

                for ax in (
                    overlay_axs
                )

                if (
                    ax.vb.win ==
                    window
                )
            ]
        )

        for ax in (
                axs
        ):
            _improve_significants(
                ax
            )

        vbs = [
            ax.vb

            for ax in (
                axs
            )
        ]

        for vb in vbs:
            vb.pre_process_data()

        if g_fin_plot_global_state.get_viewrestore():
            if _loadwindata(window):
                continue

        _set_max_zoom(vbs)
        for vb in vbs:
            datasrc = vb.datasrc_or_standalone
            if datasrc and (vb.linkedView(0) is None or vb.linkedView(0).datasrc is None or vb.master_viewbox):
                vb.update_y_zoom(
                    datasrc.init_x0,
                    datasrc.init_x1
                )

    _repaint_candles()

    master_data_by_window_map = (
        g_fin_plot_global_state.get_master_data_by_window_map()
    )

    for window, master_window_data in (
            master_data_by_window_map.items()
    ):
        for vb in (
                master_window_data
        ):
            if type(vb) != str:  # ignore 'default'
                _mouse_moved(
                    window,
                    vb,
                    None
                )


def show(qt_exec=True):
    refresh()

    windows = (
        g_fin_plot_global_state.get_windows()
    )

    for window in (
            windows
    ):
        if isinstance(window, FinWindow) or qt_exec:
            if window.show_maximized:
                window.showMaximized()
            else:
                window.show()

    if (
            windows and
            qt_exec
    ):
        app = QtGui.QGuiApplication.instance()

        g_fin_plot_global_state.set_app(
            app
        )

        app.exec()

        g_fin_plot_global_state.clear_windows()
        g_fin_plot_global_state.clear_overlay_axs()
        g_fin_plot_global_state.clear_timers()
        g_fin_plot_global_state.clear_sound_effect_by_filename_map()
        g_fin_plot_global_state.clear_master_data_by_window_map()

        g_fin_plot_global_state.set_last_ax(
            None
        )


def play_sound(filename):
    sound_effect = (
        g_fin_plot_global_state.create_or_get_sound_effect(
            filename
        )
    )

    sound_effect.play()


def screenshot(file, fmt='png'):
    if (
            _is_internal_windows_only() and
            not g_fin_plot_global_state.get_app()
    ):
        print(
            'ERROR: screenshot must be callbacked from e.g. timer_callback()'
        )

        return False

    try:
        buffer = QtCore.QBuffer()

        app = (
            g_fin_plot_global_state.get_app()
        )

        first_window = (
            next(
                iter(
                    g_fin_plot_global_state.get_windows()
                )
            )
        )

        app.primaryScreen().grabWindow(
            first_window.winId()
        ).save(
            buffer,
            fmt
        )

        file.write(
            buffer.data()
        )

        return True
    except Exception as e:
        print(
            'Screenshot error:',
            type(e),
            e
        )
    return False


def experiment(*args, **kwargs):
    if 'opengl' in args or kwargs.get('opengl'):
        try:
            # pip install PyOpenGL PyOpenGL-accelerate to get this going
            import OpenGL
            pg.setConfigOptions(useOpenGL=True, enableExperimental=True)
        except Exception as e:
            print('WARNING: OpenGL init error.', type(e), e)
