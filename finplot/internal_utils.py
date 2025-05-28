import math
import os.path

from ast import literal_eval

from collections import (
    defaultdict
)

from datetime import (
    datetime  # ,
    # timezone
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

from .global_state import (
    g_fin_plot_global_state
)

from .item.fin_legend import (
    FinLegendItem
)

from .item.fin_plot import (
    FinPlotItem
)


# ################### INTERNALS ####################


def lerp(t, a, b):
    return (
        (
            t *
            b
        ) +

        (
            1 -
            t
        ) *

        a
    )


def _openfile(*args):
    return open(*args)


def _loadwindata(win):
    try:
        os.mkdir(os.path.expanduser('~/.finplot'))
    except:
        pass
    try:
        f = os.path.expanduser('~/.finplot/' + win.title.replace('/', '-') + '.ini')
        settings = [
            (
                k.strip(),

                literal_eval(
                    v.strip()
                )
            )

            for line in _openfile(f)
            for k, d, v in [
                line.partition(
                    '='
                )
            ]
            if v
        ]
        if not settings:
            return
    except:
        return

    kvs = {
        k: v
        for k, v in (
            settings
        )
    }

    right_margin_candles = (
        g_fin_plot_global_state.get_right_margin_candles()
    )

    vbs = set(ax.vb for ax in win.axs)
    zoom_set = False
    for vb in vbs:
        ds = vb.datasrc
        if ds and (vb.linkedView(0) is None or vb.linkedView(0).datasrc is None or vb.master_viewbox):
            period_ns = ds.period_ns
            if kvs['min_x'] >= ds.x.iloc[0] - period_ns and kvs['max_x'] <= ds.x.iloc[-1] + period_ns:
                x0, x1 = ds.x.loc[ds.x >= kvs['min_x']].index[0], ds.x.loc[ds.x <= kvs['max_x']].index[-1]

                if x1 == len(ds.x) - 1:
                    x1 += right_margin_candles

                x1 += 0.5
                zoom_set = vb.update_y_zoom(x0, x1)

    return zoom_set


def _savewindata(win):
    if not g_fin_plot_global_state.get_viewrestore():
        return

    try:
        min_x = int(1e100)
        max_x = int(-1e100)
        for ax in win.axs:
            if ax.vb.targetRect().right() < 4:  # ignore empty plots
                continue
            if ax.vb.datasrc is None:
                continue
            t0, t1, _, _, _ = ax.vb.datasrc.hilo(ax.vb.targetRect().left(), ax.vb.targetRect().right())
            min_x = np.nanmin([min_x, t0])
            max_x = np.nanmax([max_x, t1])
        if np.max(np.abs([min_x, max_x])) < 1e99:
            s = 'min_x = %s\nmax_x = %s\n' % (min_x, max_x)
            f = os.path.expanduser('~/.finplot/' + win.title.replace('/', '-') + '.ini')
            try:
                changed = _openfile(f).read() != s
            except:
                changed = True
            if changed:
                _openfile(f, 'wt').write(s)
                # # print('%s saved' % win.title)
    except Exception as e:
        print('Error saving plot:', e)


def _ax_set_visible(ax, crosshair=None, xaxis=None, yaxis=None, xgrid=None, ygrid=None):
    if crosshair is False:
        ax.crosshair.hide()

    if xaxis is not None:
        ax.getAxis(
            'bottom'
        ).setStyle(
            showValues=xaxis
        )

    if yaxis is not None:
        ax.getAxis(
            'right'
        ).setStyle(
            showValues=yaxis
        )

    if xgrid is not None or ygrid is not None:
        ax.showGrid(
            x=xgrid,
            y=ygrid,

            alpha=(
                FinPlotConstants.grid_alpha
            )
        )

        right_axis = (
            ax.getAxis(
                'right'
            )
        )

        if right_axis:
            right_axis.setEnabled(
                False
            )

        bottom_axis = (
            ax.getAxis(
                'bottom'
            )
        )

        if bottom_axis:
            bottom_axis.setEnabled(
                False
            )


def _ax_decouple(ax):
    ax.setXLink(None)
    if ax.prev_ax:
        ax.prev_ax.set_visible(xaxis=True)


def _ax_disable_x_index(ax, decouple=True):
    ax.vb.x_indexed = False
    if decouple:
        _ax_decouple(ax)


def _ax_invert_y(ax):
    ax.inverted = not ax.inverted
    ax.setTransform(ax.transform().scale(1, -1).translate(0, -ax.height()))


def _ax_expand(ax):
    vb = ax.vb
    axs = vb.win.axs
    for axx in axs:
        if axx.inverted:  # roll back inversion if changing expansion
            axx.invert_y()
    show_all = any(not ax.isVisible() for ax in axs)
    for rowi, axx in enumerate(axs):
        if axx != ax:
            axx.setVisible(show_all)
            for axo in axx.axos:
                axo.vb.setVisible(show_all)
    ax.set_visible(xaxis=not show_all)
    axs[-1].set_visible(xaxis=True)


def _create_legend(ax):
    if ax.vb.master_viewbox:
        ax = ax.vb.master_viewbox.parent()

    if ax.legend is None:
        ax.legend = (
            FinLegendItem(
                border_color=(
                    FinPlotConstants.legend_border_color
                ),

                fill_color=(
                    FinPlotConstants.legend_fill_color
                ),

                size=None,
                offset=(3, 2)
            )
        )

        ax.legend.setParentItem(
            ax.vb
        )


def _update_significants(ax, datasrc, force):
    # check if no epsilon set yet
    default_dec = (
        0.99 <

        (
            ax.significant_decimals /
            FinPlotConstants.significant_decimals
        ) <

        1.01
    )

    default_eps = (
        0.99 <

        (
            ax.significant_eps /
            FinPlotConstants.significant_eps
        ) <

        1.01
    )

    if force or (default_dec and default_eps):
        try:
            (
                sd,
                se
            ) = (
                datasrc.calc_significant_decimals(
                    full=force
                )
            )

            if (
                    sd or

                    (
                        se !=
                        FinPlotConstants.significant_eps
                    )
            ):
                if (force and not ax.significant_forced) or default_dec or sd > ax.significant_decimals:
                    ax.significant_decimals = sd
                    ax.significant_forced |= force
                if (force and not ax.significant_forced) or default_eps or se < ax.significant_eps:
                    ax.significant_eps = se
                    ax.significant_forced |= force
        except:
            pass  # datasrc probably full av NaNs


def _is_standalone(timeser):
    # more than N percent gaps or time reversals probably means this is a standalone plot
    return timeser.isnull().sum() + (timeser.diff() <= 0).sum() > len(timeser) * 0.1


def _create_series(a):
    return a if isinstance(a, pd.Series) else pd.Series(a)


def _set_datasrc(ax, datasrc, addcols=True):
    viewbox = ax.vb
    if not datasrc.standalone:
        if viewbox.datasrc is None:
            viewbox.set_datasrc(datasrc)  # for mwheel zoom-scaling
            _set_x_limits(ax, datasrc)
        else:
            t0 = viewbox.datasrc.x.loc[0]
            if addcols:
                viewbox.datasrc.addcols(datasrc)
            else:
                viewbox.datasrc.df = datasrc.df
            # check if we need to re-render previous plots due to changed indices
            indices_updated = viewbox.datasrc.timebased() and t0 != viewbox.datasrc.x.loc[0]
            for item in ax.items:
                if hasattr(item, 'datasrc') and not item.datasrc.standalone:
                    item.datasrc.set_df(viewbox.datasrc.df)  # every plot here now has the same time-frame
                    if indices_updated:
                        _start_visual_update(item)
                        _end_visual_update(item)
            _set_x_limits(ax, datasrc)
            viewbox.set_datasrc(viewbox.datasrc)  # update zoom
    else:
        viewbox.standalones.add(datasrc)
        datasrc.update_init_x(viewbox.init_steps)
        if datasrc.timebased() and viewbox.datasrc is not None:
            ## print('WARNING: re-indexing standalone, time-based plot')
            vdf = viewbox.datasrc.df
            d = {v: k for k, v in enumerate(vdf[vdf.columns[0]])}
            datasrc.df.index = [d[i] for i in datasrc.df[datasrc.df.columns[0]]]
        ## if not viewbox.x_indexed:
        ## _set_x_limits(ax, datasrc)
    # update period if this datasrc has higher time resolution

    epoch_period = (
        g_fin_plot_global_state.get_epoch_period()
    )

    if (
            datasrc.timebased() and

            (
                epoch_period > 1e7 or
                not datasrc.standalone
            )
    ):
        ep_secs = datasrc.period_ns / 1e9

        if ep_secs < epoch_period:
            epoch_period = ep_secs

            g_fin_plot_global_state.set_epoch_period(
                epoch_period
            )


def _has_timecol(df):
    return (
        len(
            df.columns
        ) >=

        2
    )


def _set_max_zoom(vbs):
    """Set the relative allowed zoom level between axes groups, where the lowest-resolution
       plot in each group uses g_fin_plot_global_state.get_max_zoom_points(),
       while the others get a scale >=1 of their respective highest zoom level."""
    groups = defaultdict(set)
    for vb in vbs:
        master_vb = vb.linkedView(0)
        if vb.datasrc and master_vb and master_vb.datasrc:
            groups[master_vb].add(vb)
            groups[master_vb].add(master_vb)
    for group in groups.values():
        minlen = min(len(vb.datasrc.df) for vb in group)
        for vb in group:
            vb.max_zoom_points_f = len(vb.datasrc.df) / minlen


def _adjust_renko_datasrc(bins, step, datasrc):
    if not bins and not step:
        bins = 50
    if not step:
        step = (datasrc.y.max() - datasrc.y.min()) / bins
    bricks = datasrc.y.diff() / step
    bricks = (datasrc.y[bricks.isnull() | (bricks.abs() >= 0.5)] / step).round().astype(int)
    extras = datasrc.df.iloc[:, datasrc.col_data_offset + 1:]
    ts = datasrc.x[bricks.index]
    up = bricks.iloc[0] + 1
    dn = up - 2
    data = []
    for t, i, brick in zip(ts, bricks.index, bricks):
        s = 0
        if brick >= up:
            x0, x1, s = up - 1, brick, +1
            up = brick + 1
            dn = brick - 2
        elif brick <= dn:
            x0, x1, s = dn, brick - 1, -1
            up = brick + 2
            dn = brick - 1
        if s:
            for x in range(x0, x1, s):
                td = abs(x1 - x) - 1
                ds = 0 if s > 0 else step
                y = x * step
                z = list(extras.loc[i])
                data.append([t - td, y + ds, y + step - ds, y + step, y] + z)
    datasrc.set_df(pd.DataFrame(data, columns='time open close high low'.split() + list(extras.columns)))


def _adjust_renko_log_datasrc(bins, step, datasrc):
    datasrc.df.loc[:, datasrc.df.columns[1]] = np.log10(datasrc.df.iloc[:, 1])
    _adjust_renko_datasrc(bins, step, datasrc)
    datasrc.df.loc[:, datasrc.df.columns[1:5]] = 10 ** datasrc.df.iloc[:, 1:5]


def _adjust_volume_datasrc(datasrc):
    if len(datasrc.df.columns) <= 4:
        _insert_col(datasrc, 3, '_zero_', [0] * len(datasrc.df))  # base of candles is always zero
    datasrc.set_df(datasrc.df.iloc[:, [0, 3, 4, 1, 2]])  # re-arrange columns for rendering
    datasrc.scale_cols = [1, 2]  # scale by both baseline and volume


def _preadjust_horiz_datasrc(datasrc):
    arrayify = lambda d: d.values if type(d) == pd.DataFrame else d
    # create a dataframe from the input array
    times = [t for t, row in datasrc]
    if len(times) == 1:  # add an empty time slot
        times = [times[0], times[0] + 1]
        datasrc = datasrc + [[times[1], [(p, 0) for p, v in arrayify(datasrc[0][1])]]]
    data = [[e for v in arrayify(row) for e in v] for t, row in datasrc]
    maxcols = max(len(row) for row in data)
    return pd.DataFrame(columns=range(maxcols), data=data, index=times)


def _adjust_horiz_datasrc(datasrc):
    # to be able to scale properly, move the last two values to the last two columns
    values = datasrc.df.iloc[:, 1:].values
    for i, nrow in enumerate(values):
        orow = nrow[~np.isnan(nrow)]
        if len(nrow) == len(orow) or len(orow) <= 2:
            continue
        nrow[-2:] = orow[-2:]
        nrow[len(orow) - 2:len(orow)] = np.nan
    datasrc.df[datasrc.df.columns[1:]] = values


def _adjust_bar_datasrc(datasrc, order_cols=True):
    if len(datasrc.df.columns) <= 2:
        _insert_col(datasrc, 1, '_base_', [0] * len(datasrc.df))  # base
    if len(datasrc.df.columns) <= 4:
        _insert_col(datasrc, 1, '_open_', [0] * len(datasrc.df))  # "open" for color
        _insert_col(datasrc, 2, '_close_', datasrc.df.iloc[:, 3])  # "close" (actual bar value) for color
    if order_cols:
        datasrc.set_df(datasrc.df.iloc[:, [0, 3, 4, 1, 2]])  # re-arrange columns for rendering
    datasrc.scale_cols = [1, 2]  # scale by both baseline and volume


def _update_gfx(item):
    _start_visual_update(item)
    for i in item.ax.items:
        if i == item or not hasattr(i, 'datasrc'):
            continue
        lc = len(item.datasrc.df)
        li = len(i.datasrc.df)
        if lc and li and max(lc, li) / min(lc, li) > 100:  # TODO: should be typed instead
            continue
        cc = item.datasrc.df.columns
        ci = i.datasrc.df.columns
        c0 = [c for c in ci if c in cc]
        c1 = [c for c in ci if not c in cc]
        df_clipped = item.datasrc.df[c0]
        if c1:
            df_clipped = df_clipped.copy()
            for c in c1:
                df_clipped[c] = i.datasrc.df[c]
        i.datasrc.set_df(df_clipped)
        break

    right_margin_candles = (
        g_fin_plot_global_state.get_right_margin_candles()
    )

    update_sigdig = False

    if not (
            item.datasrc.standalone or
            item.ax.vb.win._isMouseLeftDrag
    ):
        # new limits when extending/reducing amount of data
        x_min, x1 = (
            _set_x_limits(
                item.ax,
                item.datasrc
            )
        )

        # scroll all plots if we're at the far right

        tr = (
            item.ax.vb.targetRect()
        )

        x0 = (
            x1 -
            tr.width()
        )

        if (
                x0 <

                (
                    right_margin_candles +
                    FinPlotConstants.side_margin
                )
        ):
            x0 = (
                -FinPlotConstants.side_margin
            )

        if (
                tr.right() <

                (
                    x1 -
                    5 -

                    (
                        2 *
                        right_margin_candles
                    )
                )
        ):
            x0 = x1 = None

        prev_top = item.ax.vb.targetRect().top()
        item.ax.vb.update_y_zoom(x0, x1)
        this_top = item.ax.vb.targetRect().top()

        if this_top and not (0.99 < abs(prev_top / this_top) < 1.01):
            update_sigdig = True

        if item.ax.axes['bottom']['item'].isVisible():  # update axes if visible
            item.ax.axes['bottom']['item'].hide()
            item.ax.axes['bottom']['item'].show()

    _end_visual_update(item)

    if update_sigdig:
        _update_significants(item.ax, item.ax.vb.datasrc, force=True)


def _start_visual_update(item):
    if isinstance(item, FinPlotItem):
        item.ax.removeItem(item)
        item.dirty = True
    else:
        y = item.datasrc.y / item.ax.vb.yscale.scalef

        if item.ax.vb.yscale.scaletype == 'log':
            y += FinPlotConstants.log_plot_offset

        x = item.datasrc.index if item.ax.vb.x_indexed else item.datasrc.x
        item.setData(x, y)


def _end_visual_update(item):
    if isinstance(item, FinPlotItem):
        item.ax.addItem(item)
        item.repaint()


def _set_x_limits(ax, datasrc):
    x0, x1 = _xminmax(datasrc, x_indexed=ax.vb.x_indexed)
    ax.setLimits(xMin=x0, xMax=x1)
    return x0, x1


def _xminmax(datasrc, x_indexed, init_steps=None, extra_margin=0):
    right_margin_candles = (
        g_fin_plot_global_state.get_right_margin_candles()
    )

    if (
            x_indexed and
            init_steps
    ):
        # initial zoom

        x0 = (
            max(
                (
                    datasrc.xlen -
                    init_steps
                ),

                0
            ) -

            FinPlotConstants.side_margin -
            extra_margin
        )

        x1 = (
            datasrc.xlen +
            right_margin_candles +
            FinPlotConstants.side_margin +
            extra_margin
        )
    elif x_indexed:
        # total x size for indexed data
        x0 = (
            -FinPlotConstants.side_margin -
            extra_margin
        )

        x1 = (  # add another margin to get the "snap back" sensation
            datasrc.xlen +
            right_margin_candles -
            1 +
            FinPlotConstants.side_margin +
            extra_margin
        )
    else:
        # x size for plain Y-over-X data (i.e. not indexed)
        x0 = datasrc.x.min()
        x1 = datasrc.x.max()
        # extend margin on decoupled plots
        d = (x1 - x0) * (0.2 + extra_margin)
        x0 -= d
        x1 += d

    return x0, x1


def _repaint_candles():
    """Candles are only partially drawn, and therefore needs manual dirty reminder whenever it goes off-screen."""
    axs = (
        [
            ax

            for window in (
                g_fin_plot_global_state.get_windows()
            )

            for ax in (
                window.axs
            )
        ] +
        g_fin_plot_global_state.get_overlay_axs()
    )

    for ax in axs:
        for item in list(ax.items):
            if isinstance(item, FinPlotItem):
                _start_visual_update(
                    item
                )

                _end_visual_update(
                    item
                )


def _paint_scatter(item, p, *args):
    with (
            np.errstate(
                invalid=(
                    'ignore'  # make pg's mask creation calls to numpy shut up
                )
            )
    ):
        item._dopaint(  # noqa
            p,
            *args
        )


def _key_pressed(vb, ev):
    if ev.text() == 'g':  # grid
        clamp_grid = (
            g_fin_plot_global_state.get_clamp_grid()
        )

        clamp_grid = (
            not clamp_grid
        )

        g_fin_plot_global_state.set_clamp_grid(
            clamp_grid
        )

        for window in (
                g_fin_plot_global_state.get_windows()
        ):
            for ax in window.axs:
                ax.crosshair.update()
    elif ev.text() == 'i':  # invert y-axes
        for window in (
                g_fin_plot_global_state.get_windows()
        ):
            for ax in window.axs:
                ax.invert_y()
    elif ev.text() == 'f':  # focus/expand active axis
        vb.parent().expand()
    elif ev.text() in ('\r', ' '):  # enter, space
        vb.set_draw_line_color(
            FinPlotConstants.draw_done_color
        )

        vb.draw_line = None
    elif ev.text() in ('\x7f', '\b'):  # del, backspace
        if not vb.remove_last_roi():
            return False
    elif ev.key() == QtCore.Qt.Key.Key_Left:
        vb.pan_x(percent=-15)
    elif ev.key() == QtCore.Qt.Key.Key_Right:
        vb.pan_x(percent=+15)
    elif ev.key() == QtCore.Qt.Key.Key_Home:
        vb.pan_x(steps=-1e10)
        _repaint_candles()
    elif ev.key() == QtCore.Qt.Key.Key_End:
        vb.pan_x(steps=+1e10)
        _repaint_candles()
    elif (
            (
                ev.key() ==
                QtCore.Qt.Key.Key_Escape
            ) and

            FinPlotConstants.key_esc_close
    ):
        vb.win.close()
    else:
        return False
    return True


def _mouse_clicked(vb, ev):
    if ev.button() == 8:  # back
        vb.pan_x(percent=-30)
    elif ev.button() == 16:  # fwd
        vb.pan_x(percent=+30)
    else:
        return False
    return True


def _mouse_moved(master, vb, evs):
    if hasattr(master, 'closing') and master.closing:
        return

    master_data_by_window_map = (
        g_fin_plot_global_state.get_master_data_by_window_map()
    )

    master_window_data = (
        master_data_by_window_map.get(
            master
        )
    )

    assert (
        master_window_data is not None
    ), (
        master
    )

    md = (
        master_window_data.get(vb) or
        master_window_data.get('default')
    )

    if md is None:
        return

    if not evs:
        evs = md['last_mouse_evs']
        if not evs:
            return

    md['last_mouse_evs'] = evs
    pos = evs[-1]
    # allow inter-pixel moves if moving mouse slowly
    y = pos.y()
    dy = y - md['last_mouse_y']
    if 0 < abs(dy) <= 1:
        pos.setY(pos.y() - dy / 2)
    md['last_mouse_y'] = y
    # apply to all crosshairs
    for ax in master.axs:
        if ax.isVisible() and ax.crosshair:
            point = ax.vb.mapSceneToView(pos)
            ax.crosshair.update(point)


def _wheel_event_wrapper(self, orig_func, ev):
    # scrolling on the border is simply annoying, pop in a couple of pixels to make sure
    d = QtCore.QPointF(-2, 0)

    ev = QtGui.QWheelEvent(
        ev.position() + d,
        ev.globalPosition() + d,
        ev.pixelDelta(),
        ev.angleDelta(),
        ev.buttons(),
        ev.modifiers(),
        ev.phase(),
        False
    )

    orig_func(self, ev)


def _mcallback_click(ax, callback, when, evs):
    if evs[-1].accepted:
        return
    if when == 'dclick' and evs[-1].double():
        pass
    elif when == 'lclick' and evs[-1].button() == QtCore.Qt.MouseButton.LeftButton:
        pass
    elif when == 'mclick' and evs[-1].button() == QtCore.Qt.MouseButton.MiddleButton:
        pass
    elif when == 'rclick' and evs[-1].button() == QtCore.Qt.MouseButton.RightButton:
        pass
    elif when == 'any':
        pass
    else:
        return
    pos = evs[-1].scenePos()
    return _mcallback_pos(ax, callback, (pos,))


def _mcallback_pos(ax, callback, poss):
    if not ax.vb.datasrc:
        return

    point = ax.vb.mapSceneToView(poss[-1])
    t = point.x() + 0.5

    try:
        t = ax.vb.datasrc.closest_time(t)
    except KeyError:  # when clicking beyond g_fin_plot_global_state.get_right_margin_candles()
        if g_fin_plot_global_state.get_clamp_grid():
            t = ax.vb.datasrc.x.iloc[-1 if t > 0 else 0]

    try:
        callback(t, point.y())
    except OSError as e:
        pass
    except Exception as e:
        print('Inspection error:', type(e), e)


def brighten(
        color,
        f
):
    if not color:
        return color

    return (
        pg.mkColor(
            color
        ).lighter(
            int(
                f *
                100
            )
        )
    )


def _get_color(ax, style, wanted_color):
    if type(wanted_color) in (str, QtGui.QColor):
        return wanted_color

    index = wanted_color if type(wanted_color) == int else None
    is_line = lambda style: style is None or any(ch in style for ch in '-_.')
    get_handed_color = lambda item: item.opts.get('handed_color')

    this_line = (
        is_line(
            style
        )
    )

    if this_line:
        colors = FinPlotConstants.soft_colors
    else:
        colors = FinPlotConstants.hard_colors

    if index is None:
        avoid = set(i.opts['handed_color'] for i in ax.items if
                    isinstance(i, pg.PlotDataItem) and get_handed_color(i) is not None and this_line == is_line(
                        i.opts['symbol']))
        index = len([i for i in ax.items if
                     isinstance(i, pg.PlotDataItem) and get_handed_color(i) is None and this_line == is_line(
                         i.opts['symbol'])])
        while index in avoid:
            index += 1

    return (
        colors[
            index %

            len(
                colors
            )
        ]
    )


def _pdtime2epoch(t):
    if isinstance(t, pd.Series):
        if isinstance(t.iloc[0], pd.Timestamp):
            dtype = str(t.dtype)
            if dtype.endswith('[s]'):
                return t.astype('int64') * int(1e9)
            elif dtype.endswith('[ms]'):
                return t.astype('int64') * int(1e6)
            elif dtype.endswith('us'):
                return t.astype('int64') * int(1e3)
            return t.astype('int64')
        h = np.nanmax(t.values)
        if h < 1e10:  # handle s epochs
            return (t * 1e9).astype('int64')
        if h < 1e13:  # handle ms epochs
            return (t * 1e6).astype('int64')
        if h < 1e16:  # handle us epochs
            return (t * 1e3).astype('int64')
        return t.astype('int64')
    return t


def _pdtime2index(ax, ts, any_end=False, require_time=False):
    if isinstance(ts.iloc[0], pd.Timestamp):
        ts = ts.astype('int64')
    else:
        h = np.nanmax(ts.values)
        if h < 1e7:
            if require_time:
                assert False, 'not a time series'
            return ts
        if h < 1e10:  # handle s epochs
            ts = ts.astype('float64') * 1e9
        elif h < 1e13:  # handle ms epochs
            ts = ts.astype('float64') * 1e6
        elif h < 1e16:  # handle us epochs
            ts = ts.astype('float64') * 1e3

    datasrc = _get_datasrc(ax)
    xs = datasrc.x

    # try exact match before approximate match
    if all(xs.isin(ts)):
        exact = datasrc.index[ts].to_list()
        if len(exact) == len(ts):
            return exact

    r = []
    for i, t in enumerate(ts):
        xss = xs.loc[xs > t]
        if len(xss) == 0:
            t0 = xs.iloc[-1]
            if any_end or t0 == t:
                r.append(len(xs) - 1)
                continue
            if i > 0:
                continue
            assert t <= t0, 'must plot this primitive in prior time-range'
        i1 = xss.index[0]
        i0 = i1 - 1
        if i0 < 0:
            i0, i1 = 0, 1
        t0, t1 = xs.loc[i0], xs.loc[i1]
        if t0 == t1:
            r.append(i0)
        else:
            dt = (t - t0) / (t1 - t0)
            r.append(lerp(dt, i0, i1))
    return r


def _is_str_midnight(s):
    return s.endswith(' 00:00') or s.endswith(' 00:00:00')


def _get_datasrc(ax, require=True):
    if (
            ax.vb.datasrc is not None or
            not ax.vb.x_indexed
    ):
        return ax.vb.datasrc

    vbs = [
        ax.vb

        for window in (
            g_fin_plot_global_state.get_windows()
        )

        for ax in (
            window.axs
        )
    ]

    for vb in vbs:
        if vb.datasrc:
            return vb.datasrc

    if require:
        assert ax.vb.datasrc, 'not possible to plot this primitive without a prior time-range to compare to'


def _insert_col(datasrc, col_idx, col_name, data):
    if col_name not in datasrc.df.columns:
        datasrc.df.insert(col_idx, col_name, data)


def _millisecond_tz_wrap(s):
    if len(s) > 6 and s[-6] in '+-' and s[-3] == ':':  # +01:00 fmt timezone present?
        s = s[:-6]
    return (s + '.000000') if '.' not in s else s


def _x2local_t(datasrc, x):
    if FinPlotConstants.display_timezone is None:
        return (
            _x2utc(
                datasrc,
                x
            )
        )

    return (
        _x2t(
            datasrc,
            x,

            lambda t: (
                _millisecond_tz_wrap(
                    datetime.fromtimestamp(
                        t / 1e9,
                        tz=FinPlotConstants.display_timezone
                    ).strftime(
                        FinPlotConstants.timestamp_format
                    )
                )
            )
        )
    )


def _x2utc(datasrc, x):
    # using pd.to_datetime allow for pre-1970 dates
    return (
        _x2t(
            datasrc,
            x,

            lambda t: (
                pd.to_datetime(
                    t,
                    unit='ns'
                ).strftime(
                    FinPlotConstants.timestamp_format
                )
            )
        )
    )


def _x2t(datasrc, x, ts2str):
    if not datasrc:
        return '', False

    try:
        x += 0.5
        t, _, _, _, cnt = datasrc.hilo(x, x)
        if cnt:
            if not datasrc.timebased():
                return '%g' % t, False

            s = ts2str(t)

            if not FinPlotConstants.truncate_timestamp:
                return s, True

            epoch_period = (
                g_fin_plot_global_state.get_epoch_period()
            )

            if epoch_period >= 23 * 60 * 60:  # daylight savings, leap seconds, etc
                i = s.index(' ')
            elif epoch_period >= 59:  # consider leap seconds
                i = s.rindex(':')
            elif epoch_period >= 1:
                i = s.index('.') if '.' in s else len(s)
            elif epoch_period >= 0.001:
                i = -3
            else:
                i = len(s)

            return s[:i], True
    except Exception as e:
        import traceback
        traceback.print_exc()

    return '', datasrc.timebased()


def _x2year(datasrc, x):
    t, hasds = _x2local_t(datasrc, x)
    return t[:4], hasds


def _round_to_significant(rng, rngmax, x, significant_decimals, significant_eps):
    is_highres = (rng / significant_eps > 1e2 and rngmax < 1e-2) or abs(rngmax) > 1e7 or rng < 1e-5
    sd = significant_decimals
    if is_highres and abs(x) > 0:
        exp10 = math.floor(np.log10(abs(x)))
        x = x / (10 ** exp10)
        rm = int(abs(np.log10(rngmax))) if rngmax > 0 else 0
        sd = min(3, sd + rm)
        fmt = '%%%i.%ife%%i' % (sd, sd)
        r = fmt % (x, exp10)
    else:
        eps = math.fmod(x, significant_eps)
        if abs(eps) >= significant_eps / 2:
            # round up
            eps -= np.sign(eps) * significant_eps
        xx = x - eps
        fmt = '%%%i.%if' % (sd, sd)
        r = fmt % xx
        if abs(x) > 0 and rng < 1e4 and r.startswith('0.0') and float(r[:-1]) == 0:
            r = '%.2e' % x
    return r


def _roihandle_move_snap(vb, orig_func, pos, modifiers=QtCore.Qt.KeyboardModifier, finish=True):
    pos = vb.mapDeviceToView(pos)
    pos = _clamp_point(vb.parent(), pos)
    pos = vb.mapViewToDevice(pos)
    orig_func(pos, modifiers=modifiers, finish=finish)


def _clamp_xy(ax, x, y):
    if not g_fin_plot_global_state.get_clamp_grid():
        return x, y

    y = ax.vb.yscale.xform(y)
    # scale x
    if ax.vb.x_indexed:
        ds = ax.vb.datasrc
        if x < 0 or (ds and x > len(ds.df) - 1):
            x = 0 if x < 0 else len(ds.df) - 1
        x = _round(x)
    # scale y
    if y < 0.1 and ax.vb.yscale.scaletype == 'log':
        magnitude = int(3 - np.log10(y))  # round log to N decimals
        y = round(y, magnitude)
    else:  # linear
        eps = ax.significant_eps
        if eps > 1e-8:
            eps2 = np.sign(y) * 0.5 * eps
            y -= math.fmod(y + eps2, eps) - eps2
    y = ax.vb.yscale.invxform(y, verify=True)
    return x, y


def _clamp_point(
        ax,
        p
):
    if g_fin_plot_global_state.get_clamp_grid():
        x, y = (
            _clamp_xy(
                ax,
                p.x(),
                p.y()
            )
        )

        return (
            pg.Point(
                x,
                y
            )
        )

    return p


def _draw_line_segment_text(polyline, segment, pos0, pos1):
    fsecs = None
    datasrc = polyline.vb.datasrc

    if (
            datasrc and
            g_fin_plot_global_state.get_clamp_grid()
    ):
        try:
            x0, x1 = pos0.x() + 0.5, pos1.x() + 0.5
            t0, _, _, _, cnt0 = datasrc.hilo(x0, x0)
            t1, _, _, _, cnt1 = datasrc.hilo(x1, x1)

            if cnt0 and cnt1:
                fsecs = abs(t1 - t0) / 1e9
        except:
            pass

    epoch_period = (
        g_fin_plot_global_state.get_epoch_period()
    )

    diff = pos1 - pos0
    if fsecs is None:
        fsecs = (
            abs(
                diff.x() *
                epoch_period
            )
        )

    secs = int(fsecs)
    mins = secs // 60
    hours = mins // 60
    mins = mins % 60
    secs = secs % 60

    if (
            hours == 0 and
            mins == 0 and
            secs < 60 and
            epoch_period < 1
    ):
        msecs = int((fsecs - int(fsecs)) * 1000)
        ts = '%0.2i:%0.2i.%0.3i' % (mins, secs, msecs)
    elif (
            hours == 0 and
            mins < 60 and
            epoch_period < 60
    ):
        ts = '%0.2i:%0.2i:%0.2i' % (hours, mins, secs)
    elif hours < 24:
        ts = '%0.2i:%0.2i' % (hours, mins)
    else:
        days = hours // 24
        hours %= 24
        ts = '%id %0.2i:%0.2i' % (days, hours, mins)
        if ts.endswith(' 00:00'):
            ts = ts.partition(' ')[0]
    ysc = polyline.vb.yscale
    if polyline.vb.y_positive:
        y0, y1 = ysc.xform(pos0.y()), ysc.xform(pos1.y())
        if y0:
            gain = y1 / y0 - 1
            if gain > 10:
                value = 'x%i' % gain
            else:
                value = '%+.2f %%' % (100 * gain)
        elif not y1:
            value = '0'
        else:
            value = '+∞' if y1 > 0 else '-∞'
    else:
        dy = ysc.xform(diff.y())
        if dy and (abs(dy) >= 1e4 or abs(dy) <= 1e-2):
            value = '%+3.3g' % dy
        else:
            value = '%+2.2f' % dy

    extra = _draw_line_extra_text(polyline, segment, pos0, pos1)

    return '%s %s (%s)' % (value, extra, ts)


def _draw_line_extra_text(polyline, segment, pos0, pos1):
    """Shows the proportions of this line height compared to the previous segment."""
    prev_text = None
    for text in polyline.texts:
        if prev_text is not None and text.segment == segment:
            h0 = prev_text.segment.handles[0]['item']
            h1 = prev_text.segment.handles[1]['item']
            prev_change = h1.pos().y() - h0.pos().y()
            this_change = pos1.y() - pos0.y()
            if not abs(prev_change) > 1e-14:
                break
            change_part = abs(this_change / prev_change)
            return ' = 1:%.2f ' % change_part
        prev_text = text
    return ''


def _makepen(color, style=None, width=1):
    if style is None or style == '-':
        return pg.mkPen(color=color, width=width)
    dash = []
    for ch in style:
        if ch == '-':
            dash += [4, 2]
        elif ch == '_':
            dash += [10, 2]
        elif ch == '.':
            dash += [1, 2]
        elif ch == ' ':
            if dash:
                dash[-1] += 2
    return pg.mkPen(color=color, style=QtCore.Qt.PenStyle.CustomDashLine, dash=dash, width=width)


def _round(v):
    return (
        math.floor(
            v +
            0.5
        )
    )
