import os
import os.path

from finplot.other import *


lerp = lambda t,a,b: t*b+(1-t)*a


def create_plot(title='Finance Plot', rows=1, init_zoom_periods=1e10, maximize=True, yscale='linear'):
    pg.setConfigOptions(foreground=foreground, background=background)
    win = FinWindow(title)
    win.show_maximized = maximize
    ax0 = axs = create_plot_widget(master=win, rows=rows, init_zoom_periods=init_zoom_periods, yscale=yscale)
    axs = axs if type(axs) in (tuple,list) else [axs]
    for ax in axs:
        win.addItem(ax, col=1)
        win.nextRow()
    return ax0


def create_plot_widget(master, rows=1, init_zoom_periods=1e10, yscale='linear'):
    pg.setConfigOptions(foreground=foreground, background=background)
    global last_ax
    if master not in windows:
        windows.append(master)
    axs = []
    prev_ax = None
    for n in range(rows):
        ysc = yscale[n] if type(yscale) in (list,tuple) else yscale
        ysc = YScale(ysc, 1) if type(ysc) == str else ysc
        viewbox = FinViewBox(master, init_steps=init_zoom_periods, yscale=ysc, v_zoom_scale=1-y_pad, enableMenu=False)
        ax = prev_ax = _add_timestamp_plot(master=master, prev_ax=prev_ax, viewbox=viewbox, index=n, yscale=ysc)
        if axs:
            ax.setXLink(axs[0].vb)
        else:
            viewbox.setFocus()
        axs += [ax]
    if master not in master_data:
        master_data[master] = {}
    if isinstance(master, pg.GraphicsLayoutWidget):
        proxy = pg.SignalProxy(master.scene().sigMouseMoved, rateLimit=144, slot=partial(_mouse_moved, master, axs[0].vb))
        master_data[master][axs[0].vb] = dict(proxymm=proxy, last_mouse_evs=None, last_mouse_y=0)
        if 'default' not in master_data[master]:
            master_data[master]['default'] = master_data[master][axs[0].vb]
    else:
        for ax in axs:
            proxy = pg.SignalProxy(ax.ax_widget.scene().sigMouseMoved, rateLimit=144, slot=partial(_mouse_moved, master, ax.vb))
            master_data[master][ax.vb] = dict(proxymm=proxy, last_mouse_evs=None, last_mouse_y=0)
    last_ax = axs[0]
    return axs[0] if len(axs) == 1 else axs


def close():
    for win in windows:
        try:
            win.close()
        except Exception as e:
            print('Window closing error:', type(e), e)
    global last_ax
    windows.clear()
    overlay_axs.clear()
    _clear_timers()
    sounds.clear()
    master_data.clear()
    last_ax = None


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
    '''The sections argument is a (starting_index, color_name) array.'''
    def _colorfilter(sections, item, datasrc, df):
        if not sections:
            return volume_colorfilter(item, datasrc, df)
        for (i0,colname),(i1,_) in zip(sections, sections[1:]+[(None,'neutral')]):
            rows = df.iloc[i0:i1, :]
            yield item.rowcolors(colname) + [rows]
    return partial(_colorfilter, sections)


def horizvol_colorfilter(sections=[]):
    '''The sections argument is a (starting_index, color_name) array.'''
    def _colorfilter(sections, item, datasrc, data):
        if not sections:
            yield item.rowcolors('neutral') + [data]
        for (i0,colname),(i1,_) in zip(sections, sections[1:]+[(None,'neutral')]):
            rows = data[:, i0:i1]
            yield item.rowcolors(colname) + [rows]
    return partial(_colorfilter, sections)


def candlestick_ochl(datasrc, draw_body=True, draw_shadow=True, candle_width=0.6, ax=None, colorfunc=price_colorfilter):
    ax = _create_plot(ax=ax, maximize=False)
    datasrc = _create_datasrc(ax, datasrc, ncols=5)
    datasrc.scale_cols = [3,4] # only hi+lo scales
    _set_datasrc(ax, datasrc)
    item = CandlestickItem(ax=ax, datasrc=datasrc, draw_body=draw_body, draw_shadow=draw_shadow, candle_width=candle_width, colorfunc=colorfunc, resamp='hilo')
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
    item = candlestick_ochl(datasrc, draw_shadow=False, candle_width=1, ax=ax, colorfunc=colorfunc)
    item.colors['bull_body'] = item.colors['bull_frame']
    item.update_data = partial(_update_data, None, step_adjust_renko_datasrc, item)
    item.update_gfx = partial(_update_gfx, item)
    global epoch_period
    epoch_period = (origdf.iloc[1,0] - origdf.iloc[0,0]) // int(1e9)
    return item


def volume_ocv(datasrc, candle_width=0.8, ax=None, colorfunc=volume_colorfilter):
    ax = _create_plot(ax=ax, maximize=False)
    datasrc = _create_datasrc(ax, datasrc, ncols=4)
    _adjust_volume_datasrc(datasrc)
    _set_datasrc(ax, datasrc)
    item = CandlestickItem(ax=ax, datasrc=datasrc, draw_body=True, draw_shadow=False, candle_width=candle_width, colorfunc=colorfunc, resamp='sum')
    _update_significants(ax, datasrc, force=True)
    item.colors['bull_body'] = item.colors['bull_frame']
    if colorfunc == volume_colorfilter: # assume normal volume plot
        item.colors['bull_frame'] = volume_bull_color
        item.colors['bull_body']  = volume_bull_body_color
        item.colors['bear_frame'] = volume_bear_color
        item.colors['bear_body']  = volume_bear_color
        ax.vb.v_zoom_baseline = 0
    else:
        item.colors['weak_bull_frame'] = brighten(volume_bull_color, 1.2)
        item.colors['weak_bull_body']  = brighten(volume_bull_color, 1.2)
    item.update_data = partial(_update_data, None, _adjust_volume_datasrc, item)
    item.update_gfx = partial(_update_gfx, item)
    ax.addItem(item)
    item.setZValue(-20)
    return item


def horiz_time_volume(datasrc, ax=None, **kwargs):
    '''Draws multiple fixed horizontal volumes. The input format is:
       [[time0, [(price0,volume0),(price1,volume1),...]], ...]

       This chart needs to be plot last, so it knows if it controls
       what time periods are shown, or if its using time already in
       place by another plot.'''
    # update handling default if necessary
    global max_zoom_points, right_margin_candles
    if max_zoom_points > 15:
        max_zoom_points = 4
    if right_margin_candles > 3:
        right_margin_candles = 1

    ax = _create_plot(ax=ax, maximize=False)
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
    '''Expensive function. Only use on small data sets. See HeatmapItem for kwargs. Input datasrc
       has x (time) in index or first column, y (price) as column names, and intensity (color) as
       cell values.'''
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


def bar(x, y=None, width=0.8, ax=None, colorfunc=strength_colorfilter, **kwargs):
    '''Bar plots are decoupled. Use volume_ocv() if you want a bar plot which relates to other time plots.'''
    global right_margin_candles, max_zoom_points
    right_margin_candles = 0
    max_zoom_points = min(max_zoom_points, 8)
    ax = _create_plot(ax=ax, maximize=False)
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
        y = y + log_plot_offset
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
            axm.legend.addItem(item, name=legend)
        for _,label in axm.legend.items:
            if label.text == legend:
                label.setAttr('justify', 'left')
                label.setText(label.text, color=legend_text_color)
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
    return [Live() for _ in range(plots)]


def add_legend(text, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    _create_legend(ax)
    row = ax.legend.layout.rowCount()
    label = pg.LabelItem(text, color=legend_text_color, justify='left')
    ax.legend.layout.addItem(label, row, 0, 1, 2)
    return label


def fill_between(plot0, plot1, color=None):
    used_color = brighten(_get_color(plot0.ax, None, color), 1.3)
    item = pg.FillBetweenItem(plot0, plot1, brush=pg.mkBrush(used_color))
    item.ax = plot0.ax
    item.setZValue(-40)
    item.ax.addItem(item)
    # Ugly bug fix for PyQtGraph bug where downsampled/clipped plots are used in conjunction
    # with fill between. The reason is that the curves of the downsampled plots are only
    # calculated when shown, but not when added to the axis. We fix by saying the plot is
    # changed every time the zoom is changed - including initial load.
    def update_fill(vb):
        plot0.sigPlotChanged.emit(plot0)
    plot0.ax.vb.zoom_listeners.add(update_fill)
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


def add_band(y0, y1, color=band_color, ax=None):
    print('add_band() is deprecated, use add_horizontal_band() instead.')
    return add_horizontal_band(y0, y1, color, ax)


def add_horizontal_band(y0, y1, color=band_color, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    color = _get_color(ax, None, color)
    ix = ax.vb.yscale.invxform
    lr = pg.LinearRegionItem([ix(y0),ix(y1)], orientation=pg.LinearRegionItem.Horizontal, brush=pg.mkBrush(color), movable=False)
    lr.lines[0].setPen(pg.mkPen(None))
    lr.lines[1].setPen(pg.mkPen(None))
    lr.setZValue(-50)
    lr.ax = ax
    ax.addItem(lr)
    return lr


def add_vertical_band(x0, x1, color=band_color, ax=None):
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


def add_rect(p0, p1, color=band_color, interactive=False, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    x_pts = _pdtime2index(ax, pd.Series([p0[0], p1[0]]))
    ix = ax.vb.yscale.invxform
    y0,y1 = sorted([p0[1], p1[1]])
    pos  = (x_pts[0], ix(y0))
    size = (x_pts[1]-pos[0], ix(y1)-ix(y0))
    rect = FinRect(ax=ax, brush=pg.mkBrush(color), pos=pos, size=size, movable=interactive, resizable=interactive, rotatable=False)
    rect.setZValue(-40)
    if interactive:
        ax.vb.rois.append(rect)
    rect.ax = ax
    ax.addItem(rect)
    return rect


def add_line(p0, p1, color=draw_line_color, width=1, style=None, interactive=False, ax=None):
    ax = _create_plot(ax=ax, maximize=False)
    used_color = _get_color(ax, style, color)
    pen = _makepen(color=used_color, style=style, width=width)
    x_pts = _pdtime2index(ax, pd.Series([p0[0], p1[0]]))
    ix = ax.vb.yscale.invxform
    pts = [(x_pts[0], ix(p0[1])), (x_pts[1], ix(p1[1]))]
    if interactive:
        line = _create_poly_line(ax.vb, pts, closed=False, pen=pen, movable=False)
        ax.vb.rois.append(line)
    else:
        line = FinLine(pts, pen=pen)
    line.ax = ax
    ax.addItem(line)
    return line


def add_text(pos, s, color=draw_line_color, anchor=(0,0), ax=None):
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


def remove_primitive(primitive):
    ax = primitive.ax
    ax.removeItem(primitive)
    if primitive in ax.vb.rois:
        ax.vb.rois.remove(primitive)
    if hasattr(primitive, 'texts'):
        for txt in primitive.texts:
            ax.vb.removeItem(txt)


def set_mouse_callback(callback, ax=None, when='click'):
    '''Callback when clicked like so: callback(x, y).'''
    ax = ax if ax else last_ax
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
    '''Callback when crosshair updated like so: info(ax,x,y,xtext,ytext); the info()
       callback must return two values: xtext and ytext.'''
    ax = _create_plot(ax=ax, maximize=False)
    ax.crosshair.infos.append(infofunc)


def timer_callback(update_func, seconds, single_shot=False):
    global timers
    timer = QtCore.QTimer()
    timer.timeout.connect(update_func)
    if single_shot:
        timer.setSingleShot(True)
    timer.start(int(seconds*1000))
    timers.append(timer)
    return timer


def autoviewrestore(enable=True):
    '''Restor functionality saves view zoom coordinates when closing a window, and
       load them when creating the plot (with the same name) again.'''
    global viewrestore
    viewrestore = enable


def refresh():
    for win in windows:
        axs = win.axs + [ax for ax in overlay_axs if ax.vb.win==win]
        for ax in axs:
            _improve_significants(ax)
        vbs = [ax.vb for ax in axs]
        for vb in vbs:
            vb.pre_process_data()
        if viewrestore:
            if _loadwindata(win):
                continue
        _set_max_zoom(vbs)
        for vb in vbs:
            datasrc = vb.datasrc_or_standalone
            if datasrc and (vb.linkedView(0) is None or vb.linkedView(0).datasrc is None or vb.master_viewbox):
                vb.update_y_zoom(datasrc.init_x0, datasrc.init_x1)
    _repaint_candles()
    for md in master_data.values():
        for vb in md:
            if type(vb) != str: # ignore 'default'
                _mouse_moved(win, vb, None)


def show(qt_exec=True):
    refresh()
    for win in windows:
        if isinstance(win, FinWindow) or qt_exec:
            if win.show_maximized:
                win.showMaximized()
            else:
                win.show()
    if windows and qt_exec:
        global last_ax, app
        app = QtGui.QGuiApplication.instance()
        app.exec()
        windows.clear()
        overlay_axs.clear()
        _clear_timers()
        sounds.clear()
        master_data.clear()
        last_ax = None


def play_sound(filename):
    if filename not in sounds:
        from PyQt6.QtMultimedia import QSoundEffect
        s = sounds[filename] = QSoundEffect() # disallow gc
        s.setSource(QtCore.QUrl.fromLocalFile(filename))
    s = sounds[filename]
    s.play()


def screenshot(file, fmt='png'):
    if _internal_windows_only() and not app:
        print('ERROR: screenshot must be callbacked from e.g. timer_callback()')
        return False
    try:
        buffer = QtCore.QBuffer()
        app.primaryScreen().grabWindow(windows[0].winId()).save(buffer, fmt)
        file.write(buffer.data())
        return True
    except Exception as e:
        print('Screenshot error:', type(e), e)
    return False


def experiment(*args, **kwargs):
    if 'opengl' in args or kwargs.get('opengl'):
        try:
            # pip install PyOpenGL PyOpenGL-accelerate to get this going
            import OpenGL
            pg.setConfigOptions(useOpenGL=True, enableExperimental=True)
        except Exception as e:
            print('WARNING: OpenGL init error.', type(e), e)


#################### INTERNALS ####################


def _openfile(*args):
    return open(*args)


def _loadwindata(win):
    try:
        os.mkdir(os.path.expanduser('~/.finplot'))
    except:
        pass
    try:
        f = os.path.expanduser('~/.finplot/' + win.title.replace('/', '-') + '.ini')
        settings = [(k.strip(), literal_eval(v.strip())) for line in _openfile(f) for k, d, v in [line.partition('=')]
                    if v]
        if not settings:
            return
    except:
        return
    kvs = {k: v for k, v in settings}
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
    if not viewrestore:
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
                ## print('%s saved' % win.title)
    except Exception as e:
        print('Error saving plot:', e)


def _internal_windows_only():
    return all(isinstance(win, FinWindow) for win in windows)


def _create_plot(ax=None, **kwargs):
    if ax:
        return ax
    if last_ax:
        return last_ax
    return create_plot(**kwargs)


def _create_axis(pos, **kwargs):
    if pos == 'x':
        return EpochAxisItem(**kwargs)
    elif pos == 'y':
        return YAxisItem(**kwargs)


def _clear_timers():
    for timer in timers:
        timer.timeout.disconnect()
    timers.clear()


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
    if y_label_width:
        ax.axes['right']['item'].setWidth(
            y_label_width)  # this is to put all graphs on equal footing when texts vary from 0.4 to 2000000
    ax.axes['right']['item'].setStyle(
        tickLength=-5)  # some bug, totally unexplicable (why setting the default value again would fix repaint width as axis scale down)
    ax.axes['right']['item'].setZValue(30)  # put axis in front instead of behind data
    ax.axes['bottom']['item'].setZValue(30)
    ax.setLogMode(y=(yscale.scaletype == 'log'))
    ax.significant_forced = False
    ax.significant_decimals = significant_decimals
    ax.significant_eps = significant_eps
    ax.inverted = False
    ax.axos = []
    ax.crosshair = FinCrossHair(ax, color=cross_hair_color)
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
        viewbox.setBackgroundColor(odd_plot_background)
    viewbox.setParent(ax)
    return ax


def _ax_overlay(ax, scale=0.25, yaxis=False):
    '''The scale parameter defines how "high up" on the initial plot this overlay will show.
       The yaxis parameter can be one of [False, 'linear', 'log'].'''
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
        viewbox.setGeometry(ax.vb.sceneBoundingRect())

    axo = pg.PlotItem(enableMenu=False)
    axo.significant_forced = False
    axo.significant_decimals = significant_decimals
    axo.significant_eps = significant_eps
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
    ax.vb.sigResized.connect(updateView)
    overlay_axs.append(axo)
    ax.axos.append(axo)
    updateView()
    return axo


def _ax_set_visible(ax, crosshair=None, xaxis=None, yaxis=None, xgrid=None, ygrid=None):
    if crosshair == False:
        ax.crosshair.hide()
    if xaxis is not None:
        ax.getAxis('bottom').setStyle(showValues=xaxis)
    if yaxis is not None:
        ax.getAxis('right').setStyle(showValues=yaxis)
    if xgrid is not None or ygrid is not None:
        ax.showGrid(x=xgrid, y=ygrid, alpha=grid_alpha)
        if ax.getAxis('right'):
            ax.getAxis('right').setEnabled(False)
        if ax.getAxis('bottom'):
            ax.getAxis('bottom').setEnabled(False)


def _ax_decouple(ax):
    ax.setXLink(None)
    if ax.prev_ax:
        ax.prev_ax.set_visible(xaxis=True)


def _ax_disable_x_index(ax, decouple=True):
    ax.vb.x_indexed = False
    if decouple:
        _ax_decouple(ax)


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
        ax.legend = FinLegendItem(border_color=legend_border_color, fill_color=legend_fill_color, size=None,
                                  offset=(3, 2))
        ax.legend.setParentItem(ax.vb)


def _update_significants(ax, datasrc, force):
    # check if no epsilon set yet
    default_dec = 0.99 < ax.significant_decimals / significant_decimals < 1.01
    default_eps = 0.99 < ax.significant_eps / significant_eps < 1.01
    if force or (default_dec and default_eps):
        try:
            sd, se = datasrc.calc_significant_decimals(full=force)
            if sd or se != significant_eps:
                if (force and not ax.significant_forced) or default_dec or sd > ax.significant_decimals:
                    ax.significant_decimals = sd
                    ax.significant_forced |= force
                if (force and not ax.significant_forced) or default_eps or se < ax.significant_eps:
                    ax.significant_eps = se
                    ax.significant_forced |= force
        except:
            pass  # datasrc probably full av NaNs


def _improve_significants(ax):
    '''Force update of the EPS if we both have no bars/candles AND a log scale.
       This is intended to fix the lower part of the grid on line plots on a log scale.'''
    if ax.vb.yscale.scaletype == 'log':
        if not any(isinstance(item, CandlestickItem) for item in ax.items):
            _update_significants(ax, ax.vb.datasrc, force=True)


def _is_standalone(timeser):
    # more than N percent gaps or time reversals probably means this is a standalone plot
    return timeser.isnull().sum() + (timeser.diff() <= 0).sum() > len(timeser) * 0.1


def _create_poly_line(*args, **kwargs):
    return FinPolyLine(*args, **kwargs)


def _create_series(a):
    return a if isinstance(a, pd.Series) else pd.Series(a)


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
    global epoch_period
    if datasrc.timebased() and (epoch_period > 1e7 or not datasrc.standalone):
        ep_secs = datasrc.period_ns / 1e9
        epoch_period = ep_secs if ep_secs < epoch_period else epoch_period


def _has_timecol(df):
    return len(df.columns) >= 2


def _set_max_zoom(vbs):
    '''Set the relative allowed zoom level between axes groups, where the lowest-resolution
       plot in each group uses max_zoom_points, while the others get a scale >=1 of their
       respective highest zoom level.'''
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
    update_sigdig = False
    if not item.datasrc.standalone and not item.ax.vb.win._isMouseLeftDrag:
        # new limits when extending/reducing amount of data
        x_min, x1 = _set_x_limits(item.ax, item.datasrc)
        # scroll all plots if we're at the far right
        tr = item.ax.vb.targetRect()
        x0 = x1 - tr.width()
        if x0 < right_margin_candles + side_margin:
            x0 = -side_margin
        if tr.right() < x1 - 5 - 2 * right_margin_candles:
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
            y = y + log_plot_offset
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
    if x_indexed and init_steps:
        # initial zoom
        x0 = max(datasrc.xlen - init_steps, 0) - side_margin - extra_margin
        x1 = datasrc.xlen + right_margin_candles + side_margin + extra_margin
    elif x_indexed:
        # total x size for indexed data
        x0 = -side_margin - extra_margin
        x1 = datasrc.xlen + right_margin_candles - 1 + side_margin + extra_margin  # add another margin to get the "snap back" sensation
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
    '''Candles are only partially drawn, and therefore needs manual dirty reminder whenever it goes off-screen.'''
    axs = [ax for win in windows for ax in win.axs] + overlay_axs
    for ax in axs:
        for item in list(ax.items):
            if isinstance(item, FinPlotItem):
                _start_visual_update(item)
                _end_visual_update(item)


def _paint_scatter(item, p, *args):
    with np.errstate(invalid='ignore'):  # make pg's mask creation calls to numpy shut up
        item._dopaint(p, *args)


def _key_pressed(vb, ev):
    if ev.text() == 'g':  # grid
        global clamp_grid
        clamp_grid = not clamp_grid
        for win in windows:
            for ax in win.axs:
                ax.crosshair.update()
    elif ev.text() == 'i':  # invert y-axes
        for win in windows:
            for ax in win.axs:
                ax.invert_y()
    elif ev.text() == 'f':  # focus/expand active axis
        vb.parent().expand()
    elif ev.text() in ('\r', ' '):  # enter, space
        vb.set_draw_line_color(draw_done_color)
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
    elif ev.key() == QtCore.Qt.Key.Key_Escape and key_esc_close:
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
    md = master_data[master].get(vb) or master_data[master].get('default')
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


def _get_link_group(master, vb):
    '''return the viewboxes in the same linked group as the parameter one'''
    vbs = [ax.vb for ax in master.axs]
    vb_master_to_linked = defaultdict(set)
    for vb0 in vbs:
        vb_master = vb0.linkedView(0)
        if vb_master:
            vb_master_to_linked[vb_master].add(vb_master)
            vb_master_to_linked[vb_master].add(vb0)
            continue
        vb_master_to_linked[vb0].add(vb0)  # add master itself
    affected_vbs = vb_master_to_linked[vb.linkedView(0) or vb]
    return affected_vbs


def _wheel_event_wrapper(self, orig_func, ev):
    # scrolling on the border is simply annoying, pop in a couple of pixels to make sure
    d = QtCore.QPointF(-2, 0)
    ev = QtGui.QWheelEvent(ev.position() + d, ev.globalPosition() + d, ev.pixelDelta(), ev.angleDelta(), ev.buttons(),
                           ev.modifiers(), ev.phase(), False)
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
    except KeyError:  # when clicking beyond right_margin_candles
        if clamp_grid:
            t = ax.vb.datasrc.x.iloc[-1 if t > 0 else 0]
    try:
        callback(t, point.y())
    except OSError as e:
        pass
    except Exception as e:
        print('Inspection error:', type(e), e)


def brighten(color, f):
    if not color:
        return color
    return pg.mkColor(color).lighter(int(f * 100))


def _get_color(ax, style, wanted_color):
    if type(wanted_color) in (str, QtGui.QColor):
        return wanted_color
    index = wanted_color if type(wanted_color) == int else None
    is_line = lambda style: style is None or any(ch in style for ch in '-_.')
    get_handed_color = lambda item: item.opts.get('handed_color')
    this_line = is_line(style)
    if this_line:
        colors = soft_colors
    else:
        colors = hard_colors
    if index is None:
        avoid = set(i.opts['handed_color'] for i in ax.items if
                    isinstance(i, pg.PlotDataItem) and get_handed_color(i) is not None and this_line == is_line(
                        i.opts['symbol']))
        index = len([i for i in ax.items if
                     isinstance(i, pg.PlotDataItem) and get_handed_color(i) is None and this_line == is_line(
                         i.opts['symbol'])])
        while index in avoid:
            index += 1
    return colors[index % len(colors)]


def _pdtime2epoch(t):
    if isinstance(t, pd.Series):
        if isinstance(t.iloc[0], pd.Timestamp):
            dtype = str(t.dtype)
            if '[s' in dtype:
                return t.astype('int64') * int(1e9)
            elif '[ms' in dtype:
                return t.astype('int64') * int(1e6)
            elif '[us' in dtype:
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
    if ax.vb.datasrc is not None or not ax.vb.x_indexed:
        return ax.vb.datasrc
    vbs = [ax.vb for win in windows for ax in win.axs]
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
    if display_timezone == None:
        return _x2utc(datasrc, x)
    return _x2t(datasrc, x, lambda t: _millisecond_tz_wrap(
        datetime.fromtimestamp(t / 1e9, tz=display_timezone).strftime(timestamp_format)))


def _x2utc(datasrc, x):
    # using pd.to_datetime allow for pre-1970 dates
    return _x2t(datasrc, x, lambda t: pd.to_datetime(t, unit='ns').strftime(timestamp_format))


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
            if not truncate_timestamp:
                return s, True
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
        exp10 = floor(np.log10(abs(x)))
        x = x / (10 ** exp10)
        rm = int(abs(np.log10(rngmax))) if rngmax > 0 else 0
        sd = min(3, sd + rm)
        fmt = '%%%i.%ife%%i' % (sd, sd)
        r = fmt % (x, exp10)
    else:
        eps = fmod(x, significant_eps)
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
    if not clamp_grid:
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
            y -= fmod(y + eps2, eps) - eps2
    y = ax.vb.yscale.invxform(y, verify=True)
    return x, y


def _clamp_point(ax, p):
    if clamp_grid:
        x, y = _clamp_xy(ax, p.x(), p.y())
        return pg.Point(x, y)
    return p


def _comp_set_clamp_pos(set_pos_func, pos):
    if clamp_grid:
        if isinstance(pos, QtCore.QPointF):
            pos = pg.Point(int(pos.x()), pos.y())
        else:
            pos = int(pos)
    set_pos_func(pos)


def _set_clamp_pos(comp):
    comp.setPos = partial(_comp_set_clamp_pos, comp.setPos)


def _draw_line_segment_text(polyline, segment, pos0, pos1):
    fsecs = None
    datasrc = polyline.vb.datasrc
    if datasrc and clamp_grid:
        try:
            x0, x1 = pos0.x() + 0.5, pos1.x() + 0.5
            t0, _, _, _, cnt0 = datasrc.hilo(x0, x0)
            t1, _, _, _, cnt1 = datasrc.hilo(x1, x1)
            if cnt0 and cnt1:
                fsecs = abs(t1 - t0) / 1e9
        except:
            pass
    diff = pos1 - pos0
    if fsecs is None:
        fsecs = abs(diff.x() * epoch_period)
    secs = int(fsecs)
    mins = secs // 60
    hours = mins // 60
    mins = mins % 60
    secs = secs % 60
    if hours == 0 and mins == 0 and secs < 60 and epoch_period < 1:
        msecs = int((fsecs - int(fsecs)) * 1000)
        ts = '%0.2i:%0.2i.%0.3i' % (mins, secs, msecs)
    elif hours == 0 and mins < 60 and epoch_period < 60:
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
    '''Shows the proportions of this line height compared to the previous segment.'''
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
    return floor(v + 0.5)
