from math import ceil, floor
import numpy as np
import pandas as pd
from pyqtgraph import QtCore, QtGui
from finplot.constants import *  # TODO
from finplot.utils import (
    brighten,
    horizvol_colorfilter,
    _x2year,
    _x2local_t,
    _is_str_midnight,
    _get_datasrc,
    _pdtime2index,
    _makepen,
)


class EpochAxisItem(pg.AxisItem):
    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb

    def tickStrings(self, values, scale, spacing):
        if self.mode == 'num':
            return ['%g'%v for v in values]
        conv = _x2year if self.mode=='years' else _x2local_t
        strs = [conv(self.vb.datasrc, value)[0] for value in values]
        if all(_is_str_midnight(s) for s in strs if s): # all at midnight -> round to days
            strs = [s.partition(' ')[0] for s in strs]
        return strs

    def tickValues(self, minVal, maxVal, size):
        self.mode = 'num'
        ax = self.vb.parent()
        datasrc = _get_datasrc(ax, require=False)
        if datasrc is None or not self.vb.x_indexed:
            return super().tickValues(minVal, maxVal, size)
        # calculate if we use years, days, etc.
        t0,t1,_,_,_ = datasrc.hilo(minVal, maxVal)
        t0,t1 = pd.to_datetime(t0), pd.to_datetime(t1)
        dts = (t1-t0).total_seconds()
        gfx_width = int(size)
        for mode, dtt, freq, ticklen in time_splits:
            if dts > dtt:
                self.mode = mode
                desired_ticks = gfx_width / ((ticklen+2) * 10) - 1 # an approximation is fine
                if self.vb.datasrc is not None and not self.vb.datasrc.is_smooth_time():
                    desired_ticks -= 1 # leave more space for unevenly spaced ticks
                desired_ticks = max(desired_ticks, 4)
                to_midnight = freq in ('YS', 'MS', 'W-MON', 'D')
                tz = display_timezone if to_midnight else None # for shorter timeframes, timezone seems buggy
                rng = pd.date_range(t0, t1, tz=tz, normalize=to_midnight, freq=freq)
                steps = len(rng) if len(rng)&1==0 else len(rng)+1 # reduce jitter between e.g. 5<-->10 ticks for resolution close to limit
                step = int(steps/desired_ticks) or 1
                rng = rng[::step]
                if not to_midnight:
                    try:    rng = rng.round(freq=freq)
                    except: pass
                ax = self.vb.parent()
                rng = _pdtime2index(ax=ax, ts=pd.Series(rng), require_time=True)
                indices = [ceil(i) for i in rng if i>-1e200]
                return [(0, indices)]
        return [(0,[])]

    def generateDrawSpecs(self, p):
        specs = super().generateDrawSpecs(p)
        if specs:
            if not self.style['showValues']:
                pen,p0,p1 = specs[0] # axis specs
                specs = [(_makepen('#fff0'),p0,p1)] + list(specs[1:]) # don't draw axis if hiding values
            else:
                # throw out ticks that are out of bounds
                text_specs = specs[2]
                if len(text_specs) >= 4:
                    rect,flags,text = text_specs[0]
                    if rect.left() < 0:
                        del text_specs[0]
                    rect,flags,text = text_specs[-1]
                    if rect.right() > self.geometry().width():
                        del text_specs[-1]
                # ... and those that overlap
                x = 1e6
                for i,(rect,flags,text) in reversed(list(enumerate(text_specs))):
                    if rect.right() >= x:
                        del text_specs[i]
                    else:
                        x = rect.left()
        return specs


class YAxisItem(pg.AxisItem):
    def __init__(self, vb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vb = vb
        self.hide_strings = False
        self.style['autoExpandTextSpace'] = False
        self.style['autoReduceTextSpace'] = False
        self.next_fmt = '%g'

    def tickValues(self, minVal, maxVal, size):
        vs = super().tickValues(minVal, maxVal, size)
        if len(vs) < 3:
            return vs
        return self.fmt_values(vs)

    def logTickValues(self, minVal, maxVal, size, stdTicks):
        v1 = int(floor(minVal))
        v2 = int(ceil(maxVal))
        minor = []
        for v in range(v1, v2):
            minor.extend([v+l for l in np.log10(np.linspace(1, 9.9, 90))])
        minor = [x for x in minor if x>minVal and x<maxVal]
        if not minor:
            minor.extend(np.geomspace(minVal, maxVal, 7)[1:-1])
        if len(minor) > 10:
            minor = minor[::len(minor)//5]
        vs = [(None, minor)]
        return self.fmt_values(vs)

    def tickStrings(self, values, scale, spacing):
        if self.hide_strings:
            return []
        xform = self.vb.yscale.xform
        return [self.next_fmt%xform(value) for value in values]

    def fmt_values(self, vs):
        xform = self.vb.yscale.xform
        gs = ['%g'%xform(v) for v in vs[-1][1]]
        if not gs:
            return vs
        if any(['e' in g for g in gs]):
            maxdec = max([len((g).partition('.')[2].partition('e')[0]) for g in gs if 'e' in g])
            self.next_fmt = '%%.%ie' % maxdec
        elif gs:
            maxdec = max([len((g).partition('.')[2]) for g in gs])
            self.next_fmt = '%%.%if' % maxdec
        else:
            self.next_fmt = '%g'
        return vs


class FinLegendItem(pg.LegendItem):
    def __init__(self, border_color, fill_color, **kwargs):
        super().__init__(**kwargs)
        self.layout.setVerticalSpacing(2)
        self.layout.setHorizontalSpacing(20)
        self.layout.setContentsMargins(2, 2, 10, 2)
        self.border_color = border_color
        self.fill_color = fill_color

    def paint(self, p, *args):
        p.setPen(pg.mkPen(self.border_color))
        p.setBrush(pg.mkBrush(self.fill_color))
        p.drawRect(self.boundingRect())


class FinPlotItem(pg.GraphicsObject):
    def __init__(self, ax, datasrc, lod):
        super().__init__()
        self.ax = ax
        self.datasrc = datasrc
        self.picture = QtGui.QPicture()
        self.painter = QtGui.QPainter()
        self.dirty = True
        self.lod = lod
        self.cachedRect = None

    def repaint(self):
        self.dirty = True
        self.paint(None)

    def paint(self, p, *args):
        if self.datasrc.is_sparse:
            self.dirty = True
        self.update_dirty_picture(self.viewRect())
        if p is not None:
            p.drawPicture(0, 0, self.picture)

    def update_dirty_picture(self, visibleRect):
        if self.dirty or \
            (self.lod and # regenerate when zoom changes?
                (visibleRect.left() < self.cachedRect.left() or \
                 visibleRect.right() > self.cachedRect.right() or \
                 visibleRect.width() < self.cachedRect.width() / cache_candle_factor)): # optimize when zooming in
            self._generate_picture(visibleRect)

    def _generate_picture(self, boundingRect):
        w = boundingRect.width()
        self.cachedRect = QtCore.QRectF(boundingRect.left()-(cache_candle_factor-1)*0.5*w, 0, cache_candle_factor*w, 0)
        self.painter.begin(self.picture)
        self._generate_dummy_picture(self.viewRect())
        self.generate_picture(self.cachedRect)
        self.painter.end()
        self.dirty = False

    def _generate_dummy_picture(self, boundingRect):
        if self.datasrc.is_sparse:
            # just draw something to ensure PyQt will paint us again
            self.painter.setPen(pg.mkPen(background))
            self.painter.setBrush(pg.mkBrush(background))
            l,r = boundingRect.left(), boundingRect.right()
            self.painter.drawRect(QtCore.QRectF(l, boundingRect.top(), 1e-3, boundingRect.height()*1e-5))
            self.painter.drawRect(QtCore.QRectF(r, boundingRect.bottom(), -1e-3, -boundingRect.height()*1e-5))

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class CandlestickItem(FinPlotItem):
    def __init__(self, ax, datasrc, draw_body, draw_shadow, candle_width, colorfunc, resamp=None):
        self.colors = dict(bull_shadow      = candle_bull_color,
                           bull_frame       = candle_bull_color,
                           bull_body        = candle_bull_body_color,
                           bear_shadow      = candle_bear_color,
                           bear_frame       = candle_bear_color,
                           bear_body        = candle_bear_body_color,
                           weak_bull_shadow = brighten(candle_bull_color, 1.2),
                           weak_bull_frame  = brighten(candle_bull_color, 1.2),
                           weak_bull_body   = brighten(candle_bull_color, 1.2),
                           weak_bear_shadow = brighten(candle_bear_color, 1.5),
                           weak_bear_frame  = brighten(candle_bear_color, 1.5),
                           weak_bear_body   = brighten(candle_bear_color, 1.5))
        self.draw_body = draw_body
        self.draw_shadow = draw_shadow
        self.candle_width = candle_width
        self.shadow_width = candle_shadow_width
        self.colorfunc = colorfunc
        self.resamp = resamp
        self.x_offset = 0
        super().__init__(ax, datasrc, lod=True)

    def generate_picture(self, boundingRect):
        left,right = boundingRect.left(), boundingRect.right()
        p = self.painter
        df,origlen = self.datasrc.rows(5, left, right, yscale=self.ax.vb.yscale, resamp=self.resamp)
        f = origlen / len(df) if len(df) else 1
        w = self.candle_width * f
        w2 = w * 0.5
        for shadow,frame,body,df_rows in self.colorfunc(self, self.datasrc, df):
            idxs = df_rows.index
            rows = df_rows.values
            if self.x_offset:
                idxs += self.x_offset
            if self.draw_shadow:
                p.setPen(pg.mkPen(shadow, width=self.shadow_width))
                for x,(t,open,close,high,low) in zip(idxs, rows):
                    if high > low:
                        p.drawLine(QtCore.QPointF(x, low), QtCore.QPointF(x, high))
            if self.draw_body:
                p.setPen(pg.mkPen(frame))
                p.setBrush(pg.mkBrush(body))
                for x,(t,open,close,high,low) in zip(idxs, rows):
                    p.drawRect(QtCore.QRectF(x-w2, open, w, close-open))

    def rowcolors(self, prefix):
        return [self.colors[prefix+'_shadow'], self.colors[prefix+'_frame'], self.colors[prefix+'_body']]


class HeatmapItem(FinPlotItem):
    def __init__(self, ax, datasrc, rect_size=0.9, filter_limit=0, colmap=colmap_clash, whiteout=0.0, colcurve=lambda x:pow(x,4)):
        self.rect_size = rect_size
        self.filter_limit = filter_limit
        self.colmap = colmap
        self.whiteout = whiteout
        self.colcurve = colcurve
        self.col_data_end = len(datasrc.df.columns)
        super().__init__(ax, datasrc, lod=False)

    def generate_picture(self, boundingRect):
        prices = self.datasrc.df.columns[self.datasrc.col_data_offset:self.col_data_end]
        h0 = (prices[0] - prices[1]) * (1-self.rect_size)
        h1 = (prices[0] - prices[1]) * (1-(1-self.rect_size)*2)
        rect_size2 = 0.5 * self.rect_size
        df = self.datasrc.df.iloc[:, self.datasrc.col_data_offset:self.col_data_end]
        values = df.values
        # normalize
        values -= np.nanmin(values)
        values = values / (np.nanmax(values) / (1+self.whiteout)) # overshoot for coloring
        lim = self.filter_limit * (1+self.whiteout)
        p = self.painter
        for t,row in enumerate(values):
            for ci,price in enumerate(prices):
                v = row[ci]
                if v >= lim:
                    v = 1 - self.colcurve(1 - (v-lim)/(1-lim))
                    color = self.colmap.map(v, mode='qcolor')
                    p.fillRect(QtCore.QRectF(t-rect_size2, self.ax.vb.yscale.invxform(price+h0), self.rect_size, self.ax.vb.yscale.invxform(h1)), color)


class HorizontalTimeVolumeItem(CandlestickItem):
    def __init__(self, ax, datasrc, candle_width=0.8, draw_va=0.0, draw_vaw=1.0, draw_body=0.4, draw_poc=0.0, colorfunc=None):
        '''A negative draw_body does not mean that the candle is drawn in the opposite direction (use negative volume for that),
           but instead that screen scale will be used instead of interval-relative scale.'''
        self.draw_va = draw_va
        self.draw_vaw = draw_vaw
        self.draw_poc = draw_poc
        ## self.col_data_end = len(datasrc.df.columns)
        colorfunc = colorfunc or horizvol_colorfilter() # resolve function lower down in source code
        super().__init__(ax, datasrc, draw_shadow=False, candle_width=candle_width, draw_body=draw_body, colorfunc=colorfunc)
        self.lod = False
        self.colors.update(dict(neutral_shadow  = volume_neutral_color,
                                neutral_frame   = volume_neutral_color,
                                neutral_body    = volume_neutral_color,
                                bull_body       = candle_bull_color))

    def generate_picture(self, boundingRect):
        times = self.datasrc.df.iloc[:, 0]
        vals = self.datasrc.df.values
        prices = vals[:, self.datasrc.col_data_offset::2]
        volumes = vals[:, self.datasrc.col_data_offset+1::2].T
        # normalize
        try:
            index = _pdtime2index(self.ax, times, require_time=True)
            index_steps = pd.Series(index).diff().shift(-1)
            index_steps[index_steps.index[-1]] = index_steps.median()
        except AssertionError:
            index = times
            index_steps = [1]*len(index)
        draw_body = self.draw_body
        wf = 1
        if draw_body < 0:
            wf = -draw_body * self.ax.vb.targetRect().width()
            draw_body = 1
        binc = len(volumes)
        if not binc:
            return
        divvol = np.nanmax(np.abs(volumes), axis=0)
        divvol[divvol==0] = 1
        volumes = (volumes * wf / divvol).T
        p = self.painter
        h = 1e-10
        for i in range(len(prices)):
            f = index_steps[i] * wf
            prcr = prices[i]
            prv = prcr[~np.isnan(prcr)]
            if len(prv) > 1:
                h = np.diff(prv).min()
            t = index[i]
            volr = np.nan_to_num(volumes[i])

            # calc poc
            pocidx = np.nanargmax(volr)

            # draw value area
            if self.draw_va:
                volrs = volr / np.nansum(volr)
                v = volrs[pocidx]
                a = b = pocidx
                while True:
                    if v >= self.draw_va - 1e-5:
                        break
                    aa = a - 1
                    bb = b + 1
                    va = volrs[aa] if aa>=0 else 0
                    vb = volrs[bb] if bb<binc else 0
                    if va >= vb: # NOTE both == is also ok
                        a = max(0, aa)
                        v += va
                    if va <= vb: # NOTE both == is also ok
                        b = min(binc-1, bb)
                        v += vb
                    if a==0 and b==binc-1:
                        break
                color = pg.mkColor(band_color)
                p.fillRect(QtCore.QRectF(t, prcr[a], f*self.draw_vaw, prcr[b]-prcr[a]+h), color)

            # draw horizontal bars
            if draw_body:
                h0 = h * (1-self.candle_width)/2
                h1 = h * self.candle_width
                for shadow,frame,body,data in self.colorfunc(self, self.datasrc, np.array([prcr, volr])):
                    p.setPen(pg.mkPen(frame))
                    p.setBrush(pg.mkBrush(body))
                    prcr_,volr_ = data
                    for w,y in zip(volr_, prcr_):
                        if abs(w) > 1e-15:
                            p.drawRect(QtCore.QRectF(t, y+h0, w*f*draw_body, h1))

            # draw poc line
            if self.draw_poc:
                y = prcr[pocidx] + h / 2
                p.setPen(pg.mkPen(poc_color))
                p.drawLine(QtCore.QPointF(t, y), QtCore.QPointF(t+f*self.draw_poc, y))


class ScatterLabelItem(FinPlotItem):
    def __init__(self, ax, datasrc, color, anchor):
        self.color = color
        self.text_items = {}
        self.anchor = anchor
        self.show = False
        super().__init__(ax, datasrc, lod=True)

    def generate_picture(self, bounding_rect):
        rows = self.getrows(bounding_rect)
        if len(rows) > lod_labels: # don't even generate when there's too many of them
            self.clear_items(list(self.text_items.keys()))
            return
        drops = set(self.text_items.keys())
        created = 0
        for x,t,y,txt in rows:
            txt = str(txt)
            ishtml = '<' in txt and '>' in txt
            key = '%s:%.8f' % (t, y)
            if key in self.text_items:
                item = self.text_items[key]
                (item.setHtml if ishtml else item.setText)(txt)
                item.setPos(x, y)
                drops.remove(key)
            else:
                kws = {'html':txt} if ishtml else {'text':txt}
                self.text_items[key] = item = pg.TextItem(color=self.color, anchor=self.anchor, **kws)
                item.setPos(x, y)
                item.setParentItem(self)
                created += 1
        if created > 0 or self.dirty: # only reduce cache if we've added some new or updated
            self.clear_items(drops)

    def clear_items(self, drop_keys):
        for key in drop_keys:
            item = self.text_items[key]
            item.scene().removeItem(item)
            del self.text_items[key]

    def getrows(self, bounding_rect):
        left,right = bounding_rect.left(), bounding_rect.right()
        df,_ = self.datasrc.rows(3, left, right, yscale=self.ax.vb.yscale, lod=False)
        rows = df.dropna()
        idxs = rows.index
        rows = rows.values
        rows = [(i,t,y,txt) for i,(t,y,txt) in zip(idxs, rows) if txt]
        return rows

    def boundingRect(self):
        return self.viewRect()