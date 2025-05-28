from functools import partial

import numpy as np
import pandas as pd
import pyqtgraph as pg

from pyqtgraph import (
    QtCore
)

from finplot.constants import (
    FinPlotConstants
)

from finplot.internal_utils import (
    _pdtime2index
)

from finplot.item.candlestick import (
    CandlestickItem
)


def horizvol_colorfilter(sections=[]):
    """The sections argument is a (starting_index, color_name) array."""
    def _colorfilter(sections, item, datasrc, data):
        if not sections:
            yield item.rowcolors('neutral') + [data]
        for (i0,colname),(i1,_) in zip(sections, sections[1:]+[(None,'neutral')]):
            rows = data[:, i0:i1]
            yield item.rowcolors(colname) + [rows]
    return partial(_colorfilter, sections)


class HorizontalTimeVolumeItem(CandlestickItem):
    def __init__(self, ax, datasrc, candle_width=0.8, draw_va=0.0, draw_vaw=1.0, draw_body=0.4, draw_poc=0.0, colorfunc=None):
        """A negative draw_body does not mean that the candle is drawn in the opposite direction (use negative volume for that),
           but instead that screen scale will be used instead of interval-relative scale."""
        self.draw_va = draw_va
        self.draw_vaw = draw_vaw
        self.draw_poc = draw_poc

        # # self.col_data_end = len(datasrc.df.columns)

        colorfunc = colorfunc or horizvol_colorfilter()  # resolve function lower down in source code

        super(
            HorizontalTimeVolumeItem,
            self
        ).__init__(
            ax,
            datasrc,
            draw_shadow=False,
            candle_width=candle_width,
            draw_body=draw_body,
            colorfunc=colorfunc
        )

        self.lod = False

        self.colors.update(
            dict(
                neutral_shadow=FinPlotConstants.volume_neutral_color,
                neutral_frame=FinPlotConstants.volume_neutral_color,
                neutral_body=FinPlotConstants.volume_neutral_color,
                bull_body=FinPlotConstants.candle_bull_color
            )
        )

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

                color = (
                    pg.mkColor(
                        FinPlotConstants.band_color
                    )
                )

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

                p.setPen(
                    pg.mkPen(
                        FinPlotConstants.poc_color
                    )
                )

                p.drawLine(QtCore.QPointF(t, y), QtCore.QPointF(t+f*self.draw_poc, y))