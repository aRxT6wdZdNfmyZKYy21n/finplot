import pyqtgraph as pg

from pyqtgraph import (
    QtCore
)

from finplot.constants import (
    FinPlotConstants
)

from finplot.item.fin_plot import (
    FinPlotItem
)

from finplot.internal_utils import (
    brighten
)


class CandlestickItem(FinPlotItem):
    def __init__(self, ax, datasrc, draw_body, draw_shadow, candle_width, colorfunc, resamp=None):
        self.colors = (
            dict(
                bull_shadow=FinPlotConstants.candle_bull_color,
                bull_frame=FinPlotConstants.candle_bull_color,
                bull_body=FinPlotConstants.candle_bull_body_color,
                bear_shadow=FinPlotConstants.candle_bear_color,
                bear_frame=FinPlotConstants.candle_bear_color,
                bear_body=FinPlotConstants.candle_bear_body_color,
                weak_bull_shadow=brighten(FinPlotConstants.candle_bull_color, 1.2),
                weak_bull_frame=brighten(FinPlotConstants.candle_bull_color, 1.2),
                weak_bull_body=brighten(FinPlotConstants.candle_bull_color, 1.2),
                weak_bear_shadow=brighten(FinPlotConstants.candle_bear_color, 1.5),
                weak_bear_frame=brighten(FinPlotConstants.candle_bear_color, 1.5),
                weak_bear_body=brighten(FinPlotConstants.candle_bear_color, 1.5)
            )
        )

        self.draw_body = draw_body
        self.draw_shadow = draw_shadow
        self.candle_width = candle_width
        self.shadow_width = FinPlotConstants.candle_shadow_width
        self.colorfunc = colorfunc
        self.resamp = resamp
        self.x_offset = 0
        super(
            CandlestickItem,
            self
        ).__init__(
            ax,
            datasrc,
            lod=True
        )

    def generate_picture(
            self,

            boundingRect
    ):
        left = boundingRect.left()
        right = boundingRect.right()

        p = self.painter

        df, origlen = (
            self.datasrc.rows(
                5,
                left,
                right,
                yscale=self.ax.vb.yscale,
                resamp=self.resamp
            )
        )

        f = origlen / len(df) if len(df) else 1
        w = self.candle_width * f
        w2 = w * 0.5

        for shadow, frame, body, df_rows in (
                self.colorfunc(
                    self,
                    self.datasrc,
                    df
                )
        ):
            idxs = df_rows.index
            rows = df_rows.values

            if self.x_offset:
                idxs += self.x_offset

            if self.draw_shadow:
                p.setPen(
                    pg.mkPen(
                        shadow,

                        width=(
                            self.shadow_width
                        )
                    )
                )

                for x, (t, open, close, high, low) in (
                        zip(
                            idxs,
                            rows
                        )
                ):
                    if high > low:
                        p.drawLine(
                            QtCore.QPointF(
                                x,
                                low
                            ),

                            QtCore.QPointF(
                                x,
                                high
                            )
                        )

            if self.draw_body:
                p.setPen(
                    pg.mkPen(
                        frame
                    )
                )

                p.setBrush(
                    pg.mkBrush(
                        body
                    )
                )

                for x, (t, open, close, high, low) in (
                        zip(
                            idxs,
                            rows
                        )
                ):
                    p.drawRect(
                        QtCore.QRectF(
                            x - w2,
                            open,
                            w,
                            close - open
                        )
                    )

    def rowcolors(
            self,
            prefix
    ):
        return [
            self.colors[
                prefix + '_shadow'
            ],
            self.colors[
                prefix + '_frame'
            ],
            self.colors[
                prefix + '_body'
            ]
        ]