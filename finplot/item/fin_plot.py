import pyqtgraph as pg

from pyqtgraph import (
    QtCore,
    QtGui
)

from finplot.constants import (
    FinPlotConstants
)


class FinPlotItem(pg.GraphicsObject):
    def __init__(self, ax, datasrc, lod):
        super(
            FinPlotItem,
            self
        ).__init__()

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
        if (
                self.dirty or

                (
                    self.lod and  # regenerate when zoom changes?

                    (
                        visibleRect.left() < self.cachedRect.left() or
                        visibleRect.right() > self.cachedRect.right() or

                        (
                            visibleRect.width() <

                            (
                                self.cachedRect.width() /
                                FinPlotConstants.cache_candle_factor
                            )
                        )
                    )
                )
        ):  # optimize when zooming in
            self._generate_picture(
                visibleRect
            )

    def _generate_picture(self, boundingRect):
        w = boundingRect.width()
        self.cachedRect = (
            QtCore.QRectF(
                (
                    boundingRect.left() -

                    (
                        FinPlotConstants.cache_candle_factor -
                        1
                    ) *

                    0.5 *
                    w
                ),
                0,

                (
                    FinPlotConstants.cache_candle_factor *
                    w
                ),

                0
            )
        )
        self.painter.begin(
            self.picture
        )
        self._generate_dummy_picture(self.viewRect())
        self.generate_picture(self.cachedRect)
        self.painter.end()
        self.dirty = False

    def _generate_dummy_picture(self, boundingRect):
        if self.datasrc.is_sparse:
            # just draw something to ensure PyQt will paint us again

            self.painter.setPen(
                pg.mkPen(
                    FinPlotConstants.background
                )
            )

            self.painter.setBrush(
                pg.mkBrush(
                    FinPlotConstants.background
                )
            )

            l, r = boundingRect.left(), boundingRect.right()
            self.painter.drawRect(QtCore.QRectF(l, boundingRect.top(), 1e-3, boundingRect.height()*1e-5))
            self.painter.drawRect(QtCore.QRectF(r, boundingRect.bottom(), -1e-3, -boundingRect.height()*1e-5))

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())
