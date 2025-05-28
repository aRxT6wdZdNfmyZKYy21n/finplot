from functools import partial

import pyqtgraph as pg

from pyqtgraph import (
    QtCore
)

from .constants import (
    FinPlotConstants
)

from .internal_utils import (
    _clamp_point,
    _draw_line_segment_text,
    _roihandle_move_snap
)


class FinPolyLine(pg.PolyLineROI):
    def __init__(self, vb, *args, **kwargs):
        self.vb = vb  # init before parent constructor
        self.texts = []

        super(
            FinPolyLine,
            self
        ).__init__(
            *args,
            **kwargs
        )

    def addSegment(self, h1, h2, index=None):
        super(
            FinPolyLine,
            self
        ).addSegment(
            h1,
            h2,
            index
        )

        text = pg.TextItem(
            color=FinPlotConstants.draw_line_color
        )

        text.setZValue(50)
        text.segment = self.segments[-1 if index is None else index]

        if index is None:
            self.texts.append(text)
        else:
            self.texts.insert(index, text)

        self.update_text(
            text
        )

        self.vb.addItem(
            text,
            ignoreBounds=True
        )

    def removeSegment(self, seg):
        super(
            FinPolyLine,
            self
        ).removeSegment(
            seg
        )

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
        super(
            FinPolyLine,
            self
        ).movePoint(
            handle,
            pos,
            modifiers,
            finish,
            coords
        )

        self.update_texts()

    def segmentClicked(self, segment, ev=None, pos=None):
        pos = segment.mapToParent(ev.pos())
        pos = _clamp_point(self.vb.parent(), pos)

        super(
            FinPolyLine,
            self
        ).segmentClicked(
            segment,
            pos=pos
        )

        self.update_texts()

    def addHandle(self, info, index=None):
        handle = (
            super(
                FinPolyLine,
                self
            ).addHandle(
                info,
                index
            )
        )

        handle.movePoint = partial(_roihandle_move_snap, self.vb, handle.movePoint)
        return handle