import numpy as np
import pyqtgraph as pg

from math import ceil, floor


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
