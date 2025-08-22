import numpy as np
import pandas as pd

from datetime import datetime
from math import (
    floor,
    fmod,
)

from finplot.constants import (
    clamp_grid,
    display_timezone,
    truncate_timestamp,
    timestamp_format,
    epoch_period,
)


def clamp_xy(ax, x, y):
    if not clamp_grid:
        return x, y
    y = ax.vb.yscale.xform(y)
    # scale x
    if ax.vb.x_indexed:
        ds = ax.vb.datasrc
        if x < 0 or (ds and x > len(ds.df)-1):
            x = 0 if x < 0 else len(ds.df)-1
        x = round_(x)
    # scale y
    if y < 0.1 and ax.vb.yscale.scaletype == 'log':
        magnitude = int(3 - np.log10(y)) # round log to N decimals
        y = round(y, magnitude)
    else: # linear
        eps = ax.significant_eps
        if eps > 1e-8:
            eps2 = np.sign(y) * 0.5 * eps
            y -= fmod(y+eps2, eps) - eps2
    y = ax.vb.yscale.invxform(y, verify=True)
    return x, y


def round_(v):
    return floor(
        v+0.5,
    )


def round_to_significant(rng, rngmax, x, significant_decimals, significant_eps):
    is_highres = (rng/significant_eps > 1e2 and rngmax<1e-2) or abs(rngmax) > 1e7 or rng < 1e-5
    sd = significant_decimals
    if is_highres and abs(x)>0:
        exp10 = floor(np.log10(abs(x)))
        x = x / (10**exp10)
        rm = int(abs(np.log10(rngmax))) if rngmax>0 else 0
        sd = min(3, sd+rm)
        fmt = '%%%i.%ife%%i' % (sd, sd)
        r = fmt % (x, exp10)
    else:
        eps = fmod(x, significant_eps)
        if abs(eps) >= significant_eps/2:
            # round up
            eps -= np.sign(eps)*significant_eps
        xx = x - eps
        fmt = '%%%i.%if' % (sd, sd)
        r = fmt % xx
        if abs(x)>0 and rng<1e4 and r.startswith('0.0') and float(r[:-1]) == 0:
            r = '%.2e' % x
    return r


def millisecond_tz_wrap(s):
    if len(s) > 6 and s[-6] in '+-' and s[-3] == ':': # +01:00 fmt timezone present?
        s = s[:-6]
    return (s+'.000000') if '.' not in s else s


def x2utc(datasrc, x):
    # using pd.to_datetime allow for pre-1970 dates
    return x2t(datasrc, x, lambda t: pd.to_datetime(t, unit='ns').strftime(timestamp_format))


def x2t(datasrc, x, ts2str):
    if not datasrc:
        return '',False
    try:
        x += 0.5
        t,_,_,_,cnt = datasrc.hilo(x, x)
        if cnt:
            if not datasrc.timebased():
                return '%g' % t, False
            s = ts2str(t)
            if not truncate_timestamp:
                return s,True
            if epoch_period >= 23*60*60: # daylight savings, leap seconds, etc
                i = s.index(' ')
            elif epoch_period >= 59: # consider leap seconds
                i = s.rindex(':')
            elif epoch_period >= 1:
                i = s.index('.') if '.' in s else len(s)
            elif epoch_period >= 0.001:
                i = -3
            else:
                i = len(s)
            return s[:i],True
    except Exception as e:
        import traceback
        traceback.print_exc()
    return '',datasrc.timebased()


def x2local_t(datasrc, x):
    if display_timezone == None:
        return x2utc(datasrc, x)
    return x2t(datasrc, x, lambda t: millisecond_tz_wrap(datetime.fromtimestamp(t/1e9, tz=display_timezone).strftime(timestamp_format)))
