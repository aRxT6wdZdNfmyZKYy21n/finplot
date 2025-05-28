import pyqtgraph as pg

from dateutil.tz import tzlocal


# appropriate types
ColorMap = pg.ColorMap


class FinPlotConstants(object):
    # module definitions, mostly colors
    background = '#131722'  # '#fff'
    foreground = '#b2b5be'  # '#000'

    legend_border_color = foreground  # '#777'
    legend_fill_color = background  # '#666a'
    legend_text_color = foreground  # '#ddd6'
    odd_plot_background = background  # '#eaeaea'
    soft_colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf'
    ]
    hard_colors = [
        '#000000',
        '#772211',
        '#000066',
        '#555555',
        '#0022cc',
        '#ffcc00'
    ]

    colmap_clash = (
        ColorMap(
            [0.0, 0.2, 0.6, 1.0],
            [
                [127, 127, 255, 51],
                [0, 0, 127, 51],
                [255, 51, 102, 51],
                [255, 178, 76, 51]
            ]
        )
    )

    grid_alpha = 0.2  # 0.06
    candle_bull_color = '#26a69a'
    candle_bear_color = '#ef5350'
    candle_bull_body_color = background
    candle_bear_body_color = candle_bear_color
    candle_shadow_width = 1
    volume_bull_color = '#92d2cc'
    volume_bear_color = '#f7a9a7'
    volume_bull_body_color = volume_bull_color
    volume_neutral_color = '#bbb'
    poc_color = '#006'
    band_color = '#7e57c219'  # '#d2dfe6'
    cross_hair_color = '#9598a1'  # '#0007'
    draw_line_color = foreground  # '#000'
    draw_done_color = '#555'
    significant_decimals = 8
    significant_eps = 1e-8
    max_decimals = 10
    axis_height_factor = {0: 2}
    side_margin = 0.5
    lod_candles = 3000
    lod_labels = 700
    cache_candle_factor = 3  # factor extra candles rendered to buffer
    y_pad = 0.03  # 3% padding at top and bottom of autozoom plots
    y_label_width = 65
    timestamp_format = '%Y-%m-%d %H:%M:%S.%f'
    display_timezone = tzlocal()  # default to local
    truncate_timestamp = True
    win_recreate_delta = 30
    log_plot_offset = -2.2222222e-16  # I could file a bug report, probably in PyQt, but this is more fun
    # format: mode, min-duration, pd-freq-fmt, tick-str-len
    time_splits = [
        ('years', 2 * 365 * 24 * 60 * 60,    'YS',  4),
        ('months', 3 * 30 * 24 * 60 * 60,    'MS', 10),
        ('weeks',   3 * 7 * 24 * 60 * 60, 'W-MON', 10),
        ('days',        3 * 24 * 60 * 60,     'D', 10),
        ('hours',            9 * 60 * 60,    '3h', 16),
        ('hours',            3 * 60 * 60,     'h', 16),
        ('minutes',              45 * 60, '15min', 16),
        ('minutes',              15 * 60,  '5min', 16),
        ('minutes',               3 * 60,   'min', 16),
        ('seconds',                   45,   '15s', 19),
        ('seconds',                   15,    '5s', 19),
        ('seconds',                    3,     's', 19),
        ('milliseconds',               0,    'ms', 23)
    ]

    key_esc_close = True  # ESC key closes window

    # pick up win resolution
    try:
        import ctypes

        user32 = (
            ctypes.windll.user32
        )

        user32.SetProcessDPIAware()

        first_system_metrics_value = (
            user32.GetSystemMetrics(
                0
            )
        )

        lod_candles = (
            int(
                first_system_metrics_value *
                1.6
            )
        )

        # 2560 and resolutions above -> wider shadows
        candle_shadow_width = (
            int(
                (
                    first_system_metrics_value //
                    2100
                ) +

                1
            )
        )
    except:
        pass
