import pyqtgraph as pg

from pyqtgraph import (
    QtCore
)

from .constants import (
    FinPlotConstants
)

from .global_state import (
    g_fin_plot_global_state
)

from .internal_utils import (
    _savewindata
)


class FinWindow(pg.GraphicsLayoutWidget):
    def __init__(self, title, **kwargs):
        self.title = title
        pg.mkQApp()

        super(
            FinWindow,
            self
        ).__init__(
            **kwargs
        )

        self.setWindowTitle(title)

        winx = (
            g_fin_plot_global_state.get_winx()
        )

        winy = (
            g_fin_plot_global_state.get_winy()
        )

        self.setGeometry(
            winx,
            winy,
            g_fin_plot_global_state.get_winw(),
            g_fin_plot_global_state.get_winh()
        )

        winx = (
            (
                winx +
                FinPlotConstants.win_recreate_delta
            ) %

            800
        )

        winy = (
            (
                winy +
                FinPlotConstants.win_recreate_delta
            ) %

            500
        )

        g_fin_plot_global_state.set_winx(
            winx
        )

        g_fin_plot_global_state.set_winy(
            winy
        )

        self.centralWidget.installEventFilter(self)
        self.ci.setContentsMargins(0, 0, 0, 0)
        self.ci.setSpacing(-1)
        self.closing = False

    @property
    def axs(self):
        return [ax for ax in self.ci.items if isinstance(ax, pg.PlotItem)]

    def autoRangeEnabled(self):
        return [True, True]

    def close(self):
        self.closing = True

        _savewindata(
            self
        )

        g_fin_plot_global_state.clear_timers()

        return (
            super(
                FinWindow,
                self
            ).close()
        )

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.WindowDeactivate:
            _savewindata(self)
        return False

    def resizeEvent(self, ev):
        """We resize and set the top Y axis larger according to the axis_height_factor.
           No point in trying to use the "row stretch factor" in Qt which is broken
           beyond repair."""
        if ev and not self.closing:
            axs = self.axs
            new_win_height = ev.size().height()
            old_win_height = ev.oldSize().height() if ev.oldSize().height() > 0 else new_win_height
            client_borders = old_win_height - sum(ax.vb.size().height() for ax in axs)
            client_borders = min(max(client_borders, 0), 30) # hrm
            new_height = new_win_height - client_borders
            for i,ax in enumerate(axs):
                j = (
                    FinPlotConstants.axis_height_factor.get(
                        i,
                        1
                    )
                )

                f = (
                    j /

                    (
                        len(
                            axs
                        ) +

                        sum(
                            FinPlotConstants.axis_height_factor.values()
                        ) -

                        len(
                            FinPlotConstants.axis_height_factor
                        )
                    )
                )

                ax.setMinimumSize(
                    (
                        100
                        if j > 1
                        else 50
                    ),

                    (
                        new_height *
                        f
                    )
                )

        return (
            super(
                FinWindow,
                self
            ).resizeEvent(
                ev
            )
        )

    def leaveEvent(self, ev):
        if not self.closing:
            super(
                FinWindow,
                self
            ).leaveEvent(
                ev
            )
