__all__ = (
    'g_fin_plot_global_state',
)

import typing

from collections import (
    defaultdict
)

from PyQt6.QtCore import (
    QCoreApplication,
    QTimer,
    QUrl
)

from PyQt6.QtMultimedia import (
    QSoundEffect
)


class FinPlotGlobalState(object):
    __slots__ = (
        '__app',
        '__epoch_period',
        '__clamp_grid',
        '__last_ax',
        '__master_data_by_window_map',
        '__max_zoom_points',
        '__overlay_axs',
        '__right_margin_candles',
        '__sound_effect_by_filename_map',
        '__timers',
        '__viewrestore',
        '__windows',
        '__winx',
        '__winy',
        '__winw',
        '__winh'
    )

    def __init__(self) -> None:
        super(
            FinPlotGlobalState,
            self
        ).__init__()

        self.__app: (
            typing.Optional[
                QCoreApplication
            ]
        ) = None

        self.__clamp_grid = True

        self.__epoch_period: (
            float
        ) = 1e30

        self.__last_ax: (
            typing.Optional[
                typing.Any
            ]
        ) = None  # always assume we want to plot in the last axis, unless explicitly specified

        self.__master_data_by_window_map: (
            typing.Dict[
                typing.Any,  # TODO
                typing.Dict[
                    typing.Any,  # TODO
                    typing.Any  # TODO
                ]
            ]
        ) = (
            defaultdict(
                dict
            )
        )

        self.__max_zoom_points = 20  # number of visible candles when maximum zoomed in

        self.__overlay_axs: (
            typing.List[
                typing.Any  # TODO
            ]
        ) = []  # for keeping track of candlesticks in overlays

        self.__right_margin_candles = 5  # whitespace at the right-hand side

        self.__sound_effect_by_filename_map: (
            typing.Dict[
                str,
                QSoundEffect
            ]
        ) = {}  # no gc

        self.__timers: (
            typing.List[
                QTimer
            ]
        ) = []  # no gc

        self.__viewrestore = False

        self.__windows: (
            typing.List[
                typing.Any  # TODO
            ]
        ) = []  # no gc

        self.__winx = 300
        self.__winy = 150
        self.__winw = 800
        self.__winh = 400

    def add_overlay_axo(
            self,

            overlay_axo: (
                typing.Any  # TODO
            )
    ) -> None:
        self.__overlay_axs.append(
            overlay_axo
        )

    def add_timer(
            self,

            timer: (
                QTimer
            )
    ) -> None:
        self.__timers.append(
            timer
        )

    def add_window(
            self,

            window: typing.Any  # TODO
    ) -> None:
        self.__windows.append(
            window
        )

    def clear_master_data_by_window_map(
            self
    ) -> None:
        self.__master_data_by_window_map.clear()

    def clear_overlay_axs(
            self
    ) -> None:
        self.__overlay_axs.clear()

    def clear_sound_effect_by_filename_map(
            self
    ) -> None:
        self.__sound_effect_by_filename_map.clear()

    def clear_timers(
            self
    ) -> None:
        timers = (
            self.__timers
        )

        for timer in (
                timers
        ):
            timer.timeout.disconnect()

        timers.clear()

    def clear_windows(
            self
    ) -> None:
        self.__windows.clear()

    def create_or_get_sound_effect(
            self,

            filename: str
    ) -> (
            QSoundEffect
    ):
        sound_effect_by_filename_map = (
            self.__sound_effect_by_filename_map
        )

        sound_effect = (
            sound_effect_by_filename_map.get(
                filename
            )
        )

        if sound_effect is None:
            sound_effect = (
                sound_effect_by_filename_map[
                    filename
                ]
            ) = (
                QSoundEffect()  # disallow gc
            )

            sound_effect.setSource(
                QUrl.fromLocalFile(
                    filename
                )
            )

        return sound_effect

    def get_app(
            self
    ) -> (
            typing.Optional[
                QCoreApplication
            ]
    ):
        return self.__app

    def get_clamp_grid(
            self
    ) -> bool:
        return self.__clamp_grid

    def get_epoch_period(
            self
    ) -> float:
        return self.__epoch_period

    def get_last_ax(self) -> (
            typing.Optional[
                typing.Any  # TODO
            ]
    ):
        return self.__last_ax

    def get_master_data_by_window_map(
            self
    ) -> (
            typing.Dict[
                typing.Any,  # TODO
                typing.Dict[
                    typing.Any,  # TODO
                    typing.Any   # TODO
                ]
            ]
    ):
        return (
            self.__master_data_by_window_map
        )

    def get_max_zoom_points(
            self
    ) -> int:
        return self.__max_zoom_points

    def get_overlay_axs(
            self
    ) -> (
            typing.List[
                typing.Any  # TODO
            ]
    ):
        return self.__overlay_axs

    def get_right_margin_candles(
            self
    ) -> int:
        return self.__right_margin_candles

    def get_viewrestore(self) -> int:
        return self.__viewrestore

    def get_window_master_data(
            self,

            window: typing.Any  # TODO
    ) -> (
            typing.Dict[
                typing.Any,  # TODO
                typing.Any   # TODO
            ]
    ):
        window_master_data = (
            self.__master_data_by_window_map.get(
                window
            )
        )

        if window_master_data is None:
            return {}

        return (
            window_master_data.copy()
        )

    def get_windows(
            self
    ) -> (
            typing.List[
                typing.Any  # TODO
            ]
    ):
        return self.__windows

    def get_winx(self) -> int:
        return self.__winx

    def get_winy(self) -> int:
        return self.__winy

    def get_winw(self) -> int:
        return self.__winw

    def get_winh(self) -> int:
        return self.__winh

    def set_app(
            self,

            value: (
                typing.Optional[
                    QCoreApplication
                ]
            )
    ) -> None:
        self.__app = value

    def set_clamp_grid(
            self,

            value: bool
    ) -> None:
        self.__clamp_grid = value

    def set_epoch_period(
            self,

            value: float
    ) -> None:
        self.__epoch_period = value

    def set_last_ax(
            self,

            value: (
                typing.Optional[
                    typing.Any  # TODO
                ]
            )
    ) -> None:
        self.__last_ax = value

    def set_max_zoom_points(
            self,

            value: int
    ) -> None:
        self.__max_zoom_points = value

    def set_right_margin_candles(
            self,

            value: int
    ) -> None:
        self.__right_margin_candles = value

    def set_viewrestore(
            self,

            value: bool
    ) -> None:
        self.__viewrestore = value

    def set_window_master_data(
            self,

            window: typing.Any,  # TODO

            window_master_data: (
                typing.Dict[
                    typing.Any,  # TODO
                    typing.Any   # TODO
                ]
            )
    ) -> None:
        (
            self.__master_data_by_window_map[
                window
            ]
        ) = (
            window_master_data
        )

    def set_winx(
            self,

            value: int
    ) -> None:
        self.__winx = value

    def set_winy(
            self,

            value: int
    ) -> None:
        self.__winy = value


g_fin_plot_global_state = (
    FinPlotGlobalState()
)
