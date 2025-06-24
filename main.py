import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import uhd

# Калибровочная таблица: частота (Гц) -> поправка (dB)
calibration_table = {
    400e6: -77,
    800e6: -77,
    1500e6: -77,
    3000e6: -77,
    5000e6: -77
}


def get_calibration(freq_hz):
    freqs = np.array(list(calibration_table.keys()))
    gains = np.array(list(calibration_table.values()))
    return np.interp(freq_hz, freqs, gains)


# Поток для непрерывной TX-передачи
class TXThread(QtCore.QThread):
    def __init__(self, tx_stream, tx_buffer, parent=None):
        super().__init__(parent)
        self.tx_stream = tx_stream
        self.tx_buffer = tx_buffer
        self.running = True

    def run(self):
        md = uhd.types.TXMetadata()
        # Для непрерывной передачи не устанавливаем флаги начального/конечного пакета
        while self.running:
            try:
                # Передаем tx_buffer без указания длины массива (length не нужен)
                self.tx_stream.send(self.tx_buffer, md, timeout=0.1)
            except Exception as e:
                print("TX send error:", e)

    def stop(self):
        self.running = False
        self.wait()


class SpectrumAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("USRP Wideband Spectrum Analyzer")

        # TX-параметры
        self.tx_enabled = False
        self.tx_stream = None
        self.tx_thread = None
        self.tx_buffer = None

        # Главный виджет: слева графики, справа панель управления
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        ### Левая сторона – графики
        graph_container = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(graph_container)

        # Основной спектральный график (RX)
        self.graph_plot = pg.PlotWidget(title="Spectrum in dBm")
        self.graph_plot.setLabel('left', 'Power', units='dBm')
        self.graph_plot.setLabel('bottom', 'Frequency', units='MHz')
        self.graph_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.curve = self.graph_plot.plot(pen='y')  # нормальный (RX) спектр, желтый
        self.curve.setDownsampling(auto=False)
        graph_layout.addWidget(self.graph_plot)

        # Интерактивный курсор (перекрестье + текст)
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('c'))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('c'))
        self.graph_plot.addItem(self.vLine, ignoreBounds=True)
        self.graph_plot.addItem(self.hLine, ignoreBounds=True)
        self.cursorLabel = pg.TextItem("", anchor=(1, 1), fill=pg.mkBrush(0, 0, 0, 150))
        self.graph_plot.addItem(self.cursorLabel)
        self.proxy = pg.SignalProxy(self.graph_plot.scene().sigMouseMoved,
                                    rateLimit=60, slot=self.on_mouse_moved)
        self.current_x = None
        self.current_y = None

        # Отдельная кривая для режима maxHold (RX) – красная
        self.maxhold_curve = self.graph_plot.plot(pen='r')
        self.maxhold_enabled = False
        # Маркер пика (рассчитывается по нормальному спектру)
        self.max_marker = self.graph_plot.plot(symbol='o', symbolBrush='r', symbolSize=10)
        self.max_text = pg.TextItem(color='w', anchor=(0, 1))
        self.graph_plot.addItem(self.max_text)

        # График водопада (RX)
        self.waterfall_plot = pg.PlotWidget(title="Waterfall")
        self.waterfall_plot.setLabel('bottom', 'Frequency', units='MHz')
        self.waterfall_plot.setLabel('left', 'Scan')
        self.waterfall_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.waterfall_img = pg.ImageItem()
        self.waterfall_img.setOpts(invertY=False, axisOrder='row-major')
        lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        self.waterfall_img.setLookupTable(lut)
        self.waterfall_plot.addItem(self.waterfall_img)
        # Статичная область водопада задаётся в init_live_scan_parameters
        self.waterfall_plot.getViewBox().invertY(True)
        graph_layout.addWidget(self.waterfall_plot)

        main_layout.addWidget(graph_container, stretch=3)

        ### Правая сторона – панель управления
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QFormLayout(control_panel)
        # Настройки для приемного канала (RX)
        self.start_freq_edit = QtWidgets.QLineEdit("70")  # MHz
        self.stop_freq_edit = QtWidgets.QLineEdit("6000")  # MHz
        self.step_edit = QtWidgets.QLineEdit("56")  # MHz
        control_layout.addRow("Start (MHz):", self.start_freq_edit)
        control_layout.addRow("Stop (MHz):", self.stop_freq_edit)
        control_layout.addRow("Step (MHz):", self.step_edit)

        self.samples_combo = QtWidgets.QComboBox()
        for v in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            self.samples_combo.addItem(str(v))
        self.samples_combo.setCurrentText("4096")
        control_layout.addRow("FFT size:", self.samples_combo)

        # self.samp_rate_edit = QtWidgets.QLineEdit("56")  # MHz
        # control_layout.addRow("Samp_rate:", self.start_freq_edit)

        self.gain_spin = QtWidgets.QSpinBox()
        self.gain_spin.setRange(0, 60)
        self.gain_spin.setValue(30)
        control_layout.addRow("Gain:", self.gain_spin)

        self.waterfall_lines_spin = QtWidgets.QSpinBox()
        self.waterfall_lines_spin.setRange(10, 2000)
        self.waterfall_lines_spin.setValue(200)
        control_layout.addRow("Waterfall size:", self.waterfall_lines_spin)

        self.scan_button = QtWidgets.QPushButton("Start scanning")
        control_layout.addRow(self.scan_button)
        self.maxhold_button = QtWidgets.QPushButton("Turn Max Hold On")
        control_layout.addRow(self.maxhold_button)

        # Разделительная линия для настройки передатчика (TX)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        control_layout.addRow(separator)

        # Настройки передающего канала
        self.tx_freq_edit = QtWidgets.QLineEdit("2400")  # MHz – значение по умолчанию
        control_layout.addRow("Tx Frequency (MHz):", self.tx_freq_edit)
        self.tx_gain_spin = QtWidgets.QSpinBox()
        self.tx_gain_spin.setRange(0, 60)
        self.tx_gain_spin.setValue(30)
        control_layout.addRow("Gain Tx:", self.tx_gain_spin)
        self.tx_start_button = QtWidgets.QPushButton("Start trasmission")
        control_layout.addRow(self.tx_start_button)
        self.tx_status_label = QtWidgets.QLabel("")
        control_layout.addRow("Status:", self.tx_status_label)

        # Разделительная линия для настройки sweep режима
        separator_sweep = QtWidgets.QFrame()
        separator_sweep.setFrameShape(QtWidgets.QFrame.HLine)
        separator_sweep.setFrameShadow(QtWidgets.QFrame.Sunken)
        control_layout.addRow(separator_sweep)

        # Настройки sweep режима
        self.sweep_start_edit = QtWidgets.QLineEdit("1000")  # MHz
        self.sweep_stop_edit = QtWidgets.QLineEdit("2000")  # MHz
        self.sweep_step_edit = QtWidgets.QLineEdit("10")  # MHz
        self.sweep_dwell_edit = QtWidgets.QLineEdit("100")  # ms
        control_layout.addRow("Sweep start (MHz):", self.sweep_start_edit)
        control_layout.addRow("Sweep stop (MHz):", self.sweep_stop_edit)
        control_layout.addRow("Sweep step (MHz):", self.sweep_step_edit)
        control_layout.addRow("Time at each step (ms):", self.sweep_dwell_edit)

        self.sweep_gain_spin = QtWidgets.QSpinBox()
        self.sweep_gain_spin.setRange(0, 60)
        self.sweep_gain_spin.setValue(30)
        control_layout.addRow("Gain Sweep:", self.sweep_gain_spin)

        self.sweep_start_button = QtWidgets.QPushButton("Start Sweep")
        control_layout.addRow(self.sweep_start_button)
        self.sweep_status_label = QtWidgets.QLabel("")
        control_layout.addRow("Status Sweep:", self.sweep_status_label)

        main_layout.addWidget(control_panel, stretch=1)

        # Привязываем кнопки
        self.scan_button.clicked.connect(self.toggle_live_scanning)
        self.maxhold_button.clicked.connect(self.toggle_maxhold)
        self.tx_start_button.clicked.connect(self.start_transmission)
        self.sweep_start_button.clicked.connect(self.toggle_sweep_transmission)

        # Параметры сканирования и водопада (RX)
        self.sample_rate = float(self.step_edit.text())*1000000  # Гц
        self.num_samples = 4096  # значение по умолчанию (пересчитывается из samples_combo)
        self.waterfall_history = 200  # обновляется из waterfall_lines_spin при старте
        self.waterfall_data = np.full((self.waterfall_history, self.num_samples), -140.0)
        self.waterfall_index = 0
        self.waterfall_img.setImage(self.waterfall_data, autoLevels=False, levels=(-140, -30))

        # Переменные для композитного (wideband) сканирования (RX)
        self.wb_centers = None  # список центральных частот (в Гц)
        self.wb_index = 0
        self.composite_spectrum = None  # нормальный композитный спектр (в dBm)
        self.maxhold_data_arr = None  # массив для режима maxHold (в dBm)
        self.common_freq = None  # общая фиксированная ось (в MHz)

        # Переменные для sweep режима (TX)
        self.sweep_enabled = False
        self.sweep_timer = QtCore.QTimer()
        self.sweep_timer.timeout.connect(self.next_sweep_step)
        self.current_sweep_freq = None
        self.sweep_freqs = []
        self.live_scanning = False
        self.init_usrp()

    def init_usrp(self):
        self.usrp = uhd.usrp.MultiUSRP()
        channel = 0
        self.center_freq = 1000e6  # начальное значение (RX)
        self.gain = 30
        self.usrp.set_rx_rate(self.sample_rate)
        self.usrp.set_rx_freq(self.center_freq)
        self.usrp.set_rx_gain(self.gain)
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        self.rx_streamer = self.usrp.get_rx_stream(st_args)
        self.buffer = np.zeros(self.num_samples, dtype=np.complex64)
        self.md = uhd.types.RXMetadata()
        self.stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        self.stream_cmd.num_samps = self.num_samples
        self.stream_cmd.stream_now = True

    def acquire_one_spectrum(self):
        self.usrp.issue_stream_cmd(self.stream_cmd)
        self.rx_streamer.recv(self.buffer, self.md, timeout=3.0)
        spectrum = np.fft.fftshift(np.fft.fft(self.buffer))
        power = np.abs(spectrum) ** 2 / self.num_samples
        cal_offset = get_calibration(self.center_freq)
        power_dbm = 10 * np.log10(power / 1e-3 + 1e-12) + cal_offset
        f_start = (self.center_freq - self.sample_rate / 2) / 1e6
        f_end = (self.center_freq + self.sample_rate / 2) / 1e6
        freq_axis = np.linspace(f_start, f_end, self.num_samples, endpoint=True)
        return freq_axis, power_dbm

    def toggle_live_scanning(self):
        if self.live_scanning:
            self.live_scanning = False
            self.scan_button.setText("Start scanning")
        else:
            self.num_samples = int(self.samples_combo.currentText())
            self.buffer = np.zeros(self.num_samples, dtype=np.complex64)
            self.stream_cmd.num_samps = self.num_samples
            self.waterfall_history = self.waterfall_lines_spin.value()
            self.waterfall_data = np.full((self.waterfall_history, self.num_samples), -140.0)
            self.waterfall_index = 0
            self.waterfall_img.setImage(self.waterfall_data, autoLevels=False, levels=(-140, -30))
            self.gain = self.gain_spin.value()
            self.usrp.set_rx_gain(self.gain)

            self.live_scanning = True
            self.scan_button.setText("Pause")
            self.init_live_scan_parameters()
            self.composite_scan_cycle()

    def init_live_scan_parameters(self):
        try:
            start_mhz = float(self.start_freq_edit.text())
            stop_mhz = float(self.stop_freq_edit.text())
            step_mhz = float(self.step_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Enter a valid number")
            self.live_scanning = False
            self.scan_button.setText("Start scanning")
            return
        start_hz = start_mhz * 1e6
        stop_hz = stop_mhz * 1e6
        step_hz = step_mhz * 1e6
        self.wb_centers = np.arange(start_hz + self.sample_rate / 2,
                                    stop_hz - self.sample_rate / 2 + step_hz,
                                    step_hz)
        self.wb_index = 0
        common_res = 0.1  # MHz
        num_points = int(np.round((stop_mhz - start_mhz) / common_res)) + 1
        self.common_freq = np.linspace(start_mhz, stop_mhz, num_points, endpoint=True)
        self.composite_spectrum = np.full(self.common_freq.shape, -140.0)
        self.maxhold_data_arr = np.full(self.common_freq.shape, -140.0)
        self.waterfall_img.setRect(QtCore.QRectF(start_mhz, 0, stop_mhz - start_mhz, self.waterfall_history))
        self.curve.clear()
        self.waterfall_data.fill(-140)
        self.waterfall_index = 0

    def composite_scan_cycle(self):
        if not self.live_scanning:
            return
        if self.wb_index < len(self.wb_centers):
            new_center = self.wb_centers[self.wb_index]
            self.usrp.set_rx_freq(new_center)
            self.center_freq = new_center
            QtCore.QTimer.singleShot(1, self.do_composite_measurement)
        else:
            self.curve.setData(self.common_freq, self.composite_spectrum)
            self.current_x = self.common_freq.copy()
            self.current_y = self.composite_spectrum.copy()
            if self.maxhold_enabled:
                self.maxhold_curve.setData(self.common_freq, self.maxhold_data_arr)
            else:
                self.maxhold_curve.clear()
            if self.composite_spectrum.size > 0:
                idx_peak = np.argmax(self.composite_spectrum)
                self.max_marker.setData([self.common_freq[idx_peak]], [self.composite_spectrum[idx_peak]])
                self.max_text.setText(
                    f"{self.common_freq[idx_peak]:.2f} MHz\n{self.composite_spectrum[idx_peak]:.1f} dBm")
                self.max_text.setPos(self.common_freq[idx_peak], self.composite_spectrum[idx_peak])
            row_data = np.interp(np.linspace(self.common_freq[0], self.common_freq[-1], self.num_samples),
                                 self.common_freq, self.composite_spectrum)
            if self.waterfall_index < self.waterfall_history:
                self.waterfall_data[self.waterfall_index, :] = row_data
                self.waterfall_index += 1
            else:
                self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
                self.waterfall_data[-1, :] = row_data
            self.waterfall_img.setImage(self.waterfall_data, autoLevels=False, levels=(-140, -30))
            self.wb_index = 0
            QtCore.QTimer.singleShot(1, self.composite_scan_cycle)

    def do_composite_measurement(self):
        meas_freq, meas_power = self.acquire_one_spectrum()
        seg_min = meas_freq[0]
        seg_max = meas_freq[-1]
        mask = (self.common_freq >= seg_min) & (self.common_freq <= seg_max)
        if np.any(mask):
            interp_power = np.interp(self.common_freq[mask], meas_freq, meas_power, left=-140, right=-140)
            self.composite_spectrum[mask] = interp_power
            if self.maxhold_enabled:
                self.maxhold_data_arr[mask] = np.maximum(self.maxhold_data_arr[mask], interp_power)
        self.wb_index += 1
        QtCore.QTimer.singleShot(1, self.composite_scan_cycle)

    def toggle_maxhold(self):
        self.maxhold_enabled = not self.maxhold_enabled
        if self.maxhold_enabled:
            self.maxhold_button.setText("Turn Max Hold off")
            if self.common_freq is not None:
                self.maxhold_data_arr = self.composite_spectrum.copy()
        else:
            self.maxhold_button.setText("Turn Max Hold on")
            self.maxhold_curve.clear()

    def start_transmission(self):
        if not self.tx_enabled:
            try:
                tx_freq = float(self.tx_freq_edit.text()) * 1e6  # перевод в Гц
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", "Enter a valid number")
                return
            tx_gain = self.tx_gain_spin.value()
            self.usrp.set_tx_freq(tx_freq)
            self.usrp.set_tx_gain(tx_gain)
            # Создаем TX поток
            tx_stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            self.tx_stream = self.usrp.get_tx_stream(tx_stream_args)
            # Генерируем тональный сигнал (1 кГц) с длиной self.num_samples
            t = np.arange(self.num_samples)
            tone_freq = 1000  # 1 кГц
            self.tx_buffer = np.exp(1j * 2 * np.pi * tone_freq * t / self.sample_rate).astype(np.complex64)
            # Запускаем поток TX
            self.tx_thread = TXThread(self.tx_stream, self.tx_buffer)
            self.tx_thread.start()
            self.tx_enabled = True
            self.tx_start_button.setText("Pause transmission")
            self.tx_status_label.setText(f"Transmission at {tx_freq / 1e6:.2f} MHz, Gain Tx {tx_gain}")
        else:
            self.tx_enabled = False
            if self.tx_thread is not None:
                self.tx_thread.stop()
            self.tx_start_button.setText("Start transmission")
            self.tx_status_label.setText("Transmission stopped")

    def toggle_sweep_transmission(self):
        if not self.sweep_enabled:
            # Начинаем sweep передачу
            try:
                start_freq = float(self.sweep_start_edit.text()) * 1e6  # MHz -> Hz
                stop_freq = float(self.sweep_stop_edit.text()) * 1e6  # MHz -> Hz
                step_freq = float(self.sweep_step_edit.text()) * 1e6  # MHz -> Hz
                dwell_time = float(self.sweep_dwell_edit.text())  # ms
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", "Enter a valid number")
                return

            if step_freq <= 0:
                QtWidgets.QMessageBox.warning(self, "Error", "Enter a valid number")
                return

            # Генерируем список частот для sweep
            if start_freq < stop_freq:
                self.sweep_freqs = np.arange(start_freq, stop_freq + step_freq, step_freq)
            else:
                self.sweep_freqs = np.arange(start_freq, stop_freq - step_freq, -step_freq)

            if len(self.sweep_freqs) == 0:
                QtWidgets.QMessageBox.warning(self, "Error", "Incorrect sweeps parameters")
                return

            # Устанавливаем параметры передачи
            self.usrp.set_tx_gain(self.sweep_gain_spin.value())

            # Генерируем тональный сигнал (1 кГц) с длиной self.num_samples
            t = np.arange(self.num_samples)
            tone_freq = 1000  # 1 кГц
            self.tx_buffer = np.exp(1j * 2 * np.pi * tone_freq * t / self.sample_rate).astype(np.complex64)

            # Создаем TX поток, если его еще нет
            if self.tx_stream is None:
                tx_stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
                self.tx_stream = self.usrp.get_tx_stream(tx_stream_args)

            # Запускаем поток TX, если он не запущен
            if self.tx_thread is None or not self.tx_thread.isRunning():
                self.tx_thread = TXThread(self.tx_stream, self.tx_buffer)
                self.tx_thread.start()

            # Начинаем sweep
            self.sweep_enabled = True
            self.current_sweep_freq = 0
            self.sweep_timer.start(int(dwell_time))  # Преобразуем в int
            self.sweep_start_button.setText("Stop sweep")
            self.sweep_status_label.setText(f"Sweep transmission... {self.sweep_freqs[0] / 1e6:.2f} MHz")

            # Устанавливаем первую частоту
            self.usrp.set_tx_freq(self.sweep_freqs[0])
        else:
            # Останавливаем sweep передачу
            self.sweep_enabled = False
            self.sweep_timer.stop()
            self.sweep_start_button.setText("Start sweep")
            self.sweep_status_label.setText("Sweep stopped")

            # Останавливаем поток TX, если нет других активных передач
            if not self.tx_enabled and self.tx_thread is not None:
                self.tx_thread.stop()

    def next_sweep_step(self):
        if not self.sweep_enabled:
            return

        self.current_sweep_freq += 1
        if self.current_sweep_freq >= len(self.sweep_freqs):
            self.current_sweep_freq = 0  # Зацикливаем sweep

        freq = self.sweep_freqs[self.current_sweep_freq]
        self.usrp.set_tx_freq(freq)
        self.sweep_status_label.setText(f"Sweep transmission... {freq / 1e6:.2f} MHz")

    def on_mouse_moved(self, evt):
        pos = evt[0]
        if self.graph_plot.sceneBoundingRect().contains(pos) and self.current_x is not None:
            mouse_point = self.graph_plot.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            self.vLine.setPos(x)
            self.hLine.setPos(mouse_point.y())
            idx = (np.abs(self.current_x - x)).argmin()
            freq_val = self.current_x[idx]
            power_val = self.current_y[idx]
            self.cursorLabel.setText(f"{freq_val:.2f} MHz\n{power_val:.1f} dBm")
            self.cursorLabel.setPos(freq_val, power_val)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    analyzer = SpectrumAnalyzer()
    analyzer.show()
    sys.exit(app.exec_())
