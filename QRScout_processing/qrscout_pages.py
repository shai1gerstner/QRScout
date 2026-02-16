from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableView,
    QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
    QCheckBox, QComboBox, QHeaderView
)

from PySide6.QtWidgets import QFileDialog, QMessageBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from qrscout_core import DataFrameModel


class TablePage(QWidget):
    def __init__(self, title: str, on_open_clicked, on_print_clicked, on_save_clicked):
        super().__init__()
        self.title_text = title

        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: 600;")

        self.table = QTableView()
        self.table.setSortingEnabled(True)

        self.status = QLabel("Open a CSV to view data.")

        btn_row = QHBoxLayout()

        self.btn_open = QPushButton("Open CSV…")
        self.btn_open.clicked.connect(on_open_clicked)

        self.btn_print = QPushButton("Print…")
        self.btn_print.clicked.connect(lambda: on_print_clicked(self.title_text, self.table))

        self.btn_save = QPushButton("Save as CSV…")
        self.btn_save.clicked.connect(lambda: on_save_clicked(self.title_text, self.table))

        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(self.btn_print)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addLayout(btn_row)
        layout.addWidget(self.status)
        layout.addWidget(self.table)
        self.setLayout(layout)


class FuelRankingPage(QWidget):
    def __init__(self, on_print_clicked, on_save_clicked):
        super().__init__()
        self.title_text = "Robot Fuel Ranking"

        self.title = QLabel(self.title_text)
        self.title.setStyleSheet("font-weight: 600;")

        self.info = QLabel("Fuel ranking will appear after a CSV is loaded.")
        self.table = QTableView()
        self.table.setSortingEnabled(True)

        btn_row = QHBoxLayout()

        self.btn_print = QPushButton("Print…")
        self.btn_print.clicked.connect(lambda: on_print_clicked(self.title_text, self.table))

        self.btn_save = QPushButton("Save as CSV…")
        self.btn_save.clicked.connect(lambda: on_save_clicked(self.title_text, self.table))

        btn_row.addWidget(self.btn_print)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addLayout(btn_row)
        layout.addWidget(self.info)
        layout.addWidget(self.table)
        self.setLayout(layout)


class AddDataPage(QWidget):
    def __init__(self, on_add_paste_clicked, on_add_manual_clicked):
        super().__init__()
        self.title_text = "Add Data"

        self.title = QLabel(self.title_text)
        self.title.setStyleSheet("font-weight: 600;")

        self.info = QLabel("Load a CSV first, then paste a TAB-separated row or enter values manually.")
        self.loaded_label = QLabel("CSV: (none)")

        self.auto_save = QCheckBox("Auto-save back to loaded CSV after adding")
        self.auto_save.setChecked(True)

        self.paste_box = QTextEdit()
        self.paste_box.setPlaceholderText("Paste one row here (TAB-separated). Example: Team\\tMatch\\tAuto...")

        paste_btn_row = QHBoxLayout()
        self.btn_add_paste = QPushButton("Add Pasted Row")
        self.btn_add_paste.clicked.connect(on_add_paste_clicked)
        paste_btn_row.addWidget(self.btn_add_paste)
        paste_btn_row.addStretch(1)

        self.manual_table = QTableWidget(1, 0)
        self.manual_table.setRowCount(1)

        manual_btn_row = QHBoxLayout()
        self.btn_add_manual = QPushButton("Add Row From Editor")
        self.btn_add_manual.clicked.connect(on_add_manual_clicked)
        manual_btn_row.addWidget(self.btn_add_manual)
        manual_btn_row.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.info)
        layout.addWidget(self.loaded_label)
        layout.addWidget(self.auto_save)

        layout.addWidget(QLabel("Paste a row"))
        layout.addWidget(self.paste_box)
        layout.addLayout(paste_btn_row)

        layout.addWidget(QLabel("Or enter manually"))
        layout.addWidget(self.manual_table)
        layout.addLayout(manual_btn_row)

        self.setLayout(layout)

    def set_loaded_path(self, path: str | None):
        self.loaded_label.setText(f"CSV: {path if path else '(none)'}")

    def set_columns(self, columns: list[str]):
        self.manual_table.clear()
        self.manual_table.setColumnCount(len(columns))
        self.manual_table.setHorizontalHeaderLabels([str(c) for c in columns])
        self.manual_table.setRowCount(1)
        for c in range(len(columns)):
            self.manual_table.setItem(0, c, QTableWidgetItem(""))


class RobotScoreCardPage(QWidget):
    """
    Tabulated score card (Metric/Value) + Print/Save buttons.
    """
    def __init__(self, on_generate_clicked, on_print_clicked, on_save_clicked):
        super().__init__()
        self.title_text = "Robot Score Card"

        header = QHBoxLayout()
        self.title = QLabel(self.title_text)
        self.title.setStyleSheet("font-weight: 600; font-size: 16px;")
        header.addWidget(self.title)
        header.addStretch(1)

        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Select Team:"))
        self.team_combo = QComboBox()
        select_row.addWidget(self.team_combo, 1)

        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(on_generate_clicked)
        select_row.addWidget(self.btn_generate)

        self.btn_print = QPushButton("Print…")
        self.btn_print.clicked.connect(on_print_clicked)
        select_row.addWidget(self.btn_print)

        self.btn_save = QPushButton("Save Report…")
        self.btn_save.clicked.connect(on_save_clicked)
        select_row.addWidget(self.btn_save)

        self.report_title = QLabel("Team Number ____ Score Card")
        self.report_title.setStyleSheet("font-size: 18px; font-weight: 700;")

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMinimumWidth(1100)


        self.status = QLabel("Load a CSV, then choose a team and click Generate.")

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addLayout(select_row)
        layout.addWidget(self.report_title)

        table_row = QHBoxLayout()
        table_row.addWidget(self.table)
        table_row.addStretch(1)
        layout.addLayout(table_row)

        layout.addWidget(self.status)
        self.setLayout(layout)

        self._last_report_rows: list[tuple[str, str]] = []
        self._last_report_title: str = self.report_title.text()

    def set_team_list(self, teams: list[str]):
        current = self.team_combo.currentText()
        self.team_combo.blockSignals(True)
        self.team_combo.clear()
        self.team_combo.addItems(teams)
        if current and current in teams:
            self.team_combo.setCurrentText(current)
        self.team_combo.blockSignals(False)

    def selected_team(self) -> str:
        return self.team_combo.currentText().strip()

    def set_report_title(self, team: str):
        self.report_title.setText(f"Team Number {team} Score Card")
        self._last_report_title = self.report_title.text()

    def set_report_rows(self, rows: list[tuple[str, str]], status: str = ""):
        self._last_report_rows = rows

        self.table.setRowCount(0)
        for metric, value in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(metric))
            self.table.setItem(r, 1, QTableWidgetItem(value))

        if status:
            self.status.setText(status)

    def last_report(self):
        return self._last_report_title, self._last_report_rows


class PreMatchDataPage(QWidget):
    """
    6-team comparison: team 1024 fixed, other 5 selected by user.
    Table is: Metric + one column per team.
    """
    def __init__(self, on_generate_clicked, on_print_clicked, on_save_clicked):
        super().__init__()
        self.title_text = "Pre-Match Data"

        self.title = QLabel(self.title_text)
        self.title.setStyleSheet("font-weight: 700; font-size: 16px;")

        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Teams:"))

        self.fixed_1024 = QComboBox()
        self.fixed_1024.setEditable(True)
        self.fixed_1024.setCurrentText("1024")
        self.fixed_1024.setEnabled(False)
        self.fixed_1024.setMinimumWidth(110)
        select_row.addWidget(self.fixed_1024)

        self.team_boxes: list[QComboBox] = []
        for _ in range(5):
            cb = QComboBox()
            cb.setEditable(True)
            cb.setMinimumWidth(120)
            self.team_boxes.append(cb)
            select_row.addWidget(cb)

#        self.btn_generate = QPushButton("Generate")
#        self.btn_generate.clicked.connect(on_generate_clicked)
#        select_row.addWidget(self.btn_generate)
#        select_row.addStretch(1)
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(on_generate_clicked)
        select_row.addWidget(self.btn_generate)

        self.btn_print = QPushButton("Print…")
        self.btn_print.clicked.connect(lambda: on_print_clicked(self.title_text, self.table))
        select_row.addWidget(self.btn_print)

        self.btn_save = QPushButton("Save as CSV…")
        self.btn_save.clicked.connect(lambda: on_save_clicked(self.title_text, self.table))
        select_row.addWidget(self.btn_save)

        select_row.addStretch(1)


        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Metric", "1024", "T2", "T3", "T4", "T5", "T6"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 7):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMinimumWidth(1100)


        self.status = QLabel("Pick 5 teams and click Generate.")

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addLayout(select_row)
        layout.addWidget(self.table)
        layout.addWidget(self.status)
        self.setLayout(layout)

    def set_team_list(self, teams: list[str]):
        for cb in self.team_boxes:
            current = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(teams)
            if current and current in teams:
                cb.setCurrentText(current)
            cb.blockSignals(False)

    def selected_teams(self) -> list[str]:
        teams = ["1024"]
        for cb in self.team_boxes:
            t = cb.currentText().strip()
            if t:
                teams.append(t)
        return teams

    def set_table(self, team_labels: list[str], metric_rows: list[tuple[str, list[str]]], status: str = ""):
        headers = ["Metric"] + team_labels
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        self.table.setRowCount(0)
        for metric, values in metric_rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(metric))
            for c, v in enumerate(values, start=1):
                self.table.setItem(r, c, QTableWidgetItem(v))

        if status:
            self.status.setText(status)

class ProgressGraphPage(QWidget):
    """
    Team progress line graphs per match:
      1) Auto fuel scored per match
      2) TeleOp fuel scored per match
      3) Cycles completed per match
    """
    def __init__(self, on_generate_clicked, on_print_clicked):
        super().__init__()
        self.title_text = "Progress Graph"

        self.title = QLabel(self.title_text)
        self.title.setStyleSheet("font-weight: 700; font-size: 16px;")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Select Team:"))

        self.team_combo = QComboBox()
        controls.addWidget(self.team_combo, 1)

        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(on_generate_clicked)
        controls.addWidget(self.btn_generate)

        self.btn_print = QPushButton("Print…")
        self.btn_print.clicked.connect(lambda: on_print_clicked(self.title_text, self.canvas))
        controls.addWidget(self.btn_print)

        controls.addStretch(1)

        self.status = QLabel("Load a CSV, choose a team, then click Generate.")

        # Matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addWidget(self.canvas, 1)
        self.setLayout(layout)

    def set_team_list(self, teams: list[str]):
        current = self.team_combo.currentText()
        self.team_combo.blockSignals(True)
        self.team_combo.clear()
        self.team_combo.addItems(teams)
        if current and current in teams:
            self.team_combo.setCurrentText(current)
        self.team_combo.blockSignals(False)

    def selected_team(self) -> str:
        return self.team_combo.currentText().strip()

    def plot_progress(self, match_nums, auto_scored, tele_scored, cycles):
        self.figure.clear()

        # 3 stacked line graphs
        ax1 = self.figure.add_subplot(3, 1, 1)
        ax2 = self.figure.add_subplot(3, 1, 2, sharex=ax1)
        ax3 = self.figure.add_subplot(3, 1, 3, sharex=ax1)

        ax1.plot(match_nums, auto_scored, marker="o")
        ax1.set_title("Auto Fuel Scored per Match")
        ax1.set_ylabel("Auto Fuel")

        ax2.plot(match_nums, tele_scored, marker="o")
        ax2.set_title("TeleOp Fuel Scored per Match")
        ax2.set_ylabel("TeleOp Fuel")

        ax3.plot(match_nums, cycles, marker="o")
        ax3.set_title("Cycles Completed per Match")
        ax3.set_xlabel("Match Number")
        ax3.set_ylabel("Cycles")

        self.figure.tight_layout()
        self.canvas.draw()
