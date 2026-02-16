import sys
import os
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget, QFileDialog, QMessageBox
)
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QPageLayout

from qrscout_core import (
    DataFrameModel,
    drop_empty_rows, find_col,
    force_int_series, force_int_value, is_known_int_column,
    compute_fuel_ranking, sort_raw_by_team,
    series_climb_levels, level_to_label,
    print_tableview, print_widget,
    save_model_as_csv, save_scorecard_report,
)
from qrscout_pages import (
    TablePage, FuelRankingPage, AddDataPage, RobotScoreCardPage, PreMatchDataPage, ProgressGraphPage
)


ROLE_KIND = Qt.UserRole + 1          # "header" or "page"
ROLE_PAGE_INDEX = Qt.UserRole + 2    # int page index


class MenuListWidget(QListWidget):
    """
    QListWidget that never allows headers to become the current item.
    Also adds ~50% extra vertical spacing between items.
    """
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QListWidget::item {
                padding-top: 12px;
                padding-bottom: 12px;
                padding-left: 8px;
                padding-right: 8px;
            }
        """)

    def is_header(self, item: QListWidgetItem | None) -> bool:
        return bool(item and item.data(ROLE_KIND) == "header")

    def skip_headers(self, direction: int):
        # direction: +1 (down) or -1 (up)
        row = self.currentRow()
        n = self.count()
        if n <= 0:
            return

        while 0 <= row < n and self.is_header(self.item(row)):
            row += direction

        row = max(0, min(n - 1, row))

        if self.is_header(self.item(row)):
            row2 = row - direction
            while 0 <= row2 < n and self.is_header(self.item(row2)):
                row2 -= direction
            if 0 <= row2 < n and not self.is_header(self.item(row2)):
                row = row2

        if 0 <= row < n and not self.is_header(self.item(row)):
            self.setCurrentRow(row)

    def keyPressEvent(self, event):
        before = self.currentRow()
        super().keyPressEvent(event)

        after = self.currentRow()
        if after == before:
            return

        direction = 1 if after > before else -1
        if self.is_header(self.currentItem()):
            self.skip_headers(direction)

    def mousePressEvent(self, event):
        #item = self.itemAt(event.pos())
        item = self.itemAt(event.position().toPoint())
        if self.is_header(item):
            return
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("QRScout - Data Analyzer FRC TEAM 1024 KIL-A-BYTES")

        self.current_path: str | None = None
        self.raw_df = pd.DataFrame()

        # Left menu
        self.menu = MenuListWidget()
        self.menu.setFixedWidth(260)
        self.menu.currentRowChanged.connect(self.on_menu_changed)

        def add_header(text: str):
            item = QListWidgetItem(text)
            item.setData(ROLE_KIND, "header")
            item.setFlags(Qt.ItemIsEnabled)  # enabled so it renders normally, but not selectable

            font = item.font()
            font.setBold(True)
            font.setUnderline(True)
            item.setFont(font)

            self.menu.addItem(item)

        def add_page(text: str, page_index: int):
            item = QListWidgetItem(text)
            item.setData(ROLE_KIND, "page")
            item.setData(ROLE_PAGE_INDEX, int(page_index))
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.menu.addItem(item)

        add_header("Data Update")
        add_page("Add Data", 0)

        add_header("Raw Data")
        add_page("All Data", 1)
        add_page("All Data (Sorted by Team)", 2)

        add_header("Reports")
        add_page("Robot Fuel Ranking", 3)
        add_page("Robot Score Card", 4)
        add_page("Pre-Match Data", 5)
        add_page("Progress Graph", 6)

        # Pages
        self.pages = QStackedWidget()

        self.logo = QLabel()
        pix = QPixmap("./1024-kil-a-bytes.png")
        self.logo.setPixmap(pix.scaledToWidth(100, Qt.SmoothTransformation))
        self.logo.setAlignment(Qt.AlignRight | Qt.AlignTop)

        self.add_page = AddDataPage(self.on_add_pasted_row, self.on_add_manual_row)
        self.raw_page = TablePage("All Data", self.open_csv_dialog, self.on_print_clicked, self.on_save_clicked)
        self.raw_sorted_page = TablePage(
            "All Data (Sorted by Team Number)", self.open_csv_dialog, self.on_print_clicked, self.on_save_clicked
        )
        self.rank_page = FuelRankingPage(self.on_print_clicked, self.on_save_clicked)

        self.score_card_page = RobotScoreCardPage(
            on_generate_clicked=self.generate_score_card,
            on_print_clicked=self.print_score_card,
            on_save_clicked=self.save_score_card
        )

        #self.prematch_page = PreMatchDataPage(on_generate_clicked=self.generate_prematch)

        self.prematch_page = PreMatchDataPage(
            on_generate_clicked=self.generate_prematch,
            on_print_clicked=self.on_print_clicked,
            on_save_clicked=self.on_save_clicked
        )

        self.progress_graph_page = ProgressGraphPage(
            on_generate_clicked=self.generate_progress_graph,
            on_print_clicked=self.on_print_widget_clicked,
        )


        self.pages.addWidget(self.add_page)         # 0
        self.pages.addWidget(self.raw_page)         # 1
        self.pages.addWidget(self.raw_sorted_page)  # 2
        self.pages.addWidget(self.rank_page)        # 3
        self.pages.addWidget(self.score_card_page)  # 4
        self.pages.addWidget(self.prematch_page)    # 5
        self.pages.addWidget(self.progress_graph_page)  # 6


        # Models
        self.raw_model = DataFrameModel(pd.DataFrame())
        self.raw_sorted_model = DataFrameModel(pd.DataFrame())
        self.rank_model = DataFrameModel(pd.DataFrame())

        self.raw_page.table.setModel(self.raw_model)
        self.raw_sorted_page.table.setModel(self.raw_sorted_model)
        self.rank_page.table.setModel(self.rank_model)

        # Layout
        root = QWidget()
        hl = QHBoxLayout()
        #hl.addWidget(self.menu)
        #hl.addWidget(self.pages, 1)

        hl.addWidget(self.menu)

        right_side = QVBoxLayout()
        right_side.addWidget(self.logo, 0, Qt.AlignRight | Qt.AlignTop)
        right_side.addWidget(self.pages, 1)

        right_widget = QWidget()
        right_widget.setLayout(right_side)

        hl.addWidget(right_widget, 1)
        #==== End the change for adding logo  =====

        root.setLayout(hl)
        self.setCentralWidget(root)

        # Bigger default size
        self.resize(1700, 950)

        # Default to first real page item (row 1: "Add Data")
        self.menu.setCurrentRow(1)

        if initial_path:
            self.load_csv(initial_path)

    def on_menu_changed(self, row: int):
        item = self.menu.item(row)
        if not item:
            return

        if item.data(ROLE_KIND) == "header":
            last = getattr(self, "_last_menu_row", row)
            direction = -1 if row < last else +1
            self.menu.skip_headers(direction)
            return

        self._last_menu_row = row

        page_index = item.data(ROLE_PAGE_INDEX)
        if page_index is None:
            return

        page_index = int(page_index)

        # Special action item: Load CSV File
        if page_index == -1:
            #self.open_csv_dialog()
            return
        self.pages.setCurrentIndex(int(page_index))


    # ---------- File open ----------
    def open_csv_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if not path:
            return
        self.load_csv(path)

    def load_csv(self, path: str):
        if not os.path.exists(path):
            QMessageBox.critical(self, "File Not Found", f"CSV file does not exist:\n{path}")
            return

        try:
            df = pd.read_csv(path)
            df = drop_empty_rows(df)
        except Exception as e:
            QMessageBox.critical(self, "CSV Load Error", f"Failed to read CSV:\n{e}")
            return

        self.current_path = path
        self.raw_df = df.reset_index(drop=True)

        # Update Add Data page schema
        self.add_page.set_loaded_path(self.current_path)
        self.add_page.set_columns([str(c) for c in self.raw_df.columns])

        self.refresh_views()

    # ---------- Printing / Saving for table pages ----------
    def on_print_clicked(self, title: str, table):
        print_tableview(self, title, table)

    def on_save_clicked(self, title: str, table):
        model = table.model()
        if not isinstance(model, DataFrameModel):
            QMessageBox.critical(self, "Save CSV", "Internal error: unexpected table model type.")
            return

        safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        default_name = f"{safe_title}.csv"
        save_model_as_csv(self, default_name, model)

    # ---------- Refresh ----------
    def refresh_team_list_for_scorecard(self):
        if self.raw_df.empty:
            self.score_card_page.set_team_list([])
            self.prematch_page.set_team_list([])
            self.progress_graph_page.set_team_list([])
            return

        team_col = find_col(self.raw_df, "Team Number")
        if team_col is None:
            self.score_card_page.set_team_list([])
            self.prematch_page.set_team_list([])
            self.progress_graph_page.set_team_list([])
            return

        teams = pd.to_numeric(self.raw_df[team_col], errors="coerce")
        teams = teams.dropna().astype(int).unique().tolist()
        teams.sort()
        team_list = [str(t) for t in teams]
        self.progress_graph_page.set_team_list(team_list)

        self.score_card_page.set_team_list(team_list)
        self.prematch_page.set_team_list(team_list)

    def refresh_views(self):
        df = self.raw_df.copy()

        # Raw
        #self.raw_model.set_df(df.fillna(""))
        self.raw_model.set_df(df.astype(str).replace("nan", ""))

        self.raw_page.table.resizeColumnsToContents()
        self.raw_page.status.setText(f"Loaded: {self.current_path}  |  Rows: {len(df)}  Cols: {len(df.columns)}")

        # Raw sorted
        try:
            sorted_df = sort_raw_by_team(df)
            display_df = sorted_df.astype(str).replace("nan", "")
            self.raw_sorted_model.set_df(display_df)
            self.raw_sorted_page.table.resizeColumnsToContents()
            self.raw_sorted_page.status.setText(
                f"Loaded + sorted by Team Number: {self.current_path}  |  Rows: {len(df)}"
            )
        except Exception as e:
            self.raw_sorted_model.set_df(pd.DataFrame())
            self.raw_sorted_page.status.setText(f"Cannot sort by Team Number: {e}")

        # Ranking
        try:
            rank_df = compute_fuel_ranking(df)
            self.rank_model.set_df(rank_df)
            self.rank_page.table.resizeColumnsToContents()
            self.rank_page.info.setText(
                f"Teams: {len(rank_df)} (sorted by Total Fuel Scored, then Scoring Percentage)"
            )
        except Exception as e:
            self.rank_model.set_df(pd.DataFrame())
            self.rank_page.info.setText(f"Cannot compute fuel ranking: {e}")

        # Team list for scorecard + prematch
        self.refresh_team_list_for_scorecard()

    # ---------- Add Data ----------
    def maybe_autosave(self):
        if not self.add_page.auto_save.isChecked():
            return
        if not self.current_path:
            QMessageBox.information(self, "Auto-save", "No CSV loaded. Use Save as CSV& to export.")
            return
        try:
            self.raw_df.to_csv(self.current_path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Auto-save Error", f"Failed to save back to CSV:\n{e}")

    def append_row_dict(self, row_dict: dict):
        if self.raw_df.empty:
            QMessageBox.information(self, "Add Data", "Load a CSV first.")
            return

        row = {}
        for col in self.raw_df.columns:
            v = row_dict.get(col, "")
            if is_known_int_column(col):
                v = force_int_value(v)
            row[col] = v

        new_df = pd.DataFrame([row], columns=self.raw_df.columns)
        new_df = drop_empty_rows(new_df)
        if len(new_df) == 0:
            QMessageBox.information(self, "Add Data", "Row is empty  nothing added.")
            return

        self.raw_df = pd.concat([self.raw_df, new_df], ignore_index=True)
        self.raw_df = drop_empty_rows(self.raw_df).reset_index(drop=True)

        self.refresh_views()
        self.maybe_autosave()

    def on_add_pasted_row(self):
        if self.raw_df.empty:
            QMessageBox.information(self, "Add Data", "Load a CSV first.")
            return

        text = self.add_page.paste_box.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "Add Data", "Paste box is empty.")
            return

        line = next((ln for ln in text.splitlines() if ln.strip()), "")
        parts = line.split("\t")

        cols = list(self.raw_df.columns)
        if len(parts) < len(cols):
            parts = parts + [""] * (len(cols) - len(parts))
        elif len(parts) > len(cols):
            parts = parts[:len(cols) - 1] + ["\t".join(parts[len(cols) - 1:])]

        row_dict = {cols[i]: parts[i] for i in range(len(cols))}
        self.append_row_dict(row_dict)
        self.add_page.paste_box.clear()

    def on_add_manual_row(self):
        if self.raw_df.empty:
            QMessageBox.information(self, "Add Data", "Load a CSV first.")
            return

        cols = list(self.raw_df.columns)
        row_dict = {}
        for c, col in enumerate(cols):
            item = self.add_page.manual_table.item(0, c)
            row_dict[col] = item.text() if item else ""

        self.append_row_dict(row_dict)

        for c in range(len(cols)):
            it = self.add_page.manual_table.item(0, c)
            if it is None:
                self.add_page.manual_table.setItem(0, c, None)
                self.add_page.manual_table.setItem(c, 0, None)
            else:
                it.setText("")

    # ---------- Robot Score Card ----------
    def generate_score_card(self):
        if self.raw_df.empty:
            self.score_card_page.set_report_rows([], status="Load a CSV first.")
            return

        team_col = find_col(self.raw_df, "Team Number")
        if team_col is None:
            self.score_card_page.set_report_rows([], status="Missing Team Number column.")
            return

        team_str = self.score_card_page.selected_team()
        if not team_str:
            self.score_card_page.set_report_rows([], status="Select a team.")
            return

        team_num = force_int_value(team_str)
        self.score_card_page.set_report_title(str(team_num))

        df_team = self.raw_df[
            pd.to_numeric(self.raw_df[team_col], errors="coerce").fillna(-1).astype(int) == team_num
        ].copy()

        if df_team.empty:
            self.score_card_page.set_report_rows([], status="No rows found for this team.")
            return

        rows = self._metrics_for_team_df(df_team)
        self.score_card_page.set_report_rows(rows, status=f"Report generated for Team {team_num}.")

    def print_score_card(self):
        title, rows = self.score_card_page.last_report()
        if not rows:
            QMessageBox.information(self, "Print", "Generate a score card first.")
            return
        print_widget(self, title, self.score_card_page)

    def save_score_card(self):
        title, rows = self.score_card_page.last_report()
        if not rows:
            QMessageBox.information(self, "Save Report", "Generate a score card first.")
            return
        safe = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        default_name = f"{safe}.tsv"
        save_scorecard_report(self, default_name, title, rows)

    # ---------- Shared metric computation (Score Card + Pre-Match) ----------
    def _metrics_for_team_df(self, df_team: pd.DataFrame) -> list[tuple[str, str]]:
        matches = len(df_team)

        # Required scoring columns
        fuel_scored_col = find_col(df_team, "Fuel scored")                      # Auto scored
        robot_fuel_scored_col = find_col(df_team, "Robot Fuel Scored")          # Tele scored
        fuel_missed_col = find_col(df_team, "Fuel shot and missed")             # Auto missed
        robot_fuel_missed_col = find_col(df_team, "Robot Fuel Shot and Missed") # Tele missed

        auto_scored = force_int_series(df_team[fuel_scored_col]) if fuel_scored_col else pd.Series([0] * matches)
        tele_scored = force_int_series(df_team[robot_fuel_scored_col]) if robot_fuel_scored_col else pd.Series([0] * matches)
        auto_missed = force_int_series(df_team[fuel_missed_col]) if fuel_missed_col else pd.Series([0] * matches)
        tele_missed = force_int_series(df_team[robot_fuel_missed_col]) if robot_fuel_missed_col else pd.Series([0] * matches)

        # Averages
        avg_tele_scored = int(round(tele_scored.mean())) if matches else 0
        avg_auto_scored = int(round(auto_scored.mean())) if matches else 0

        cycles_col = find_col(df_team, "TeleOp - Cycles Completed")
        if cycles_col is not None:
            cycles = force_int_series(df_team[cycles_col])
            avg_cycles = int(round(cycles.mean())) if matches else 0
        else:
            avg_cycles = 0


        # Accuracies
        auto_attempts_total = int((auto_scored + auto_missed).sum())
        tele_attempts_total = int((tele_scored + tele_missed).sum())
        total_scored_total = int((auto_scored + tele_scored).sum())
        total_attempts_total = int((auto_scored + tele_scored + auto_missed + tele_missed).sum())

        auto_acc = (int(auto_scored.sum()) / auto_attempts_total * 100.0) if auto_attempts_total > 0 else None
        tele_acc = (int(tele_scored.sum()) / tele_attempts_total * 100.0) if tele_attempts_total > 0 else None
        total_acc = (total_scored_total / total_attempts_total * 100.0) if total_attempts_total > 0 else None

        auto_acc_str = "" if auto_acc is None else f"{auto_acc:.2f}%"
        tele_acc_str = "" if tele_acc is None else f"{tele_acc:.2f}%"
        total_acc_str = "" if total_acc is None else f"{total_acc:.2f}%"

        # Defensive skill avg
        def_skill_col = find_col(df_team, "Defensive Skill")
        if def_skill_col is not None:
            def_skill = pd.to_numeric(df_team[def_skill_col], errors="coerce").dropna()
            avg_def_skill = f"{def_skill.mean():.2f}" if len(def_skill) else ""
        else:
            avg_def_skill = ""

        # Climb columns
        auto_climb_col = find_col(df_team, "Climbed Tower")  # Auto
        tele_climb_col = find_col(df_team, "Tower.1")        # Tele

        auto_levels = series_climb_levels(df_team[auto_climb_col]) if auto_climb_col else pd.Series([], dtype=int)
        tele_levels = series_climb_levels(df_team[tele_climb_col]) if tele_climb_col else pd.Series([], dtype=int)

        auto_success = int((auto_levels > 0).sum()) if len(auto_levels) else 0

        all_levels = (
            pd.concat([auto_levels, tele_levels], ignore_index=True)
            if (len(auto_levels) or len(tele_levels)) else pd.Series([], dtype=int)
        )
        l1_cnt = int((all_levels == 1).sum()) if len(all_levels) else 0
        l2_cnt = int((all_levels == 2).sum()) if len(all_levels) else 0
        l3_cnt = int((all_levels == 3).sum()) if len(all_levels) else 0

        def count_F(series):
            if series is None or len(series) == 0:
                return 0
            s = series.fillna("").astype(str).str.strip().str.upper()
            return int((s == "F").sum())

        f_cnt = 0
        if auto_climb_col:
            f_cnt += count_F(df_team[auto_climb_col])
        if tele_climb_col:
            f_cnt += count_F(df_team[tele_climb_col])

        # "Climbed From" counts for successful climbs
        auto_from_col = find_col(df_team, "Climbed From")
        tele_from_col = find_col(df_team, "Tele Climbed From")

        from_counts: dict[str, int] = {}

        def add_from_counts(from_series, success_mask):
            if from_series is None or len(from_series) == 0 or success_mask is None or len(success_mask) == 0:
                return
            vals = from_series.fillna("").astype(str).str.strip()
            vals = vals[success_mask]
            vals = vals[vals != ""]
            for k, v in vals.value_counts().to_dict().items():
                from_counts[k] = from_counts.get(k, 0) + int(v)

        if auto_from_col and len(auto_levels):
            add_from_counts(df_team[auto_from_col], (auto_levels > 0))
        if tele_from_col and len(tele_levels):
            add_from_counts(df_team[tele_from_col], (tele_levels > 0))

        if from_counts:
            items = sorted(from_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            from_str = ", ".join([f"{k}:{v}" for k, v in items[:6]])
        else:
            from_str = ""

        return [
            ("Matches (rows) in data", str(matches)),
            ("Average Fuel Scored in Teleop", str(avg_tele_scored)),
            ("Average Fuel Scored in Auto", str(avg_auto_scored)),
            ("Auto Accuracy %", auto_acc_str),
            ("Teleop Accuracy %", tele_acc_str),
            ("Total Accuracy %", total_acc_str),
            ("Average TeleOp Cycles Completed", str(avg_cycles)),
            ("Average Defensive Skill", avg_def_skill),
            ("Auto Successful Climbs / Matches", f"{auto_success} / {matches}"),
            ("L1 Climbs / Matches", f"{l1_cnt} / {matches}"),
            ("L2 Climbs / Matches", f"{l2_cnt} / {matches}"),
            ("L3 Climbs / Matches", f"{l3_cnt} / {matches}"),
            ("F Climbs / Matches", f"{f_cnt} / {matches}"),
            ("Climbed From (successful) counts", from_str),
        ]

    # ---------- Pre-Match Data ----------
    def generate_prematch(self):
        if self.raw_df.empty:
            self.prematch_page.set_table(["1024", "T2", "T3", "T4", "T5", "T6"], [], status="Load a CSV first.")
            return

        team_col = find_col(self.raw_df, "Team Number")
        if team_col is None:
            self.prematch_page.set_table(["1024", "T2", "T3", "T4", "T5", "T6"], [], status="Missing Team Number column.")
            return

        teams = self.prematch_page.selected_teams()
        teams = teams[:6] + [""] * max(0, 6 - len(teams))

        metric_map_per_team: list[dict[str, str]] = []
        for t in teams:
            if not t:
                metric_map_per_team.append({})
                continue
            tn = force_int_value(t)
            df_team = self.raw_df[
                pd.to_numeric(self.raw_df[team_col], errors="coerce").fillna(-1).astype(int) == tn
            ].copy()

            rows = self._metrics_for_team_df(df_team) if not df_team.empty else [("Matches (rows) in data", "0")]
            metric_map_per_team.append({k: v for k, v in rows})

        base_metrics = list(metric_map_per_team[0].keys()) if metric_map_per_team and metric_map_per_team[0] else [
            "Matches (rows) in data",
            "Average Fuel Scored in Teleop",
            "Average Fuel Scored in Auto",
            "Auto Accuracy %",
            "Teleop Accuracy %",
            "Total Accuracy %",
            "Average Defensive Skill",
            "Auto Successful Climbs / Matches",
            "L1 Climbs / Matches",
            "L2 Climbs / Matches",
            "L3 Climbs / Matches",
            "F Climbs / Matches",
            "Climbed From (successful) counts",
        ]

        table_rows: list[tuple[str, list[str]]] = []
        for m in base_metrics:
            vals = [metric_map_per_team[i].get(m, "") for i in range(6)]
            table_rows.append((m, vals))

        self.prematch_page.set_table(teams, table_rows, status="Pre-match comparison generated.")

    def generate_progress_graph(self):
        if self.raw_df.empty:
            self.progress_graph_page.status.setText("Load a CSV first.")
            return

        team_col = find_col(self.raw_df, "Team Number")
        match_col = find_col(self.raw_df, "Match Number")

        auto_scored_col = find_col(self.raw_df, "Auto - Robot Fuel scored")
        tele_scored_col = find_col(self.raw_df, "TeleOp - Robot Fuel Scored")
        cycles_col = find_col(self.raw_df, "TeleOp - Cycles Completed")

        missing = []
        for name, col in [
            ("Team Number", team_col),
            ("Match Number", match_col),
            ("Auto - Robot Fuel scored", auto_scored_col),
            ("TeleOp - Robot Fuel Scored", tele_scored_col),
            ("TeleOp - Cycles Completed", cycles_col),
        ]:
            if col is None:
                missing.append(name)

        if missing:
            self.progress_graph_page.status.setText("Missing column(s): " + ", ".join(missing))
            return

        team_str = self.progress_graph_page.selected_team()
        if not team_str:
            self.progress_graph_page.status.setText("Select a team.")
            return

        team_num = force_int_value(team_str)

        df_team = self.raw_df[
            pd.to_numeric(self.raw_df[team_col], errors="coerce").fillna(-1).astype(int) == team_num
        ].copy()

        if df_team.empty:
            self.progress_graph_page.status.setText(f"No rows found for Team {team_num}.")
            return

        # Convert + sort by match number
        df_team["_match"] = pd.to_numeric(df_team[match_col], errors="coerce")
        df_team = df_team.dropna(subset=["_match"]).sort_values("_match")
        if df_team.empty:
            self.progress_graph_page.status.setText(f"No valid match numbers for Team {team_num}.")
            return

        match_nums = df_team["_match"].astype(int).to_list()
        auto_scored = pd.to_numeric(df_team[auto_scored_col], errors="coerce").fillna(0).astype(int).to_list()
        tele_scored = pd.to_numeric(df_team[tele_scored_col], errors="coerce").fillna(0).astype(int).to_list()
        cycles = pd.to_numeric(df_team[cycles_col], errors="coerce").fillna(0).astype(int).to_list()

        self.progress_graph_page.plot_progress(match_nums, auto_scored, tele_scored, cycles)
        self.progress_graph_page.status.setText(f"Progress graph generated for Team {team_num}.")

    def on_print_widget_clicked(self, title: str, widget):
        print_widget(self, title, widget)



def main():
    initial_path = sys.argv[1] if len(sys.argv) >= 2 else None
    app = QApplication(sys.argv)
    # --- Blue / Yellow theme ---
    app.setStyleSheet("""
        QMainWindow, QWidget { background: #0b1f3a; color: #ffffff; }
        QLabel { color: #ffffff; }

        QPushButton {
            background: #f6c400;
            color: #0b1f3a;
            border: 1px solid #caa300;
            border-radius: 6px;
            padding: 6px 10px;
            font-weight: 600;
        }
        QPushButton:hover { background: #ffd24d; }
        QPushButton:pressed { background: #e0b200; }

        QListWidget {
            background: #0a1830;
            border: 1px solid #14345f;
        }
        QListWidget::item:selected {
            background: #1f5fbf;
            color: #ffffff;
        }

        QTableView, QTableWidget {
            background: #07162b;
            alternate-background-color: #0b1f3a;
            gridline-color: #14345f;
            selection-background-color: #1f5fbf;
            selection-color: #ffffff;
            border: 1px solid #14345f;
        }
        QHeaderView::section {
            background: #0a1830;
            color: #f6c400;
            padding: 6px;
            border: 1px solid #14345f;
            font-weight: 700;
        }

        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {
            background: #07162b;
            color: #ffffff;
            border: 1px solid #14345f;
            border-radius: 4px;
            padding: 4px;
        }
        QCheckBox { color: #ffffff; }
    """)
    app.setStyleSheet(app.styleSheet() + """
        * { font-size: 16pt; }
    """)

    w = MainWindow(initial_path=initial_path)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
