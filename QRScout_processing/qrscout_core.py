import numpy as np
import pandas as pd

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex, QPoint
from PySide6.QtGui import QPainter, QPageLayout, QPalette, QColor
from PySide6.QtPrintSupport import QPrinter, QPrintDialog
from PySide6.QtWidgets import QMessageBox, QFileDialog, QTableView, QWidget


# ------------------------------------------------------------
# Column mapping: NEW CSV schema -> OLD "known" schema names
# ------------------------------------------------------------
COLUMN_MAP = {
    "Scouter Name": "Scouter Name",
    "Match Number": "Match Number",
    "Team Number": "Team Number",
    "Starting Position": "Starting Position",
    "No Show": "No Show",
    "Moved?": "Moved?",
    "Auto - Robot Fuel scored": "Fuel scored",
    "Auto - Robot Fuel shot and missed": "Fuel shot and missed",
    "Auto - Climbed Tower": "Climbed Tower",
    "Auto - Climbed From": "Climbed From",
    "Auto - Fuel Collected from": "Fuel Collected from",
    "Auto - Crossed into Neutral Zone?": "Crossed into Neutral Zone?",
    "TeleOp - Robot Fuel Scored": "Robot Fuel Scored",
    "TeleOp - Robot Fuel Shot and Missed": "Robot Fuel Shot and Missed",
    "TeleOp - Fuel Delivered to Outpost": "Fuel Delivered to Outpost",
    "TeleOp - Fuel Collected from Outpost": "Fuel Collected from Outpost",
    "TeleOp - Cycles Completed": "TeleOp - Cycles Completed",
    "Tele - Fuel Collected from": "Tele Fuel Collected from",
    "Tele - Robot activity while Hub not active": "Robot activity while Hub not active",
    "Was Robot Defended by Other Alliance?": "Was Robot Defended by Other Alliance?",
    "Tele Climbed Tower": "Tower.1",
    "Tele Climbed From": "Tele Climbed From",
    "Offense Skill": "Offense Skill",
    "Defensive Skill": "Defensive Skill",
    "Died?": "Died?",
    "Tipped/Fell Over?": "Tipped/Fell Over?",
    "Fouls": "Fouls",
    "Yellow/Red Card": "Yellow/Red Card",
    "Final Alliance Score": "Final Alliance Score",
    "Did the Alliance win?": "Did the Alliance win?",
    "Comments": "Comments",
}



def normalize_col_name(name: str) -> str:
    return " ".join(str(name).strip().split()).casefold()


# ---------- Qt model ----------
class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df.copy()

    def set_df(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            return "" if pd.isna(value) else str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])

    def sort(self, column, order):
        colname = self._df.columns[column]
        self.layoutAboutToBeChanged.emit()

        try:
            numeric_col = pd.to_numeric(self._df[colname], errors="coerce")
            if numeric_col.notna().any():
                self._df["_sort_key"] = numeric_col
                self._df["_sort_key_str"] = self._df[colname].astype(str)
                self._df = self._df.sort_values(
                    by=["_sort_key", "_sort_key_str"],
                    ascending=(order == Qt.AscendingOrder),
                    na_position="last"
                ).drop(columns=["_sort_key", "_sort_key_str"])
            else:
                self._df = self._df.sort_values(
                    by=colname,
                    ascending=(order == Qt.AscendingOrder),
                    na_position="last"
                )
        except Exception:
            self._df = self._df.sort_values(
                by=colname,
                ascending=(order == Qt.AscendingOrder),
                na_position="last"
            )

        self._df = self._df.reset_index(drop=True)
        self.layoutChanged.emit()


# ---------- Data helpers ----------
def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace(r"^\s*$", np.nan, regex=True)
    cleaned = cleaned.dropna(how="all")
    return cleaned


def find_col(df: pd.DataFrame, desired: str) -> str | None:
    """
    Finds the correct column in df.

    Supports:
    - exact match on desired
    - fuzzy normalized match
    - mapping from new schema (COLUMN_MAP) into old known names
    """
    if desired in df.columns:
        return desired

    desired_key = normalize_col_name(desired)

    for c in df.columns:
        if normalize_col_name(c) == desired_key:
            return c

    # desired is old name like "Fuel scored" but df has new schema column
    for csv_col, known_name in COLUMN_MAP.items():
        if normalize_col_name(known_name) == desired_key:
            for actual in df.columns:
                if normalize_col_name(actual) == normalize_col_name(csv_col):
                    return actual

    return None


def force_int_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return s.astype(int)


def force_int_value(x) -> int:
    if x is None:
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


def is_known_int_column(col_name: str) -> bool:
    key = normalize_col_name(col_name)
    return key in {
        "team number",
        "fuel scored",
        "robot fuel scored",
        "fuel shot and missed",
        "robot fuel shot and missed",
        "teleop - cycles completed",
        "match number",
        "final alliance score",
        "fouls",
        "offense skill",
        "defensive skill",
    }


# ---------- Climb parsing: No / F / L1 / L2 / L3 ----------
def climb_level_from_value(v) -> int:
    s = str(v).strip().upper()
    if s == "L1":
        return 1
    if s == "L2":
        return 2
    if s == "L3":
        return 3
    return 0  # No / F / blank / anything else


def series_climb_levels(series: pd.Series) -> pd.Series:
    return series.fillna("").map(climb_level_from_value).astype(int)


def level_to_label(level: int) -> str:
    return f"L{level}" if level > 0 else "No"


# ---------- Computations ----------
def compute_fuel_ranking(df: pd.DataFrame) -> pd.DataFrame:
    team_col = find_col(df, "Team Number")

    auto_scored_col = find_col(df, "Auto - Robot Fuel scored")
    auto_missed_col = find_col(df, "Auto - Robot Fuel shot and missed")

    tele_scored_col = find_col(df, "TeleOp - Robot Fuel Scored")
    tele_missed_col = find_col(df, "TeleOp - Robot Fuel Shot and Missed")

    cycles_col = find_col(df, "TeleOp - Cycles Completed")

    missing = []
    for name, col in [
        ("Team Number", team_col),
        ("Auto - Robot Fuel scored", auto_scored_col),
        ("Auto - Robot Fuel shot and missed", auto_missed_col),
        ("TeleOp - Robot Fuel Scored", tele_scored_col),
        ("TeleOp - Robot Fuel Shot and Missed", tele_missed_col),
        ("TeleOp - Cycles Completed", cycles_col),
    ]:
        if col is None:
            missing.append(name)

    if missing:
        raise KeyError("Missing required column(s): " + ", ".join(missing))

    work = df[[team_col, auto_scored_col, auto_missed_col, tele_scored_col, tele_missed_col, cycles_col]].copy()

    work[team_col] = force_int_series(work[team_col])
    work[auto_scored_col] = force_int_series(work[auto_scored_col])
    work[auto_missed_col] = force_int_series(work[auto_missed_col])
    work[tele_scored_col] = force_int_series(work[tele_scored_col])
    work[tele_missed_col] = force_int_series(work[tele_missed_col])
    work[cycles_col] = force_int_series(work[cycles_col])

    grouped = work.groupby(team_col, dropna=False).sum(numeric_only=True)
    match_counts = work.groupby(team_col, dropna=False).size()

    auto_scored = grouped[auto_scored_col]
    auto_attempts = auto_scored + grouped[auto_missed_col]

    tele_scored = grouped[tele_scored_col]
    tele_attempts = tele_scored + grouped[tele_missed_col]

    auto_pct = np.where(
        auto_attempts.to_numpy() > 0,
        (auto_scored.to_numpy() / auto_attempts.to_numpy()) * 100.0,
        np.nan
    )

    tele_pct = np.where(
        tele_attempts.to_numpy() > 0,
        (tele_scored.to_numpy() / tele_attempts.to_numpy()) * 100.0,
        np.nan
    )

    avg_cycles = np.where(
        match_counts.to_numpy() > 0,
        grouped[cycles_col].to_numpy() / match_counts.to_numpy(),
        np.nan
    )

    out = pd.DataFrame({
        "Team Number": grouped.index.astype(int),
        "Auto Fuel Scored": auto_scored.astype(int),
        "Auto Scoring %": auto_pct,
        "TeleOp Fuel Scored": tele_scored.astype(int),
        "Teleop Scoring %": tele_pct,
        "Avg Cycles": avg_cycles,
    })

    out["Auto Scoring %"] = pd.Series(out["Auto Scoring %"]).map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}%"
    )
    out["Teleop Scoring %"] = pd.Series(out["Teleop Scoring %"]).map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}%"
    )
    out["Avg Cycles"] = pd.Series(out["Avg Cycles"]).map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )

    out = out.sort_values(
        by=["TeleOp Fuel Scored", "Teleop Scoring %", "Auto Fuel Scored", "Auto Scoring %"],
        ascending=[False, False, False, False]
    )

    return out.reset_index(drop=True)



def sort_raw_by_team(df: pd.DataFrame) -> pd.DataFrame:
    team_col = find_col(df, "Team Number")
    if team_col is None:
        raise KeyError("Missing required column: Team Number")

    tmp = df.copy()
    tmp["_team_sort_key"] = pd.to_numeric(tmp[team_col], errors="coerce")
    tmp["_team_sort_str"] = tmp[team_col].astype(str)

    tmp = tmp.sort_values(
        by=["_team_sort_key", "_team_sort_str"],
        ascending=[True, True],
        na_position="last"
    ).drop(columns=["_team_sort_key", "_team_sort_str"])

    return tmp


# ---------- Printing helpers (force black on white) ----------
def _collect_widgets(root: QWidget) -> list[QWidget]:
    out = [root]
    out.extend(root.findChildren(QWidget))
    return out


def _force_print_bw(root: QWidget):
    """
    Temporarily removes stylesheets + forces a white/black palette for root and all children.
    Returns restore() callable.
    """
    widgets = _collect_widgets(root)
    saved: list[tuple[QWidget, str, QPalette, bool]] = []

    for w in widgets:
        saved.append((w, w.styleSheet(), w.palette(), w.autoFillBackground()))
        w.setStyleSheet("")  # remove theme colors

        pal = w.palette()
        pal.setColor(QPalette.Window, QColor("white"))
        pal.setColor(QPalette.Base, QColor("white"))
        pal.setColor(QPalette.AlternateBase, QColor("white"))
        pal.setColor(QPalette.Text, QColor("black"))
        pal.setColor(QPalette.WindowText, QColor("black"))
        pal.setColor(QPalette.Button, QColor("white"))
        pal.setColor(QPalette.ButtonText, QColor("black"))
        pal.setColor(QPalette.Highlight, QColor("lightgray"))
        pal.setColor(QPalette.HighlightedText, QColor("black"))

        w.setPalette(pal)
        w.setAutoFillBackground(True)

    def restore():
        for w, ss, pal, autofill in saved:
            w.setStyleSheet(ss)
            w.setPalette(pal)
            w.setAutoFillBackground(autofill)

    return restore


# ---------- Printing / Saving ----------
def print_tableview(parent, title: str, table: QTableView):
    model = table.model()
    if model is None or model.rowCount() == 0 or model.columnCount() == 0:
        QMessageBox.information(parent, "Print", "Nothing to print yet.")
        return

    printer = QPrinter(QPrinter.HighResolution)
    printer.setPageOrientation(QPageLayout.Landscape)

    dialog = QPrintDialog(printer, parent)
    dialog.setWindowTitle(f"Print - {title}")
    if dialog.exec() != QPrintDialog.Accepted:
        return

    painter = QPainter()
    if not painter.begin(printer):
        QMessageBox.critical(parent, "Print Error", "Could not start printer painter.")
        return

    # --- temporarily expand table to full content size so render() prints everything ---
    old_size = table.size()
    old_h_policy = table.horizontalScrollBarPolicy()
    old_v_policy = table.verticalScrollBarPolicy()

    try:
        page_rect = printer.pageRect(QPrinter.DevicePixel)
        painter.fillRect(page_rect, Qt.white)

        # Make sure section sizes are up-to-date
        table.resizeColumnsToContents()
        try:
            table.resizeRowsToContents()
        except Exception:
            pass

        # Disable scrollbars so viewport matches full content size
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        #full_width = table.verticalHeader().width() + table.horizontalHeader().length()
        #full_height = table.horizontalHeader().height() + table.verticalHeader().length()


        full_width = table.verticalHeader().width() + table.horizontalHeader().length() + 20
        full_height = table.horizontalHeader().height() + table.verticalHeader().length() + 40

        # Resize the widget so render() includes all rows/cols (not just visible area)
        table.resize(full_width, full_height)

        sx = page_rect.width() / max(1, full_width)
        sy = page_rect.height() / max(1, full_height)
        scale = min(sx, sy)

        restore = _force_print_bw(table)
        try:
            painter.save()
            painter.translate(page_rect.left(), page_rect.top())
            painter.scale(scale, scale)

            # Qt6/Windows: use the overload with QPoint offset
            table.render(painter, QPoint(0, 0))

            painter.restore()
        finally:
            restore()

    finally:
        # Restore table UI
        table.resize(old_size)
        table.setHorizontalScrollBarPolicy(old_h_policy)
        table.setVerticalScrollBarPolicy(old_v_policy)
        painter.end()



def print_widget(parent, title: str, widget: QWidget):
    printer = QPrinter(QPrinter.HighResolution)
    printer.setPageOrientation(QPageLayout.Portrait)

    dialog = QPrintDialog(printer, parent)
    dialog.setWindowTitle(f"Print - {title}")
    if dialog.exec() != QPrintDialog.Accepted:
        return

    painter = QPainter()
    if not painter.begin(printer):
        QMessageBox.critical(parent, "Print Error", "Could not start printer painter.")
        return

    try:
        page_rect = printer.pageRect(QPrinter.DevicePixel)
        painter.fillRect(page_rect, Qt.white)

        w = max(1, widget.width())
        h = max(1, widget.height())
        sx = page_rect.width() / w
        sy = page_rect.height() / h
        scale = min(sx, sy)

        restore = _force_print_bw(widget)
        try:
            painter.save()
            painter.translate(page_rect.left(), page_rect.top())
            painter.scale(scale, scale)

            widget.render(painter, QPoint(0, 0))
            painter.restore()
        finally:
            restore()
    finally:
        painter.end()


def save_model_as_csv(parent, default_name: str, model: DataFrameModel):
    if model is None or model.rowCount() == 0 or model.columnCount() == 0:
        QMessageBox.information(parent, "Save CSV", "Nothing to save yet.")
        return

    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save as CSV",
        default_name,
        "CSV Files (*.csv);;All Files (*.*)"
    )
    if not path:
        return

    try:
        df = model.dataframe()
        if not path.lower().endswith(".csv"):
            path += ".csv"
        df.to_csv(path, index=False)
    except Exception as e:
        QMessageBox.critical(parent, "Save Error", f"Failed to save CSV:\n{e}")
        return

    QMessageBox.information(parent, "Saved", f"Saved CSV:\n{path}")


def save_scorecard_report(parent, default_name: str, title: str, rows: list[tuple[str, str]]):
    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save Score Card Report",
        default_name,
        "TSV Files (*.tsv);;Text Files (*.txt);;All Files (*.*)"
    )
    if not path:
        return

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(title + "\n")
            f.write("Metric\tValue\n")
            for k, v in rows:
                f.write(f"{k}\t{v}\n")
    except Exception as e:
        QMessageBox.critical(parent, "Save Error", f"Failed to save report:\n{e}")
        return

    QMessageBox.information(parent, "Saved", f"Saved report:\n{path}")
