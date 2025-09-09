#!/usr/bin/env python3
"""GCFScape inspired GUI for browsing and extracting Steam cache files.

This module provides a reasonably feature complete clone of the classic
GCFScape application written in Python using PyQt5.  The goal is to mirror
the look and feel of the original tool while keeping the code portable and
easy to understand.  It intentionally favours readability over raw
performance and serves as a reference implementation of how the original
GCFScape behaves.

Highlights
=========

* Tree based navigation of cache contents with live filtering.
* File preview pane that attempts to display text files and falls back to a
  hexadecimal dump for binary data.
* Context menus, toolbars and menu layout mirroring the original tool.
* Extraction of individual files or entire folders with progress reporting.
* Simple properties dialog for both files and folders.
* Recent file list stored via :class:`~PyQt5.QtCore.QSettings` for a more
  native desktop experience.
* Placeholder implementations of advanced features such as defragmentation
  and validation to keep parity with the original UI.  These placeholders
  can be expanded with real logic if desired.

The code is intentionally verbose and heavily commented to make the control
flow clear.  This also helps align the implementation more closely with the
user interface of GCFScape where many seemingly small behaviours are
performed behind the scenes.
"""

from __future__ import annotations

import os
import sys
import traceback
import tempfile
import shutil
import fnmatch
import re
from pathlib import Path
from typing import Callable, Iterable, List

from PyQt5.QtCore import (
    QObject,
    Qt,
    QThread,
    QSettings,
    pyqtSignal,
    QSize,
    QMimeData,
    QUrl,
    QModelIndex,
    QPoint,
)
from PyQt5.QtGui import QIcon, QCloseEvent, QPixmap, QDesktopServices, QDrag, QMouseEvent
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QActionGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPlainTextEdit,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QCheckBox,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QInputDialog,
    QStyle,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QAbstractItemView,
    QDockWidget,
    QFrame,
    QListView,
)

# The pysteam cache file parser is used to read GCF/NCF archives.  It
# exposes a similar API to the original C++ version used by GCFScape.
from pysteam.fs.cachefile import CacheFile, CacheFileManifestEntry
from pysteam.fs.archive import open_archive
from pysteam.bsp.preview import BSPViewWidget
from pysteam.image import ImageViewWidget
from pysteam.vtf.preview import VTFViewWidget
from pysteam.mdl.preview import MDLViewWidget
from pysteam.hex.preview import HexViewWidget


# Using standard icons instead of QFileIconProvider avoids expensive
# SHGetFileInfo calls on large files.
def is_encrypted(entry) -> bool:
    """Return ``True`` if the manifest flags mark ``entry`` as encrypted."""

    manifest = getattr(entry, "_manifest_entry", None)
    if not manifest:
        return False
    return bool(manifest.directory_flags & CacheFileManifestEntry.FLAG_IS_ENCRYPTED)


# ---------------------------------------------------------------------------
# Utility widgets and helpers
# ---------------------------------------------------------------------------


class EntryItem(QTreeWidgetItem):
    """Tree widget item representing a cache entry.

    The item stores a reference to the underlying cache entry object in its
    ``entry`` attribute.  Additional convenience properties are provided to
    reduce boilerplate when accessing the entry's information.
    """

    def __init__(self, entry) -> None:  # type: ignore[override]
        super().__init__()
        self.entry = entry
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Update the visual representation to match the current entry."""

        name = self.entry.name
        size = str(self.entry.size()) if self.entry.is_file() else ""
        etype = "File" if self.entry.is_file() else "Folder"

        self.setText(0, name)
        self.setText(1, size)
        self.setText(2, etype)
        if self.entry.is_file():
            icon = QApplication.style().standardIcon(QStyle.SP_FileIcon)
        else:
            icon = QApplication.style().standardIcon(QStyle.SP_DirIcon)

        if self.entry.is_file() and name.lower().endswith(".ico"):
            stream = None
            try:
                stream = self.entry.open("rb")
                data = stream.read(self.entry.size())
                pix = QPixmap()
                if pix.loadFromData(data):
                    icon = QIcon(pix)
            except Exception:
                pass
            finally:
                try:
                    stream and stream.close()
                except Exception:
                    pass
        self.setIcon(0, icon)

        flags = getattr(self.entry._manifest_entry, "directory_flags", 0) if hasattr(self.entry, "_manifest_entry") else 0
        encrypted = "Yes" if flags & CacheFileManifestEntry.FLAG_IS_ENCRYPTED else "No"
        copy_local = "Yes" if flags & (CacheFileManifestEntry.FLAG_IS_LOCKED | CacheFileManifestEntry.FLAG_IS_LAUNCH) else "No"
        overwrite = "Yes" if not (flags & CacheFileManifestEntry.FLAG_IS_USER_CONFIG) else "No"
        backup = "Yes" if flags & CacheFileManifestEntry.FLAG_BACKUP_PLZ else "No"
        frag = ""
        if self.entry.is_file():
            try:
                f, u = self.entry.package._get_item_fragmentation(self.entry._manifest_entry.index)
                frag = f"{(f / u * 100):.1f}%" if u else "0%"
            except Exception:
                frag = "0%"
        self.setText(3, encrypted)
        self.setText(4, copy_local)
        self.setText(5, overwrite)
        self.setText(6, backup)
        self.setText(7, hex(flags))
        self.setText(8, frag)

    # ------------------------------------------------------------------
    def __lt__(self, other: QTreeWidgetItem) -> bool:  # type: ignore[override]
        column = self.treeWidget().sortColumn() if self.treeWidget() else 0
        if column == 2 and isinstance(other, QTreeWidgetItem):
            if self.text(2) == other.text(2):
                return self.text(0).lower() < other.text(0).lower()
        return super().__lt__(other)

    # ------------------------------------------------------------------
    def path(self) -> str:
        return self.entry.path()


class FileListWidget(QTreeWidget):
    """Tree widget listing files with drag extraction support."""

    def __init__(self, window) -> None:  # type: ignore[override]
        super().__init__()
        self.window = window
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        
    # ------------------------------------------------------------------
    def startDrag(self, supportedActions: Qt.DropActions) -> None:  # type: ignore[override]
        items = [i for i in self.selectedItems() if isinstance(i, EntryItem)]
        if not items:
            return

        temp_dir = tempfile.mkdtemp(prefix="pysteam_drag_")
        paths = []
        for item in items:
            entry = item.entry
            try:
                if entry.is_file():
                    entry.extract(temp_dir, keep_folder_structure=True)
                else:
                    for f in entry.all_files():
                        f.extract(temp_dir, keep_folder_structure=True)
                rel = entry.path().lstrip("\\/").replace("\\", os.sep)
                paths.append(os.path.join(temp_dir, rel))
            except Exception:
                pass
        if not paths:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec_(Qt.CopyAction)
        self.window._temp_dirs.append(temp_dir)

    # ------------------------------------------------------------------
    def mousePressEvent(self, event):  # type: ignore[override]
        if self.window.view_mode in ("details", "list"):
            index = self.indexAt(event.pos())
            if index.isValid() and index.column() != 0:
                fake = QMouseEvent(
                    event.type(),
                    QPoint(-1, -1),
                    event.globalPos(),
                    event.button(),
                    event.buttons(),
                    event.modifiers(),
                )
                super().mousePressEvent(fake)
                return
        super().mousePressEvent(event)


class ExtractionWorker(QThread):
    """Background worker extracting a list of files.

    The worker reports progress via the :pyattr:`progress` signal and emits
    :pyattr:`finished` or :pyattr:`error` when done.  Extraction can be
    cancelled by calling :py:meth:`cancel`.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        files: Iterable,
        dest: str,
        key: bytes | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.files = list(files)
        self.dest = dest
        self.key = key
        self._cancelled = False

    # ------------------------------------------------------------------
    def run(self) -> None:  # type: ignore[override]
        total = len(self.files)
        for idx, entry in enumerate(self.files, 1):
            if self._cancelled:
                break
            try:
                self.progress.emit(int(idx / total * 100), entry.path())
                if is_encrypted(entry) and self.key is None:
                    raise ValueError("File is encrypted but no key was provided")
                entry.extract(self.dest, keep_folder_structure=True, key=self.key)
            except Exception as exc:  # pragma: no cover - worker thread
                self.error.emit(str(exc))
                return
        self.finished.emit()

    # ------------------------------------------------------------------
    def cancel(self) -> None:
        self._cancelled = True


def _format_bytes(value: int) -> str:
    gb = value / (1024 ** 3)
    mb = value / (1024 ** 2)
    kb = value / 1024
    return f"{gb:.2f} GB / {mb:.2f} MB / {kb:.2f} KB / {value} bytes"


def _count_items(folder) -> tuple[int, int]:
    files = 0
    folders = 0
    for item in folder:
        if item.is_file():
            files += 1
        else:
            f, d = _count_items(item)
            files += f
            folders += d + 1
    return files, folders


def _completion(entry) -> float:
    if entry.is_file():
        total = entry.size()
        manifest = getattr(entry, "_manifest_entry", None)
        if not manifest or total == 0:
            return 100.0
        avail = sum(b.file_data_size for b in manifest.blocks if b)
        return 100.0 * min(1.0, avail / total)
    files = entry.all_files()
    total = sum(f.size() for f in files)
    if total == 0:
        return 100.0
    avail = 0
    for f in files:
        manifest = getattr(f, "_manifest_entry", None)
        if manifest:
            avail += sum(b.file_data_size for b in manifest.blocks if b)
    return 100.0 * min(1.0, avail / total)


def _entry_location(entry) -> str:
    """Return a display-friendly location path starting with ``root``."""

    path = entry.path().lstrip("\\")
    return f"root\\{path}" if path else "root\\"


class PropertiesDialog(QDialog):
    """Dialog showing detailed information about an entry."""

    def __init__(self, entry, window, parent: QWidget | None = None) -> None:
        super().__init__(parent or window)
        self.entry = entry
        self.window = window
        self.cache = window.cachefile
        self.setWindowTitle("Properties")

        layout = QVBoxLayout(self)
        header = QHBoxLayout()

        icon_label = QLabel()
        if entry.is_file():
            icon = QApplication.style().standardIcon(QStyle.SP_FileIcon)
        else:
            icon = QApplication.style().standardIcon(QStyle.SP_DirIcon)
        if entry.is_file() and entry.name.lower().endswith(".ico"):
            stream = None
            try:
                stream = entry.open("rb")
                data = stream.read(entry.size())
                pix = QPixmap()
                if pix.loadFromData(data):
                    icon = QIcon(pix)
            except Exception:
                pass
            finally:
                try:
                    stream and stream.close()
                except Exception:
                    pass
        icon_label.setPixmap(icon.pixmap(48, 48))
        name = entry.name or getattr(self.cache, "filename", "")
        header.addWidget(icon_label)
        header.addWidget(QLabel(name))
        layout.addLayout(header)

        form = QFormLayout()
        layout.addLayout(form)

        if self.cache and entry is self.cache.root:
            form.addRow("Item type:", QLabel("Cache"))
            form.addRow("Location:", QLabel(_entry_location(entry)))
            form.addRow("Size:", QLabel(_format_bytes(entry.size())))
            blocks_used = self.cache.blocks.blocks_used if self.cache.blocks else 0
            sector_size = self.cache.header.sector_size
            form.addRow(
                "Size on disk:",
                QLabel(_format_bytes(blocks_used * sector_size)),
            )
            files, folders = _count_items(entry)
            form.addRow(
                "Contains:",
                QLabel(f"{files + folders} items, {folders} folders"),
            )
            form.addRow(
                "Percent complete:", QLabel(f"{_completion(entry):.0f}%")
            )
            header = self.cache.header
            form.addRow("GCF version:", QLabel(str(header.format_version)))
            form.addRow("Cache ID:", QLabel(str(header.application_id)))
            if self.cache.blocks:
                form.addRow(
                    "Allocated blocks:", QLabel(str(self.cache.blocks.block_count))
                )
                form.addRow(
                    "Used blocks:", QLabel(str(self.cache.blocks.blocks_used))
                )
            form.addRow("Block length:", QLabel(str(header.sector_size)))
            form.addRow(
                "Last played version:", QLabel(str(header.application_version))
            )
            if self.cache.alloc_table:
                allocs = self.cache.alloc_table.sector_count
                form.addRow("Total mapping allocations:", QLabel(str(allocs)))
                form.addRow(
                    "Total mapping memory allocated:",
                    QLabel(_format_bytes(allocs * sector_size)),
                )
                form.addRow(
                    "Total mapping memory used:",
                    QLabel(_format_bytes(blocks_used * sector_size)),
                )
            flags = getattr(self.cache.manifest, "depot_info", 0)
            form.addRow("Flags:", QLabel(hex(flags)))
            form.addRow(
                "Fragmented:",
                QLabel("Yes" if self.cache.is_fragmented() else "No"),
            )
        elif entry.is_folder():
            form.addRow("Item type:", QLabel("Folder"))
            form.addRow("Location:", QLabel(_entry_location(entry)))
            form.addRow("Size:", QLabel(_format_bytes(entry.size())))
            files, folders = _count_items(entry)
            form.addRow(
                "Contains:",
                QLabel(f"{files + folders} items, {folders} folders"),
            )
            sector = self.cache.header.sector_size if self.cache else 0
            size_on_disk = (
                sum(getattr(f, "num_of_blocks", 0) for f in entry.all_files())
                * sector
            )
            form.addRow("Size on disk:", QLabel(_format_bytes(size_on_disk)))
            form.addRow(
                "Total file completion:",
                QLabel(f"{_completion(entry):.0f}%"),
            )
            flags = getattr(getattr(entry, "_manifest_entry", None), "directory_flags", 0)
            form.addRow("Flags:", QLabel(hex(flags)))
            frag = any(getattr(f, "is_fragmented", False) for f in entry.all_files())
            form.addRow("Fragmented:", QLabel("Yes" if frag else "No"))
        else:
            form.addRow("Item type:", QLabel("File"))
            form.addRow("Location:", QLabel(_entry_location(entry)))
            form.addRow("Size:", QLabel(_format_bytes(entry.size())))
            sector = self.cache.header.sector_size if self.cache else 0
            blocks = getattr(entry, "num_of_blocks", 0)
            form.addRow(
                "Size on disk:", QLabel(_format_bytes(blocks * sector))
            )
            comp = _completion(entry)
            form.addRow("Extractable:", QLabel("True" if comp >= 100 else "False"))
            form.addRow("Completion:", QLabel(f"{comp:.0f}%"))
            manifest = getattr(entry, "_manifest_entry", None)
            flags = manifest.directory_flags if manifest else 0
            form.addRow(
                "Is encrypted:",
                QLabel(str(bool(flags & CacheFileManifestEntry.FLAG_IS_ENCRYPTED))),
            )
            form.addRow(
                "Copy locally:",
                QLabel(str(bool(flags & CacheFileManifestEntry.FLAG_IS_NO_CACHE))),
            )
            form.addRow(
                "Overwrite local copy:",
                QLabel(str(bool(flags & CacheFileManifestEntry.FLAG_IS_LOCKED))),
            )
            form.addRow(
                "Backup local copy:",
                QLabel(str(bool(flags & CacheFileManifestEntry.FLAG_BACKUP_PLZ))),
            )
            form.addRow("Flags:", QLabel(hex(flags)))
            form.addRow(
                "Fragmented:",
                QLabel("Yes" if getattr(entry, "is_fragmented", False) else "No"),
            )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class SearchDialog(QDialog):
    """Dialog used for advanced name searching within the tree."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Search")
        self.resize(360, 160)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.pattern = QLineEdit()
        form.addRow("Find what:", self.pattern)
        self.mode = QComboBox()
        self.mode.addItems([
            "Using wildcards",
            "Substring",
            "Whole String",
            "Using Regex",
        ])
        form.addRow("Match:", self.mode)
        self.case = QCheckBox("Case Sensitive")
        form.addRow(self.case)
        layout.addLayout(form)

        buttons = QDialogButtonBox()
        find_btn = buttons.addButton("Find", QDialogButtonBox.AcceptRole)
        cancel_btn = buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        find_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)


class OptionsDialog(QDialog):
    """Very small placeholder for application options.

    Only a tiny subset of the original tool's preferences are implemented.
    Currently a single checkbox allows toggling the preview pane visibility on
    application start-up.  The value is persisted via :class:`QSettings`.
    """

    def __init__(self, settings: QSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Options")
        self.resize(300, 120)

        layout = QVBoxLayout(self)
        self.preview_check = QCheckBox("Show preview at startup")
        self.preview_check.setChecked(self.settings.value("preview", True, bool))
        layout.addWidget(self.preview_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    @property
    def preview_enabled(self) -> bool:
        return self.preview_check.isChecked()

    # ------------------------------------------------------------------
    def accept(self) -> None:  # type: ignore[override]
        self.settings.setValue("preview", self.preview_check.isChecked())
        super().accept()


class PreviewWidget(QWidget):
    """Widget displaying a preview of the currently selected file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.stack = QStackedWidget()
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.image_view = ImageViewWidget()
        self.bsp_view = BSPViewWidget()
        self.vtf_view = VTFViewWidget()
        self.mdl_view = MDLViewWidget()
        self.hex_view = HexViewWidget()
        self.stack.addWidget(self.text_view)
        self.stack.addWidget(self.image_view)
        self.stack.addWidget(self.bsp_view)
        self.stack.addWidget(self.vtf_view)
        self.stack.addWidget(self.mdl_view)
        self.stack.addWidget(self.hex_view)
        layout.addWidget(self.stack)
        self.key_provider: Callable[[], bytes | None] | None = None

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.text_view.clear()
        self.image_view.clear()
        self.bsp_view.clear()
        self.vtf_view.clear()
        self.mdl_view.clear()
        self.hex_view.clear()
        self.stack.setCurrentWidget(self.text_view)

    # ------------------------------------------------------------------
    def set_entry(self, entry) -> None:
        """Display a preview for ``entry`` which may be a BSP or text file."""

        name = entry.name.lower()
        ext = os.path.splitext(name)[1]
        key = None
        # Reset any previous preview to avoid stale state from a failed load.
        self.clear()
        if is_encrypted(entry):
            if self.key_provider:
                key = self.key_provider()
            if key is None:
                return

        try:
            stream = entry.open("rb", key=key)
            data = stream.read(entry.size())
        except Exception:
            return
        finally:
            try:
                stream.close()
            except Exception:
                pass

        IMAGE_EXTS = {".gif", ".jpg", ".jpeg", ".bmp", ".png", ".tga", ".ico"}
        TEXT_EXTS = {
            ".res",
            ".txt",
            ".vmt",
            ".lst",
            ".xml",
            ".vdf",
            ".html",
            ".cfg",
            ".inf",
            ".css",
            ".js",
            ".ts",
        }

        if ext == ".bsp":
            self.bsp_view.load_map(data)
            self.stack.setCurrentWidget(self.bsp_view)
        elif ext == ".vtf":
            self.vtf_view.load_vtf(data)
            self.stack.setCurrentWidget(self.vtf_view)
        elif ext == ".mdl":
            vvd_data = vtx_data = None
            try:
                folder = entry.folder
                base = os.path.splitext(entry.name)[0]
                vvd_entry = folder.items.get(base + ".vvd")
                if vvd_entry:
                    with vvd_entry.open("rb") as s:
                        vvd_data = s.read(vvd_entry.size())
                # VTX files may have platform/LOD suffixes; pick the first match
                for name, ent in folder.items.items():
                    if name.startswith(base) and name.endswith(".vtx"):
                        with ent.open("rb") as s:
                            vtx_data = s.read(ent.size())
                        break
            except Exception:
                pass
            self.mdl_view.load_model(data, vvd_data, vtx_data)
            self.stack.setCurrentWidget(self.mdl_view)
        elif ext in IMAGE_EXTS:
            self.image_view.load_image(data)
            self.stack.setCurrentWidget(self.image_view)
        elif ext in TEXT_EXTS:
            text = data.decode("utf-8", errors="replace")
            self.text_view.setPlainText(text)
            self.stack.setCurrentWidget(self.text_view)
        else:
            self.hex_view.load_data(data)
            self.stack.setCurrentWidget(self.hex_view)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------


class GCFScapeWindow(QMainWindow):
    """Main window implementing the GCFScape GUI."""

    settings = QSettings("pysteam", "gcfscape")

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("GCFScape (Python Edition)")
        self.resize(1000, 700)

        self.cachefile: CacheFile | None = None
        self.current_path: Path | None = None
        self._decryption_key: bytes | None = None
        self.entry_to_tree_item: dict = {}
        self.history: List = []
        self.history_index = -1
        self._suppress_history = False
        self._temp_dirs: List[str] = []
        self._search_mode = False
        self._search_results: List = []
        self._search_pattern = ""

        # ------------------------------------------------------------------
        # Central layout
        # ------------------------------------------------------------------
        splitter = QSplitter(self)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search…")
        self.search.textChanged.connect(self._filter_tree)
        left_layout.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._open_context_menu)
        self.tree.itemSelectionChanged.connect(self._update_file_list)
        self.tree.itemSelectionChanged.connect(self._update_preview)
        left_layout.addWidget(self.tree)

        self.file_list = FileListWidget(self)
        all_columns = [
            "Name",
            "Size",
            "Type",
            "Encrypted",
            "Copy Locally",
            "Overwrite Local Copy",
            "Backup Local Copy",
            "Flags",
            "Fragmentation",
        ]
        self.file_list.setHeaderLabels(all_columns)
        self.file_list.setRootIsDecorated(False)
        self.file_list.setItemsExpandable(False)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._open_context_menu)
        self.file_list.itemSelectionChanged.connect(self._update_preview)
        self.file_list.itemDoubleClicked.connect(self._file_list_double_clicked)
        self.file_list.setSortingEnabled(True)
        header = self.file_list.header()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        for i in range(3, len(all_columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
            self.file_list.setColumnHidden(i, True)
        self.column_map = {name: idx for idx, name in enumerate(all_columns)}

        # Track columns resized manually so automatic sizing can be skipped for
        # them.  Columns are initially auto-sized to their contents but a manual
        # resize by the user will freeze that column width until reset.
        self._manual_columns: set[int] = set()
        self._resizing_columns = False
        header.sectionResized.connect(self._record_column_resize)

        splitter.addWidget(left_widget)
        splitter.addWidget(self.file_list)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        address_layout = QHBoxLayout()
        address_layout.addWidget(QLabel("Address:"))
        self.address = QLineEdit()
        self.address.setReadOnly(True)
        address_layout.addWidget(self.address)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addLayout(address_layout)
        center_layout.addWidget(splitter)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(center_widget)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        main_splitter.addWidget(self.console)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        self.setCentralWidget(main_splitter)

        self.preview_widget = PreviewWidget()
        self.preview_widget.key_provider = self._get_decryption_key
        self.preview_dock = QDockWidget("Preview", self)
        self.preview_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.preview_dock.setWidget(self.preview_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.preview_dock)
        self.preview_dock.hide()

        # ------------------------------------------------------------------
        # Status bar
        # ------------------------------------------------------------------
        status = QStatusBar()
        self.setStatusBar(status)
        self.path_label = QLabel()
        self.size_label = QLabel()
        self.version_label = QLabel()
        self.count_label = QLabel()
        for widget in (
            self.path_label,
            self._status_separator(),
            self.size_label,
            self._status_separator(),
            self.version_label,
            self._status_separator(),
            self.count_label,
        ):
            status.addPermanentWidget(widget)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        status.addPermanentWidget(self.progress_bar)

        # ------------------------------------------------------------------
        # Actions
        # ------------------------------------------------------------------

        self.open_action = QAction("&Open…", self)
        self.open_action.triggered.connect(self._open_file)
        self.close_action = QAction("&Close", self)
        self.close_action.triggered.connect(self._close_file)
        self.exit_action = QAction("E&xit", self)
        self.exit_action.triggered.connect(self.close)

        self.extract_action = QAction("Extract…", self)
        self.extract_action.triggered.connect(lambda: self._extract_entry(self._current_entry()))
        self.extract_all_action = QAction("Extract &All…", self)
        self.extract_all_action.triggered.connect(self._extract_all)

        self.refresh_action = QAction("&Refresh", self)
        self.refresh_action.triggered.connect(self._refresh)
        self.expand_action = QAction("Expand All", self)
        self.expand_action.triggered.connect(self.tree.expandAll)
        self.collapse_action = QAction("Collapse All", self)
        self.collapse_action.triggered.connect(self.tree.collapseAll)

        self.properties_action = QAction("Properties", self)
        self.properties_action.triggered.connect(lambda: self._show_properties(self._current_entry()))


        self.find_action = QAction("&Find…", self)
        self.find_action.triggered.connect(self._open_search_dialog)

        self.defrag_action = QAction("&Defragment…", self)
        self.defrag_action.triggered.connect(self._defragment)

        self.validate_action = QAction("&Validate", self)
        self.validate_action.triggered.connect(self._validate)

        self.convert_v1_action = QAction("Convert to &V1…", self)
        self.convert_v1_action.triggered.connect(lambda: self._convert_gcf(1))

        self.convert_latest_action = QAction("Convert to &Latest…", self)
        self.convert_latest_action.triggered.connect(lambda: self._convert_gcf(6))

        self.options_action = QAction("&Options…", self)
        self.options_action.triggered.connect(self._open_options)

        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self._about)
        self.about_qt_action = QAction("About &Qt", self)
        self.about_qt_action.triggered.connect(QApplication.instance().aboutQt)

        # ------------------------------------------------------------------
        # Menus
        # ------------------------------------------------------------------
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.close_action)

        self.recent_menu = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_menu()

        file_menu.addSeparator()
        file_menu.addAction(self.extract_action)
        file_menu.addAction(self.extract_all_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.find_action)
        edit_menu.addAction(self.refresh_action)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.expand_action)
        view_menu.addAction(self.collapse_action)
        view_menu.addSeparator()
        self.large_icons_action = QAction("Large Icons", self, checkable=True)
        self.small_icons_action = QAction("Small Icons", self, checkable=True)
        self.list_action = QAction("List", self, checkable=True)
        self.details_action = QAction("Details", self, checkable=True)
        self.view_group = QActionGroup(self)
        for act in (self.large_icons_action, self.small_icons_action, self.list_action, self.details_action):
            self.view_group.addAction(act)
        self.details_action.setChecked(True)
        self.large_icons_action.triggered.connect(lambda: self._set_view_mode("large"))
        self.small_icons_action.triggered.connect(lambda: self._set_view_mode("small"))
        self.list_action.triggered.connect(lambda: self._set_view_mode("list"))
        self.details_action.triggered.connect(lambda: self._set_view_mode("details"))
        view_menu.addAction(self.large_icons_action)
        view_menu.addAction(self.small_icons_action)
        view_menu.addAction(self.list_action)
        view_menu.addAction(self.details_action)
        self.columns_menu = view_menu.addMenu("Columns")
        self.column_actions = {}
        for name in [
            "Encrypted",
            "Copy Locally",
            "Overwrite Local Copy",
            "Backup Local Copy",
            "Flags",
            "Fragmentation",
        ]:
            act = QAction(name, self, checkable=True)
            act.toggled.connect(lambda checked, n=name: self._toggle_column(n, checked))
            self.column_actions[name] = act
            self.columns_menu.addAction(act)

        tools_menu = menubar.addMenu("&Tools")
        batch_menu = tools_menu.addMenu("Batch")
        batch_frag = QAction("Fragmentation Report", self)
        batch_frag.triggered.connect(self._batch_fragmentation)
        batch_defrag = QAction("Defragment", self)
        batch_defrag.triggered.connect(self._batch_defragment)
        batch_validate = QAction("Validate", self)
        batch_validate.triggered.connect(self._batch_validate)
        batch_menu.addAction(batch_frag)
        batch_menu.addAction(batch_defrag)
        batch_menu.addAction(batch_validate)
        tools_menu.addAction(self.defrag_action)
        tools_menu.addAction(self.validate_action)
        tools_menu.addAction(self.convert_v1_action)
        tools_menu.addAction(self.convert_latest_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.options_action)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.about_action)
        help_menu.addAction(self.about_qt_action)

        # ------------------------------------------------------------------
        # Toolbar mirroring the File menu
        # ------------------------------------------------------------------
        toolbar = QToolBar("Main", self)
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.close_action)
        toolbar.addSeparator()
        toolbar.addAction(self.extract_action)
        toolbar.addAction(self.extract_all_action)
        toolbar.addSeparator()
        toolbar.addAction(self.refresh_action)

        self.addToolBarBreak()
        nav_bar = QToolBar("Navigate", self)
        nav_bar.setIconSize(QSize(16, 16))
        self.addToolBar(nav_bar)

        self.back_action_nav = QAction(
            QApplication.style().standardIcon(QStyle.SP_ArrowBack), "Back", self
        )
        self.back_action_nav.triggered.connect(self._go_back)
        nav_bar.addAction(self.back_action_nav)

        self.forward_action_nav = QAction(
            QApplication.style().standardIcon(QStyle.SP_ArrowForward), "Forward", self
        )
        self.forward_action_nav.triggered.connect(self._go_forward)
        nav_bar.addAction(self.forward_action_nav)

        self.up_action_nav = QAction(
            QApplication.style().standardIcon(QStyle.SP_ArrowUp), "Up", self
        )
        self.up_action_nav.triggered.connect(self._go_up)
        nav_bar.addAction(self.up_action_nav)

        nav_bar.addSeparator()
        nav_bar.addAction(self.find_action)

        self.back_action_nav.setEnabled(False)
        self.forward_action_nav.setEnabled(False)
        self.up_action_nav.setEnabled(False)

        # Drag and drop support for convenience
        self.setAcceptDrops(True)
        self._set_view_mode("details")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Append ``message`` to the console output pane."""

        self.console.appendPlainText(message)

    # ------------------------------------------------------------------
    def _status_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep

    def _current_entry(self):
        """Return the entry associated with the current selection."""
        item = self.file_list.currentItem()
        if isinstance(item, EntryItem):
            return item.entry
        item = self.tree.currentItem()
        if isinstance(item, EntryItem):
            return item.entry
        return None

    def _update_status_info(self) -> None:
        if not self.cachefile or not self.current_path:
            self.path_label.clear()
            self.size_label.clear()
            self.version_label.clear()
            self.count_label.clear()
            return
        complete, total = self.cachefile.count_complete_files()
        size = self.current_path.stat().st_size if self.current_path.exists() else 0
        if size >= 1024 ** 3:
            size_str = f"{size / (1024 ** 3):.2f} GB"
        elif size >= 1024 ** 2:
            size_str = f"{size / (1024 ** 2):.2f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} bytes"
        version = getattr(self.cachefile.header, "format_version", "?")
        self.path_label.setText(str(self.current_path))
        self.size_label.setText(size_str)
        self.version_label.setText(f"v{version}")
        self.count_label.setText(f"{complete} / {total}")

    def _navigate_to(self, entry, record: bool = True) -> None:
        item = self.entry_to_tree_item.get(entry)
        if not item:
            return
        self._suppress_history = not record
        self.tree.setCurrentItem(item)
        self._suppress_history = False

    def _open_entry(self, entry) -> None:
        try:
            temp_dir = tempfile.mkdtemp(prefix="pysteam_open_")
            entry.extract(temp_dir, keep_folder_structure=False)
            path = os.path.join(temp_dir, entry.name)
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            self._temp_dirs.append(temp_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Open", str(exc))

    def _select_entry(self, entry) -> None:
        if entry.is_file():
            parent = entry.folder
            self._navigate_to(parent)
            for i in range(self.file_list.topLevelItemCount()):
                item = self.file_list.topLevelItem(i)
                if isinstance(item, EntryItem) and item.entry is entry:
                    self.file_list.setCurrentItem(item)
                    break
        else:
            self._navigate_to(entry)

    def _perform_search(self, pattern: str, mode: str, case: bool) -> None:
        root = self.cachefile.root if self.cachefile else None
        if not root or not pattern:
            return
        results = []
        patt = pattern if case else pattern.lower()
        stack = [root]
        while stack:
            entry = stack.pop()
            name = entry.name if case else entry.name.lower()
            match = False
            if mode == "Using wildcards":
                match = fnmatch.fnmatchcase(name, patt)
            elif mode == "Substring":
                match = patt in name
            elif mode == "Whole String":
                match = name == patt
            elif mode == "Using Regex":
                flags = 0 if case else re.IGNORECASE
                match = re.search(pattern, entry.name, flags) is not None
            if match:
                results.append(entry)
            if entry.is_folder():
                stack.extend(entry.items.values())
        if results:
            self._display_search_results(pattern, results)
        else:
            QMessageBox.information(self, "Search", "No results found.")

    def _display_search_results(self, pattern: str, results: List) -> None:
        self._search_mode = True
        self._search_results = results
        self._search_pattern = pattern
        self.file_list.clear()
        for entry in results:
            self.file_list.addTopLevelItem(EntryItem(entry))
        self.address.setText(f"Search results for '{pattern}'")
        msg = f"Search for '{pattern}' returned {len(results)} results."
        self.statusBar().showMessage(msg)
        self._log(msg)
        self.preview_widget.clear()
        self.preview_dock.hide()
        self._resize_columns()

    def _go_back(self) -> None:
        if self.history_index > 0:
            self.history_index -= 1
            self._navigate_to(self.history[self.history_index], record=False)

    def _go_forward(self) -> None:
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self._navigate_to(self.history[self.history_index], record=False)

    def _go_up(self) -> None:
        folder = self._current_directory()
        if folder and getattr(folder, "owner", None):
            self._navigate_to(folder.owner)

    def _set_view_mode(self, mode: str) -> None:
        self.view_mode = mode
        if mode == "details":
            self.file_list.setColumnCount(len(self.column_map))
            self.file_list.setHeaderHidden(False)
            self.file_list.header().setSectionResizeMode(0, QHeaderView.Interactive)
            for name, idx in self.column_map.items():
                if idx < 3:
                    self.file_list.setColumnHidden(idx, False)
                else:
                    action = self.column_actions.get(name)
                    self.file_list.setColumnHidden(idx, not (action and action.isChecked()))
            self.file_list.setIconSize(QSize(16, 16))
            self._resize_columns()
        else:
            self.file_list.setHeaderHidden(True)
            self.file_list.setColumnCount(1)
            if hasattr(self.file_list, "setViewMode"):
                if mode == "list":
                    self.file_list.setViewMode(QListView.ListMode)
                    if hasattr(self.file_list, "setWrapping"):
                        self.file_list.setWrapping(False)
                else:
                    self.file_list.setViewMode(QListView.IconMode)
                    if hasattr(self.file_list, "setFlow"):
                        self.file_list.setFlow(QListView.LeftToRight)
                    if hasattr(self.file_list, "setWrapping"):
                        self.file_list.setWrapping(True)
                    if hasattr(self.file_list, "setResizeMode"):
                        self.file_list.setResizeMode(QListView.Adjust)
            if mode == "large":
                self.file_list.setIconSize(QSize(64, 64))
            elif mode == "small":
                self.file_list.setIconSize(QSize(16, 16))
            else:  # list
                self.file_list.setIconSize(QSize(32, 32))

    def _toggle_column(self, name: str, checked: bool) -> None:
        idx = self.column_map.get(name)
        if idx is None:
            return
        self.file_list.setColumnHidden(idx, not checked)
        self._resize_columns()

    # ------------------------------------------------------------------
    def _record_column_resize(self, index: int, old: int, new: int) -> None:
        if not self._resizing_columns:
            self._manual_columns.add(index)

    # ------------------------------------------------------------------
    def _resize_columns(self) -> None:
        if self.view_mode != "details":
            return
        self._resizing_columns = True
        try:
            for idx in range(self.file_list.columnCount()):
                if idx in self._manual_columns:
                    continue
                if self.file_list.isColumnHidden(idx):
                    continue
                self.file_list.resizeColumnToContents(idx)
        finally:
            self._resizing_columns = False

    # ------------------------------------------------------------------
    def dragEnterEvent(self, event):  # type: ignore[override]
        event.ignore()

    # ------------------------------------------------------------------
    def dropEvent(self, event):  # type: ignore[override]
        event.ignore()

    # ------------------------------------------------------------------
    # Menu building helpers
    # ------------------------------------------------------------------

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self.settings.value("recent", [], list)
        for path in recent:
            action = QAction(path, self)
            action.triggered.connect(lambda checked=False, p=path: self._load_file(Path(p)))
            self.recent_menu.addAction(action)
        if not recent:
            self.recent_menu.setEnabled(False)
        else:
            self.recent_menu.setEnabled(True)

    # ------------------------------------------------------------------
    def _add_to_recent(self, path: Path) -> None:
        recent = self.settings.value("recent", [], list)
        path_str = str(path)
        if path_str in recent:
            recent.remove(path_str)
        recent.insert(0, path_str)
        self.settings.setValue("recent", recent[:10])
        self._rebuild_recent_menu()

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open archive file",
            "",
            "Archive Files (*.gcf *.ncf *.vpk *.pak *.wad *.xzp *.zip *.7z *.gz *.tar *.rar);;All Files (*)",
        )
        if path:
            self._load_file(Path(path))

    # ------------------------------------------------------------------
    def _load_file(self, path: Path) -> None:
        self._decryption_key = None
        try:
            if path.suffix.lower() in {".gcf", ".ncf", ".vpk"}:
                self.cachefile = CacheFile.parse(path)
            else:
                self.cachefile = open_archive(path)
        except Exception as exc:  # pragma: no cover - GUI feedback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.current_path = path
        self._add_to_recent(path)
        self.statusBar().showMessage(str(path))
        self.history.clear()
        self.history_index = -1
        self._populate_tree()
        self._update_status_info()

    # ------------------------------------------------------------------
    def _close_file(self) -> None:
        if self.cachefile:
            try:
                self.cachefile.close()
            except Exception:
                pass
        self.cachefile = None
        self.current_path = None
        self.tree.clear()
        self.file_list.clear()
        self.preview_widget.clear()
        self.preview_dock.hide()
        self._decryption_key = None
        self.statusBar().clearMessage()
        self.entry_to_tree_item.clear()
        self.history.clear()
        self.history_index = -1
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._temp_dirs.clear()
        self.back_action_nav.setEnabled(False)
        self.forward_action_nav.setEnabled(False)
        self.up_action_nav.setEnabled(False)
        self._update_status_info()

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        if self.cachefile:
            self._populate_tree()

    # ------------------------------------------------------------------
    # Tree and search functionality
    # ------------------------------------------------------------------

    def _populate_tree(self) -> None:
        self.tree.clear()
        self.file_list.clear()
        self.entry_to_tree_item.clear()
        if not self.cachefile:
            return

        root_entry = self.cachefile.root
        root_item = EntryItem(root_entry)
        root_item.setText(0, "root")
        self.tree.addTopLevelItem(root_item)
        self.entry_to_tree_item[root_entry] = root_item

        def add_dirs(folder, parent_item):
            for name, entry in sorted(folder.items.items()):
                if entry.is_folder():
                    item = EntryItem(entry)
                    parent_item.addChild(item)
                    self.entry_to_tree_item[entry] = item
                    add_dirs(entry, item)

        add_dirs(root_entry, root_item)
        self.tree.setCurrentItem(root_item)
        self._update_file_list()
        self.tree.collapseAll()
        self._filter_tree(self.search.text())

    def _current_directory(self):
        item = self.tree.currentItem()
        if isinstance(item, EntryItem):
            return item.entry
        return None

    def _update_file_list(self) -> None:
        if self._search_mode and self.sender() is self.tree:
            self._search_mode = False
            self._search_results = []
            self._search_pattern = ""
        self.file_list.clear()
        if self._search_mode:
            for entry in self._search_results:
                self.file_list.addTopLevelItem(EntryItem(entry))
            self.preview_widget.clear()
            self.preview_dock.hide()
            self.address.setText(f"Search results for '{self._search_pattern}'")
            self.statusBar().showMessage(
                f"Search for '{self._search_pattern}' returned {len(self._search_results)} results."
            )
            self._resize_columns()
            return
        folder = self._current_directory()
        if not folder:
            return
        for name, entry in sorted(folder.items.items()):
            self.file_list.addTopLevelItem(EntryItem(entry))
        self.preview_widget.clear()
        self.preview_dock.hide()
        path = folder.path().replace("/", "\\")
        if not path.startswith("root\\"):
            path = "root\\" + path.lstrip("\\")
        self.address.setText(path)
        self.statusBar().showMessage(
            f"{_entry_location(folder)} ({len(folder.items)} items)"
        )

        if not self._suppress_history:
            if self.history_index == -1 or self.history[self.history_index] is not folder:
                self.history = self.history[: self.history_index + 1]
                self.history.append(folder)
                self.history_index += 1
        self.back_action_nav.setEnabled(self.history_index > 0)
        self.forward_action_nav.setEnabled(self.history_index < len(self.history) - 1)
        root = self.cachefile.root if self.cachefile else None
        self.up_action_nav.setEnabled(folder is not root)
        self._resize_columns()

    def _file_list_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if not isinstance(item, EntryItem):
            return
        entry = item.entry
        if self._search_mode:
            if entry.is_folder():
                self._search_mode = False
                self._search_results = []
                self._search_pattern = ""
                self._navigate_to(entry)
            else:
                self._open_entry(entry)
            return
        if entry.is_folder():
            tree_item = self.entry_to_tree_item.get(entry)
            if tree_item:
                self.tree.setCurrentItem(tree_item)
        else:
            self._open_entry(entry)

    # ------------------------------------------------------------------
    def _filter_tree(self, text: str) -> None:
        text = text.lower()

        def filter_item(item: QTreeWidgetItem) -> bool:
            match = text in item.text(0).lower() if text else True
            child_match = any(filter_item(item.child(i)) for i in range(item.childCount()))
            item.setHidden(not (match or child_match))
            return match or child_match

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            filter_item(root.child(i))

    # ------------------------------------------------------------------
    def _open_search_dialog(self) -> None:
        if not self.cachefile:
            return
        dialog = SearchDialog(self)
        if dialog.exec() == QDialog.Accepted:
            pattern = dialog.pattern.text()
            mode = dialog.mode.currentText()
            case = dialog.case.isChecked()
            self._perform_search(pattern, mode, case)

    # ------------------------------------------------------------------
    def _update_preview(self) -> None:
        source = self.sender()
        entry = None
        if isinstance(source, QTreeWidget):
            item = source.currentItem()
            if isinstance(item, EntryItem):
                entry = item.entry
        if not entry:
            entry = self._current_entry()
        if not entry or entry.is_folder():
            self.preview_widget.clear()
            self.preview_dock.hide()
            return

        try:
            self.preview_widget.set_entry(entry)
            self.preview_dock.show()
        except Exception as exc:
            self.preview_widget.deleteLater()
            self.preview_widget = PreviewWidget()
            self.preview_widget.key_provider = self._get_decryption_key
            self.preview_dock.setWidget(self.preview_widget)
            self.preview_dock.hide()
            self._log(f"Preview error for {entry.path()}: {exc}")
        self.statusBar().showMessage(_entry_location(entry))

    # ------------------------------------------------------------------
    # Context menu and actions
    # ------------------------------------------------------------------

    def _open_context_menu(self, pos) -> None:
        widget = self.sender()
        item = widget.itemAt(pos) if isinstance(widget, QTreeWidget) else None
        if not isinstance(item, EntryItem):
            return

        entry = item.entry

        menu = QMenu(self)
        extract_action = QAction("Extract…", self)
        extract_action.triggered.connect(lambda: self._extract_entry(entry))
        menu.addAction(extract_action)

        copy_name_action = QAction("Copy Name", self)
        copy_name_action.triggered.connect(lambda: self._copy_text(entry.name))
        menu.addAction(copy_name_action)

        copy_path_action = QAction("Copy Path", self)
        copy_path_action.triggered.connect(lambda: self._copy_text(entry.path()))
        menu.addAction(copy_path_action)

        if self._search_mode and widget is self.file_list:
            goto_action = QAction("Go To Directory", self)
            goto_action.triggered.connect(lambda: self._go_to_directory(entry))
            menu.addAction(goto_action)

        menu.addSeparator()
        props_action = QAction("Properties", self)
        props_action.triggered.connect(lambda: self._show_properties(entry))
        menu.addAction(props_action)

        viewport = widget.viewport() if isinstance(widget, QTreeWidget) else None
        if viewport:
            menu.exec(viewport.mapToGlobal(pos))

    # ------------------------------------------------------------------
    def _go_to_directory(self, entry) -> None:
        folder = entry.folder if entry.is_file() else entry
        self._search_mode = False
        self._search_results = []
        self._search_pattern = ""
        self._navigate_to(folder)

    # ------------------------------------------------------------------
    def _extract_all(self) -> None:
        if not self.cachefile:
            return
        self._extract_entry(self.cachefile.root)

    # ------------------------------------------------------------------
    def _extract_entry(self, entry) -> None:
        if not entry:
            return
        dest = QFileDialog.getExistingDirectory(self, "Select destination")
        if not dest:
            return

        if entry.is_file():
            files = [entry]
        else:
            files = entry.all_files()

        key = None
        if any(is_encrypted(f) for f in files):
            key = self._get_decryption_key()
            if key is None:
                return

        worker = ExtractionWorker(files, dest, key, self)
        progress = QProgressDialog("Extracting…", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.canceled.connect(worker.cancel)
        worker.progress.connect(lambda val, text: (progress.setValue(val), progress.setLabelText(text)))
        worker.error.connect(lambda msg: (self._log(f"Error: {msg}"), QMessageBox.critical(self, "Error", msg)))
        worker.finished.connect(progress.close)

        self._log(f"Extracting {len(files)} file(s) to {dest}")
        worker.start()
        progress.exec()

        if not worker.isRunning():
            QMessageBox.information(self, "Extraction complete", f"Extracted to {dest}")
            self._log(f"Extraction complete: {dest}")

    # ------------------------------------------------------------------
    def _get_decryption_key(self) -> bytes | None:
        if self._decryption_key is not None:
            return self._decryption_key
        text, ok = QInputDialog.getText(self, "Encrypted file", "Enter decryption key:")
        if not ok or not text:
            return None
        try:
            self._decryption_key = bytes.fromhex(text)
        except ValueError:
            self._decryption_key = text.encode("utf-8")
        return self._decryption_key

    # ------------------------------------------------------------------
    def _copy_text(self, text: str) -> None:
        """Copy arbitrary text to the clipboard and update the status bar."""

        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"Copied: {text}", 3000)

    # ------------------------------------------------------------------
    def _show_properties(self, entry) -> None:
        if not entry:
            return
        dialog = PropertiesDialog(entry, self)
        dialog.exec()

    # ------------------------------------------------------------------
    def _defragment(self) -> None:
        """Create a defragmented copy of the currently loaded archive."""

        if not self.cachefile:
            return

        if not self.cachefile.is_fragmented():
            QMessageBox.information(self, "Defragment", "Archive is already defragmented.")
            return

        default = str(self.current_path.with_suffix(".defrag.gcf")) if self.current_path else ""
        path, _ = QFileDialog.getSaveFileName(self, "Save Defragmented GCF", default, "GCF Files (*.gcf)")
        if not path:
            return

        progress = QProgressDialog("Defragmenting…", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def cb(i, total):
            if progress.maximum() != total:
                progress.setMaximum(total)
            progress.setValue(i)
            QApplication.processEvents()

        try:
            self._log(f"Defragmenting to {path}")
            self.cachefile.defragment(path, progress=cb)
        except Exception as exc:  # pragma: no cover - GUI feedback
            traceback.print_exc()
            QMessageBox.critical(self, "Defragment", str(exc))
            self._log(f"Defragment error: {exc}")
            progress.close()
            return
        progress.close()
        QMessageBox.information(self, "Defragment", f"Defragmented archive written to {path}")
        self._log(f"Defragment complete: {path}")

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """Validate the currently loaded archive and report any errors."""

        if not self.cachefile:
            return

        files = list(self.cachefile.root.all_files())
        progress = QProgressDialog("Validating…", None, 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def cb(i, total):
            progress.setValue(i)
            QApplication.processEvents()

        errors = self.cachefile.validate(progress=cb)
        progress.close()
        if errors:
            text = "\n".join(errors[:20])
            QMessageBox.warning(
                self,
                "Validate",
                f"Problems were detected in the archive:\n{text}",
            )
            self._log(f"Validation errors:\n{text}")
        else:
            QMessageBox.information(
                self,
                "Validate",
                "Archive appears to be valid.",
            )
            self._log("Validation complete: no errors found")

    # ------------------------------------------------------------------
    def _convert_gcf(self, target_version: int) -> None:
        """Convert the loaded GCF to a given format version."""

        if not self.cachefile or not self.cachefile.is_gcf():
            QMessageBox.warning(self, "Convert", "No GCF archive loaded.")
            return

        default = os.path.splitext(self.cachefile.filename)[0] + f"_v{target_version}.gcf"
        path, _ = QFileDialog.getSaveFileName(self, "Save Converted GCF", default, "GCF Files (*.gcf)")
        if not path:
            return

        total = 0
        if self.cachefile.data_header is not None:
            total = (
                self.cachefile.data_header.sectors_used
                * self.cachefile.data_header.sector_size
            )
        progress = QProgressDialog("Converting…", None, 0, total, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def cb(written, total_bytes):
            if progress.maximum() != total_bytes:
                progress.setMaximum(total_bytes)
            progress.setValue(written)
            QApplication.processEvents()

        try:
            self._log(f"Converting to v{target_version} -> {path}")
            self.cachefile.convert_version(target_version, path, progress=cb)
            QMessageBox.information(self, "Convert", "Conversion completed.")
            self._log(f"Conversion complete: {path}")
        except NotImplementedError:
            QMessageBox.warning(self, "Convert", "Conversion is not yet implemented in this build.")
            self._log("Conversion not implemented in this build")
        except Exception as exc:
            QMessageBox.critical(self, "Convert", f"Conversion failed: {exc}")
            self._log(f"Conversion failed: {exc}")
        finally:
            progress.close()

    # ------------------------------------------------------------------
    def _batch_fragmentation(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Cache Files",
            "",
            "Cache Files (*.gcf *.ncf)",
        )
        for path in paths:
            try:
                cf = CacheFile.parse(path)
                fragmented, used = cf._get_item_fragmentation(0)
                percent = (fragmented / used * 100.0) if used else 0.0
                status = "Fragmented" if fragmented else "Complete"
                self._log(f"{path}: {percent:.2f}% ({status})")
                cf.close()
            except Exception as exc:
                self._log(f"{path}: error {exc}")

    # ------------------------------------------------------------------
    def _batch_defragment(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Cache Files",
            "",
            "Cache Files (*.gcf *.ncf)",
        )
        for path in paths:
            try:
                cf = CacheFile.parse(path)
                if not cf.is_gcf():
                    self._log(f"{path}: skipping (not a GCF archive)")
                    continue
                if not cf.is_fragmented():
                    self._log(f"{path}: already defragmented")
                    continue
                tmp = path + ".defrag"
                cf.defragment(tmp)
                cf.close()
                os.replace(tmp, path)
                cf2 = CacheFile.parse(path)
                fragmented, used = cf2._get_item_fragmentation(0)
                percent = (fragmented / used * 100.0) if used else 0.0
                cf2.close()
                self._log(f"{path}: defragmented ({percent:.2f}% fragmented)")
            except Exception as exc:
                self._log(f"{path}: defragment failed ({exc})")

    # ------------------------------------------------------------------
    def _batch_validate(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Cache Files",
            "",
            "Cache Files (*.gcf *.ncf)",
        )
        for path in paths:
            try:
                cf = CacheFile.parse(path)
                errors = cf.validate()
                cf.close()
                if errors:
                    self._log(f"{path}: {len(errors)} errors")
                    for e in errors:
                        self._log(f"  - {e}")
                else:
                    self._log(f"{path}: validation successful")
            except Exception as exc:
                self._log(f"{path}: validation failed ({exc})")

    # ------------------------------------------------------------------
    def _open_options(self) -> None:
        dialog = OptionsDialog(self.settings, self)
        if dialog.exec() == QDialog.Accepted:
            pass

    # ------------------------------------------------------------------
    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About GCFScape (Python)",
            "<b>GCFScape (Python Edition)</b><br>"
            "A Qt based reimplementation of the classic GCFScape tool.",
        )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if self.cachefile:
            res = QMessageBox.question(
                self,
                "Quit",
                "Close the current archive and exit?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                event.ignore()
                return
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        event.accept()


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    """Run the GUI application."""

    app = QApplication(argv or sys.argv)
    window = GCFScapeWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())

