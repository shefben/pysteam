"""Qt widget that renders a very small 3D preview of Valve MDL models.

Originally this module only displayed a wireframe of the model's bounding
box.  It has been rewritten to borrow logic from the Source SDK's *Half-Life
Model Viewer* so that Source-engine models are rendered using their actual
geometry.  Only static meshes are supported – animation, lighting and other
advanced features remain out of scope, but even a simple shaded mesh is far
more informative than a plain box when browsing large archives.
"""

from __future__ import annotations

import struct
from typing import List, Sequence, Tuple

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from . import detect_engine

# ``pyqtgraph`` and ``numpy`` are optional dependencies.  Import them lazily so
# we can show a helpful message when missing.
try:  # pragma: no cover - optional dependencies
    import numpy as np  # type: ignore
    import pyqtgraph.opengl as gl  # type: ignore
except Exception:  # pragma: no cover - missing optional deps
    np = None  # type: ignore
    gl = None  # type: ignore

Vector = Tuple[float, float, float]


class MDLViewWidget(QWidget):
    """Widget capable of displaying a crude 3D MDL preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if gl is None or np is None:
            self.view: QWidget = QLabel("pyqtgraph or numpy module missing")
            layout.addWidget(self.view)
        else:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor("k")
            layout.addWidget(self.view)
            self._items: List[gl.GLGraphicsItem] = []

    # ------------------------------------------------------------------
    def clear(self) -> None:
        if gl and hasattr(self, "_items"):
            for item in self._items:
                self.view.removeItem(item)
            self._items.clear()

    # ------------------------------------------------------------------
    def load_model(
        self,
        mdl_data: bytes,
        vvd_data: bytes | None = None,
        vtx_data: bytes | None = None,
    ) -> None:
        """Render an MDL model.

        Parameters
        ----------
        mdl_data:
            Contents of the ``.mdl`` file.
        vvd_data, vtx_data:
            Optional ``.vvd`` and ``.vtx`` companions for Source engine
            models.  If supplied the actual mesh is rendered; otherwise the
            method falls back to drawing the model's bounding box.
        """

        engine = detect_engine(mdl_data)
        bbox = None
        if engine == "Goldsrc":
            bbox = self._goldsrc_bounds(mdl_data)
        elif engine == "Source":
            # Attempt full geometry rendering if companion files were
            # provided.  Fall back to bounding box otherwise.
            if vvd_data and vtx_data:
                try:
                    verts, faces = self._source_geometry(vvd_data, vtx_data)
                except Exception:
                    verts = faces = None
                if verts is not None and faces is not None:
                    self._render_mesh(verts, faces)
                    bbox = (verts.min(axis=0), verts.max(axis=0))
            if bbox is None:
                bbox = self._source_bounds(mdl_data)

        if bbox is None:
            if isinstance(self.view, QLabel):
                self.view.setText("Unsupported MDL")
            return

        if not (gl and np):
            if isinstance(self.view, QLabel):
                self.view.setText("pyqtgraph or numpy module missing")
            return

        if engine != "Source" or not self._items:
            # Fall back to wireframe bounding box rendering.
            self.clear()
            self.view.setToolTip(f"{engine} model")

            (min_x, min_y, min_z), (max_x, max_y, max_z) = bbox
            verts = np.array([
                [min_x, min_y, min_z],
                [max_x, min_y, min_z],
                [max_x, max_y, min_z],
                [min_x, max_y, min_z],
                [min_x, min_y, max_z],
                [max_x, min_y, max_z],
                [max_x, max_y, max_z],
                [min_x, max_y, max_z],
            ], dtype=float)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for a, b in edges:
                pts = verts[[a, b]]
                item = gl.GLLinePlotItem(pos=pts, color=(1, 1, 1, 1), width=1, mode="line_strip")
                self.view.addItem(item)
                self._items.append(item)

        center = [
            (bbox[0][0] + bbox[1][0]) / 2,
            (bbox[0][1] + bbox[1][1]) / 2,
            (bbox[0][2] + bbox[1][2]) / 2,
        ]
        size = max(bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][2] - bbox[0][2]) or 1.0
        self.view.opts["center"] = center
        self.view.opts["distance"] = size * 2

    # ------------------------------------------------------------------
    def _render_mesh(self, verts: np.ndarray, faces: np.ndarray) -> None:
        """Render a mesh using ``pyqtgraph``."""

        if not (gl and np):  # pragma: no cover - checked earlier
            return

        self.clear()
        mesh = gl.GLMeshItem(vertexes=verts, faces=faces, color=(1, 1, 1, 1), smooth=False)
        self.view.addItem(mesh)
        self._items.append(mesh)

    # ------------------------------------------------------------------
    def _source_geometry(
        self, vvd_data: bytes, vtx_data: bytes
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return vertices and triangular faces for a Source model."""

        verts = self._parse_vvd(vvd_data)
        faces = self._parse_vtx(vtx_data)
        return verts, faces

    # ------------------------------------------------------------------
    def _parse_vvd(self, data: bytes) -> np.ndarray:
        """Extract vertex positions from a ``.vvd`` file."""

        if len(data) < 60:
            raise ValueError("vvd too small")

        header = struct.unpack_from("<3i", data, 0)
        if header[0] != 0x44535644:  # 'IDSV'
            raise ValueError("invalid vvd header")
        num_lod_verts = struct.unpack_from("<8i", data, 16)
        num_fixups, fixup_ofs, vertex_ofs, _tangent = struct.unpack_from("<4i", data, 48)
        vert_size = 48

        def read_vertex(idx: int) -> Sequence[float]:
            base = vertex_ofs + idx * vert_size + 16  # skip bone weights
            return struct.unpack_from("<3f", data, base)

        if num_fixups:
            verts: list[Sequence[float]] = []
            for i in range(num_fixups):
                lod, src, count = struct.unpack_from("<3i", data, fixup_ofs + i * 12)
                if lod != 0:
                    continue
                for v in range(count):
                    verts.append(read_vertex(src + v))
        else:
            count = num_lod_verts[0]
            verts = [read_vertex(i) for i in range(count)]

        return np.array(verts, dtype=float)

    # ------------------------------------------------------------------
    def _parse_vtx(self, data: bytes) -> np.ndarray:
        """Extract triangle indices from a ``.vtx`` file."""

        if len(data) < 28:
            raise ValueError("vtx too small")

        id_, version, checksum, num_lods, mat_ofs, num_bps, bp_ofs = struct.unpack_from(
            "<7I", data, 0
        )
        if id_ != 0x54585644:  # 'VTXD'
            # earlier versions used 'IDSV'
            pass

        faces: list[int] = []
        bp_hdr_size = 8
        mdl_hdr_size = 8
        lod_hdr_size = 12
        mesh_hdr_size = 12
        sg_hdr_size = 28
        vert_size = 16

        for bp in range(num_bps):
            bp_off = bp_ofs + bp * bp_hdr_size
            num_models, mdl_ofs = struct.unpack_from("<2I", data, bp_off)
            for m in range(num_models):
                mdl_off = bp_off + mdl_ofs + m * mdl_hdr_size
                num_lods_m, lod_ofs = struct.unpack_from("<2I", data, mdl_off)
                lod_off = mdl_off + lod_ofs
                num_meshes, mesh_ofs, _switch = struct.unpack_from("<2If", data, lod_off)
                for mesh in range(num_meshes):
                    mesh_off = lod_off + mesh_ofs + mesh * mesh_hdr_size
                    num_sgs, sg_ofs, _flags = struct.unpack_from("<2I B", data, mesh_off)
                    sg_ofs = mesh_off + sg_ofs
                    for sg in range(num_sgs):
                        sg_off = sg_ofs + sg * sg_hdr_size
                        (
                            num_verts,
                            vert_ofs,
                            num_indices,
                            idx_ofs,
                            num_strips,
                            strip_ofs,
                            _sg_flags,
                        ) = struct.unpack_from("<6I B", data, sg_off)
                        vert_base = sg_off + vert_ofs
                        verts = [
                            struct.unpack_from("<H", data, vert_base + i * vert_size + 4)[0]
                            for i in range(num_verts)
                        ]
                        idx_base = sg_off + idx_ofs
                        idxs = struct.unpack_from(f"<{num_indices}H", data, idx_base)
                        faces.extend(verts[i] for i in idxs)

        arr = np.array(faces, dtype=int)
        return arr.reshape(-1, 3)

    # ------------------------------------------------------------------
    def _goldsrc_bounds(self, data: bytes) -> Tuple[Vector, Vector] | None:
        """Return bounding box for a Goldsource model."""

        # Offsets taken from ``studiohdr_t`` in the Goldsource SDK.
        if len(data) < 136:
            return None
        bbmin = struct.unpack_from("<3f", data, 112)
        bbmax = struct.unpack_from("<3f", data, 124)
        return bbmin, bbmax

    # ------------------------------------------------------------------
    def _source_bounds(self, data: bytes) -> Tuple[Vector, Vector] | None:
        """Return bounding box for a Source engine model."""

        # Offsets taken from ``studiohdr_t`` in the Source SDK.
        if len(data) < 152:
            return None

        view_bbmin = struct.unpack_from("<3f", data, 128)
        view_bbmax = struct.unpack_from("<3f", data, 140)
        if any(view_bbmin) or any(view_bbmax):
            return view_bbmin, view_bbmax

        hull_min = struct.unpack_from("<3f", data, 104)
        hull_max = struct.unpack_from("<3f", data, 116)
        return hull_min, hull_max

