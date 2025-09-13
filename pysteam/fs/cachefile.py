from __future__ import annotations

import struct
import binascii
import zlib
import hashlib
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import BinaryIO, Optional, List, Callable
from types import SimpleNamespace

try:  # Optional dependency for AES-based decryption
    from Crypto.Cipher import AES  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    AES = None

# ---------------------------------------------------------------------------
# Flag constants pulled directly from HLLib's GCFFile.cpp
# ---------------------------------------------------------------------------
HL_GCF_FLAG_FILE = 0x00004000
HL_GCF_FLAG_ENCRYPTED = 0x00000100
HL_GCF_FLAG_BACKUP_LOCAL = 0x00000040
HL_GCF_FLAG_COPY_LOCAL = 0x0000000A
HL_GCF_FLAG_COPY_LOCAL_NO_OVERWRITE = 0x00000001
HL_GCF_CHECKSUM_LENGTH = 0x00008000

# Package and item attribute names mirroring CGCFFile's static arrays.
PACKAGE_ATTRIBUTE_NAMES = [
    "Version",
    "Cache ID",
    "Allocated Blocks",
    "Used Blocks",
    "Block Length",
    "Last Version Played",
]

ITEM_ATTRIBUTE_NAMES = [
    "Encrypted",
    "Copy Locally",
    "Overwrite Local Copy",
    "Backup Local Copy",
    "Flags",
    "Fragmentation",
]


def _normalize_key(key: bytes) -> bytes:
    for size in (16, 24, 32):
        if len(key) <= size:
            return key.ljust(size, b"\x00")
    return key[:32]


def _decrypt_aes(data: bytes, key: bytes) -> bytes:
    if AES is None:
        raise ImportError("pycryptodome is required for encrypted GCF support")
    key = _normalize_key(key)
    pad = (-len(data)) % 16
    cipher = AES.new(key, AES.MODE_CBC, b"\x00" * 16)
    dec = cipher.decrypt(data + b"\x00" * pad)
    if pad:
        dec = dec[:-pad]
    return dec


def decrypt_gcf_data(data: bytes, key: bytes) -> bytes:
    """Decrypt and decompress ``data`` from an encrypted GCF file."""
    out = bytearray()
    pos = 0
    while pos < len(data):
        chunk = data[pos : pos + HL_GCF_CHECKSUM_LENGTH]
        if len(chunk) < 8:
            break
        comp_size, uncomp_size = struct.unpack_from("<ii", chunk, 0)
        if (
            uncomp_size > HL_GCF_CHECKSUM_LENGTH
            or comp_size > uncomp_size
            or uncomp_size < -1
            or comp_size < -1
        ):
            dec = _decrypt_aes(chunk, key)
            out.extend(dec)
            pos += HL_GCF_CHECKSUM_LENGTH
        else:
            enc = chunk[: 8 + comp_size]
            dec = _decrypt_aes(enc, key)
            try:
                out.extend(zlib.decompress(dec[8:8 + comp_size]))
            except zlib.error:
                out.extend(dec[8:8 + comp_size])
            pos += 8 + comp_size
    return bytes(out)


def _derive_key(cache_id: int) -> bytes:
    """Derive the AES key for ``cache_id`` using Steam's GCF key algorithm."""
    return hashlib.md5(struct.pack("<I", cache_id)).digest()

###############################################################################
# 1.  Structure definitions (direct C++ -> Python translation)
###############################################################################


@dataclass
class GCFHeader:
    dummy0: int
    major_version: int
    minor_version: int
    cache_id: int
    last_version_played: int
    dummy1: int
    dummy2: int
    file_size: int
    block_size: int
    block_count: int
    dummy3: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFHeader":
        data = stream.read(44)
        values = struct.unpack("<11I", data)
        return cls(*values)


@dataclass
class GCFBlockEntryHeader:
    block_count: int
    blocks_used: int
    dummy0: int
    dummy1: int
    dummy2: int
    dummy3: int
    dummy4: int
    checksum: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFBlockEntryHeader":
        data = stream.read(32)
        values = struct.unpack("<8I", data)
        return cls(*values)


@dataclass
class GCFBlockEntry:
    entry_flags: int
    file_data_offset: int
    file_data_size: int
    first_data_block_index: int
    next_block_entry_index: int
    previous_block_entry_index: int
    directory_index: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFBlockEntry":
        data = stream.read(28)
        values = struct.unpack("<7I", data)
        return cls(*values)


@dataclass
class GCFFragmentationMapHeader:
    block_count: int
    first_unused_entry: int
    terminator: int
    checksum: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFFragmentationMapHeader":
        data = stream.read(16)
        values = struct.unpack("<4I", data)
        return cls(*values)


@dataclass
class GCFFragmentationMap:
    next_data_block_index: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFFragmentationMap":
        data = stream.read(4)
        (next_index,) = struct.unpack("<I", data)
        return cls(next_index)


@dataclass
class GCFBlockEntryMapHeader:
    block_count: int
    first_block_entry_index: int
    last_block_entry_index: int
    dummy0: int
    checksum: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFBlockEntryMapHeader":
        data = stream.read(20)
        values = struct.unpack("<5I", data)
        return cls(*values)


@dataclass
class GCFBlockEntryMap:
    previous_block_entry_index: int
    next_block_entry_index: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFBlockEntryMap":
        data = stream.read(8)
        values = struct.unpack("<2I", data)
        return cls(*values)


@dataclass
class GCFDirectoryHeader:
    dummy0: int
    cache_id: int
    last_version_played: int
    item_count: int
    file_count: int
    dummy1: int
    directory_size: int
    name_size: int
    info1_count: int
    copy_count: int
    local_count: int
    dummy2: int
    dummy3: int
    checksum: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFDirectoryHeader":
        data = stream.read(56)
        values = struct.unpack("<14I", data)
        return cls(*values)


@dataclass
class GCFDirectoryEntry:
    name_offset: int
    item_size: int
    checksum_index: int
    directory_flags: int
    parent_index: int
    next_index: int
    first_index: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFDirectoryEntry":
        data = stream.read(28)
        values = struct.unpack("<7I", data)
        return cls(*values)


@dataclass
class GCFDirectoryInfo1Entry:
    dummy0: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFDirectoryInfo1Entry":
        data = stream.read(4)
        (dummy0,) = struct.unpack("<I", data)
        return cls(dummy0)


@dataclass
class GCFDirectoryInfo2Entry:
    dummy0: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFDirectoryInfo2Entry":
        data = stream.read(4)
        (dummy0,) = struct.unpack("<I", data)
        return cls(dummy0)


@dataclass
class GCFDirectoryCopyEntry:
    directory_index: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFDirectoryCopyEntry":
        data = stream.read(4)
        (directory_index,) = struct.unpack("<I", data)
        return cls(directory_index)


@dataclass
class GCFDirectoryLocalEntry:
    directory_index: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFDirectoryLocalEntry":
        data = stream.read(4)
        (directory_index,) = struct.unpack("<I", data)
        return cls(directory_index)


@dataclass
class GCFDirectoryMapHeader:
    dummy0: int
    dummy1: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFDirectoryMapHeader":
        data = stream.read(8)
        values = struct.unpack("<2I", data)
        return cls(*values)


@dataclass
class GCFDirectoryMapEntry:
    first_block_index: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFDirectoryMapEntry":
        data = stream.read(4)
        (first_block_index,) = struct.unpack("<I", data)
        return cls(first_block_index)


@dataclass
class GCFChecksumHeader:
    dummy0: int
    checksum_size: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFChecksumHeader":
        data = stream.read(8)
        values = struct.unpack("<2I", data)
        return cls(*values)


@dataclass
class GCFChecksumMapHeader:
    dummy0: int
    dummy1: int
    item_count: int
    checksum_count: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFChecksumMapHeader":
        data = stream.read(16)
        values = struct.unpack("<4I", data)
        return cls(*values)


@dataclass
class GCFChecksumMapEntry:
    checksum_count: int
    first_checksum_index: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFChecksumMapEntry":
        data = stream.read(8)
        values = struct.unpack("<2I", data)
        return cls(*values)


@dataclass
class GCFChecksumEntry:
    checksum: int

    @classmethod
    def read(cls, stream: BinaryIO) -> "GCFChecksumEntry":
        data = stream.read(4)
        (checksum,) = struct.unpack("<I", data)
        return cls(checksum)


@dataclass
class GCFDataBlockHeader:
    last_version_played: int
    block_count: int
    block_size: int
    first_block_offset: int
    blocks_used: int
    checksum: int

    @classmethod
    def read(cls, stream: BinaryIO, version: int) -> "GCFDataBlockHeader":
        if version < 5:
            data = stream.read(20)
            block_count, block_size, first_block_offset, blocks_used, checksum = struct.unpack("<5I", data)
            return cls(0, block_count, block_size, first_block_offset, blocks_used, checksum)
        data = stream.read(24)
        values = struct.unpack("<6I", data)
        return cls(*values)


###############################################################################
# 1b. Directory tree structures
###############################################################################


@dataclass
class DirectoryItem:
    name: str
    index: int
    flags: int
    parent: Optional["DirectoryFolder"] = None
    package: Optional["GCFFile"] = None

    def path(self) -> str:
        parts: List[str] = []
        item: Optional["DirectoryItem"] = self
        while item and item.parent is not None:
            parts.append(item.name)
            item = item.parent
        return "\\".join(reversed(parts))

    def is_file(self) -> bool:
        return bool(self.flags & HL_GCF_FLAG_FILE)

    def is_folder(self) -> bool:
        return not self.is_file()

    @property
    def folder(self) -> Optional["DirectoryFolder"]:
        return self.parent


@dataclass
class DirectoryFile(DirectoryItem):
    file_size: int = 0

    def size(self) -> int:
        return self.file_size

    def open(self, mode: str = "rb", key: bytes | None = None) -> BinaryIO:
        if "r" not in mode:
            raise ValueError("only reading supported")
        if not self.package:
            raise ValueError("file is not associated with a package")
        return self.package.open_stream(self.index)

    @property
    def num_of_blocks(self) -> int:
        if not self.package:
            return 0
        return self.package.get_item_fragmentation(self.index)[1]

    @property
    def is_fragmented(self) -> bool:
        if not self.package:
            return False
        return self.package.get_item_fragmentation(self.index)[0] > 0


@dataclass
class DirectoryFolder(DirectoryItem):
    children: List[DirectoryItem] = field(default_factory=list)

    def add_folder(self, name: str, index: int, flags: int) -> "DirectoryFolder":
        folder = DirectoryFolder(name, index, flags, parent=self, package=self.package)
        self.children.append(folder)
        return folder

    def add_file(self, name: str, index: int, size: int, flags: int) -> DirectoryFile:
        file = DirectoryFile(
            name, index, flags, parent=self, package=self.package, file_size=size
        )
        self.children.append(file)
        return file

    def all_files(self) -> List[DirectoryFile]:
        files: List[DirectoryFile] = []
        for child in self.children:
            if isinstance(child, DirectoryFile):
                files.append(child)
            elif isinstance(child, DirectoryFolder):
                files.extend(child.all_files())
        return files

###############################################################################
# 2.  Skeleton class definition
###############################################################################


class GCFFile:
    """Python re-implementation of HLLib's CGCFFile."""

    def __init__(self, source: BinaryIO | str, read_encrypted: bool = False):
        if isinstance(source, (str, bytes, bytearray)):
            # Open in read/write mode so that defragmentation can update the file
            # in-place.  Callers that only need read access may pass an existing
            # stream opened in the desired mode.
            self.stream: BinaryIO = open(source, "r+b")
            self._owns_stream = True
        else:
            self.stream = source
            self._owns_stream = False

        # Placeholders for all major data blocks.
        self.header: Optional[GCFHeader] = None
        self.block_entry_header: Optional[GCFBlockEntryHeader] = None
        self.block_entries: List[GCFBlockEntry] = []
        self.fragmentation_map_header: Optional[GCFFragmentationMapHeader] = None
        self.fragmentation_map: List[GCFFragmentationMap] = []
        self.block_entry_map_header: Optional[GCFBlockEntryMapHeader] = None
        self.block_entry_map: List[GCFBlockEntryMap] = []
        self.directory_header: Optional[GCFDirectoryHeader] = None
        self.directory_entries: List[GCFDirectoryEntry] = []
        self.directory_names: Optional[bytes] = None
        self.directory_info1_entries: List[GCFDirectoryInfo1Entry] = []
        self.directory_info2_entries: List[GCFDirectoryInfo2Entry] = []
        self.directory_copy_entries: List[GCFDirectoryCopyEntry] = []
        self.directory_local_entries: List[GCFDirectoryLocalEntry] = []
        self.directory_map_header: Optional[GCFDirectoryMapHeader] = None
        self.directory_map_entries: List[GCFDirectoryMapEntry] = []
        self.checksum_header: Optional[GCFChecksumHeader] = None
        self.checksum_map_header: Optional[GCFChecksumMapHeader] = None
        self.checksum_map_entries: List[GCFChecksumMapEntry] = []
        self.checksum_entries: List[GCFChecksumEntry] = []
        self.data_block_header: Optional[GCFDataBlockHeader] = None
        self._version: Optional[int] = None

        # Directory tree storage populated in step 3.
        self.directory_items: List[Optional[DirectoryItem]] = []
        self.root: Optional[DirectoryFolder] = None

        # Whether encrypted files should be read and decrypted.
        self.read_encrypted = read_encrypted

        self.map_data_structures()
        self.build_directory_tree()

    @classmethod
    def parse(cls, path: str, read_encrypted: bool = False) -> "GCFFile":
        return cls(path, read_encrypted=read_encrypted)

    def _get_encryption_key(self) -> bytes:
        if not hasattr(self, "_encryption_key"):
            if not self.header:
                raise ValueError("Cache file not parsed")
            self._encryption_key = _derive_key(self.header.cache_id)
        return self._encryption_key

    # ------------------------------------------------------------------
    # Mapping / unmapping
    # ------------------------------------------------------------------
    def map_data_structures(self) -> None:
        """Parse header structures from the stream."""
        self.stream.seek(0)

        self.header = GCFHeader.read(self.stream)
        if (
            self.header.major_version != 1
            or self.header.minor_version not in (1, 3, 5, 6)
        ):
            raise ValueError(
                f"Unsupported GCF version {self.header.major_version}.{self.header.minor_version}"
            )
        self._version = self.header.minor_version

        self.block_entry_header = GCFBlockEntryHeader.read(self.stream)
        self.block_entries = [
            GCFBlockEntry.read(self.stream)
            for _ in range(self.block_entry_header.block_count)
        ]

        self.fragmentation_map_header = GCFFragmentationMapHeader.read(self.stream)
        self.fragmentation_map = [
            GCFFragmentationMap.read(self.stream)
            for _ in range(self.fragmentation_map_header.block_count)
        ]

        if self._version < 6:
            self.block_entry_map_header = GCFBlockEntryMapHeader.read(self.stream)
            self.block_entry_map = [
                GCFBlockEntryMap.read(self.stream)
                for _ in range(self.block_entry_map_header.block_count)
            ]
        else:
            self.block_entry_map_header = None
            self.block_entry_map = []

        self.directory_header = GCFDirectoryHeader.read(self.stream)
        # Remember the file offset of the directory entries so that individual
        # entries can be updated in-place (e.g. when clearing the encrypted flag
        # after decryption).
        self._directory_entries_offset = self.stream.tell()
        self.directory_entries = [
            GCFDirectoryEntry.read(self.stream)
            for _ in range(self.directory_header.item_count)
        ]
        self.directory_names = self.stream.read(self.directory_header.name_size)
        self.directory_info1_entries = [
            GCFDirectoryInfo1Entry.read(self.stream)
            for _ in range(self.directory_header.info1_count)
        ]
        self.directory_info2_entries = [
            GCFDirectoryInfo2Entry.read(self.stream)
            for _ in range(self.directory_header.item_count)
        ]
        self.directory_copy_entries = [
            GCFDirectoryCopyEntry.read(self.stream)
            for _ in range(self.directory_header.copy_count)
        ]
        self.directory_local_entries = [
            GCFDirectoryLocalEntry.read(self.stream)
            for _ in range(self.directory_header.local_count)
        ]

        if self._version >= 5:
            self.directory_map_header = GCFDirectoryMapHeader.read(self.stream)
        else:
            self.directory_map_header = None

        self.directory_map_entries = [
            GCFDirectoryMapEntry.read(self.stream)
            for _ in range(self.directory_header.item_count)
        ]

        if self._version > 1:
            self.checksum_header = GCFChecksumHeader.read(self.stream)
            self.checksum_map_header = GCFChecksumMapHeader.read(self.stream)
            self.checksum_map_entries = [
                GCFChecksumMapEntry.read(self.stream)
                for _ in range(self.checksum_map_header.item_count)
            ]
            self.checksum_entries = [
                GCFChecksumEntry.read(self.stream)
                for _ in range(self.checksum_map_header.checksum_count)
            ]
        else:
            self.checksum_header = None
            self.checksum_map_header = None
            self.checksum_map_entries = []
            self.checksum_entries = []

        self.data_block_header = GCFDataBlockHeader.read(self.stream, self._version)

    def unmap_data_structures(self) -> None:
        self.header = None
        self.block_entry_header = None
        self.block_entries = []
        self.fragmentation_map_header = None
        self.fragmentation_map = []
        self.block_entry_map_header = None
        self.block_entry_map = []
        self.directory_header = None
        self.directory_entries = []
        self.directory_names = None
        self.directory_info1_entries = []
        self.directory_info2_entries = []
        self.directory_copy_entries = []
        self.directory_local_entries = []
        self.directory_map_header = None
        self.directory_map_entries = []
        self.checksum_header = None
        self.checksum_map_header = None
        self.checksum_map_entries = []
        self.checksum_entries = []
        self.data_block_header = None
        self._version = None
        self.directory_items = []
        self.root = None

    def close(self) -> None:
        self.unmap_data_structures()
        if self._owns_stream:
            self.stream.close()

    # ------------------------------------------------------------------
    # Step 3: Directory tree construction
    # ------------------------------------------------------------------
    def build_directory_tree(self) -> None:
        if not self.directory_header:
            return
        count = self.directory_header.item_count
        self.directory_items = [None] * count

        self.root = DirectoryFolder("root", 0, 0, package=self)
        self.directory_items[0] = self.root
        self._build_folder(self.root)

    def _read_name(self, offset: int) -> str:
        if self.directory_names is None:
            return ""
        end = self.directory_names.find(b"\x00", offset)
        if end == -1:
            end = len(self.directory_names)
        return self.directory_names[offset:end].decode("utf-8", "replace")

    def _build_folder(self, folder: DirectoryFolder) -> None:
        entry = self.directory_entries[folder.index]
        index = entry.first_index
        while index and index != 0xFFFFFFFF:
            child_entry = self.directory_entries[index]
            name = self._read_name(child_entry.name_offset)
            ns = SimpleNamespace(index=index, directory_flags=child_entry.directory_flags)
            if (child_entry.directory_flags & HL_GCF_FLAG_FILE) == 0:
                child = folder.add_folder(name, index, child_entry.directory_flags)
                child._manifest_entry = ns
                self.directory_items[index] = child
                self._build_folder(child)
            else:
                file = folder.add_file(name, index, child_entry.item_size, child_entry.directory_flags)
                file._manifest_entry = ns
                self.directory_items[index] = file
            index = child_entry.next_index

    # ------------------------------------------------------------------
    # Step 4: Basic item utilities
    # ------------------------------------------------------------------
    def get_file_extractable(self, file_index: int) -> bool:
        entry = self.directory_entries[file_index]
        if entry.directory_flags & HL_GCF_FLAG_ENCRYPTED:
            return False
        size = 0
        block_index = self.directory_map_entries[file_index].first_block_index
        while block_index != self.data_block_header.block_count:
            block_entry = self.block_entries[block_index]
            size += block_entry.file_data_size
            block_index = block_entry.next_block_entry_index
        return size >= entry.item_size

    def get_item_fragmentation(self, index: int) -> tuple[int, int]:
        blocks_fragmented = 0
        blocks_used = 0

        entry = self.directory_entries[index]
        if (entry.directory_flags & HL_GCF_FLAG_FILE) == 0:
            idx = entry.first_index
            while idx and idx != 0xFFFFFFFF:
                f, u = self.get_item_fragmentation(idx)
                blocks_fragmented += f
                blocks_used += u
                idx = self.directory_entries[idx].next_index
            return blocks_fragmented, blocks_used

        data_block_terminator = (
            0x0000FFFF if self.fragmentation_map_header and self.fragmentation_map_header.terminator == 0 else 0xFFFFFFFF
        )
        last_block_index = self.data_block_header.block_count
        block_entry_index = self.directory_map_entries[index].first_block_index
        while block_entry_index != self.data_block_header.block_count:
            block_entry_size = 0
            data_block_index = self.block_entries[block_entry_index].first_data_block_index
            while (
                data_block_index < data_block_terminator
                and block_entry_size < self.block_entries[block_entry_index].file_data_size
            ):
                if last_block_index != self.data_block_header.block_count and last_block_index + 1 != data_block_index:
                    blocks_fragmented += 1
                blocks_used += 1
                last_block_index = data_block_index
                data_block_index = self.fragmentation_map[data_block_index].next_data_block_index
                block_entry_size += self.data_block_header.block_size
            block_entry_index = self.block_entries[block_entry_index].next_block_entry_index
        return blocks_fragmented, blocks_used

    def _get_item_fragmentation(self, index: int) -> tuple[int, int]:
        return self.get_item_fragmentation(index)

    def is_fragmented(self) -> bool:
        return self.get_item_fragmentation(0)[0] > 0

    def count_complete_files(self) -> tuple[int, int]:
        total = 0
        complete = 0
        for i, entry in enumerate(self.directory_entries):
            if entry.directory_flags & HL_GCF_FLAG_FILE:
                total += 1
                if self.get_file_extractable(i):
                    complete += 1
        return complete, total

    # ------------------------------------------------------------------
    # Defragmentation utilities
    # ------------------------------------------------------------------
    def defragment(
        self,
        force: bool = False,
        progress: Callable[[Optional[DirectoryFile], int, int, int, int], bool] | None = None,
    ) -> bool:
        """Rewrite data blocks sequentially to eliminate fragmentation.

        Parameters
        ----------
        force:
            If ``True`` the file will be rewritten even if no fragmentation is
            detected.  ``True`` also forces lexicographical ordering of file
            data blocks.
        progress:
            Optional callback invoked after each data block is written.  The
            callback receives ``(file, files_defragmented, files_total,
            bytes_defragmented, bytes_total)`` and should return ``True`` to
            continue or ``False`` to cancel the operation.

        Returns
        -------
        bool
            ``True`` if the operation completed, ``False`` if cancelled.
        """

        if not self.stream.writable():
            raise IOError(
                "underlying stream is not writable; defragmentation requires write access"
            )

        if not self.fragmentation_map_header or not self.data_block_header:
            return False

        blocks_fragmented = 0
        blocks_used = 0
        files_total = 0
        file_block_counts: dict[int, int] = {}
        file_indices: list[int] = []
        for i, entry in enumerate(self.directory_entries):
            if entry.directory_flags & HL_GCF_FLAG_FILE:
                f, u = self.get_item_fragmentation(i)
                blocks_fragmented += f
                blocks_used += u
                files_total += 1
                file_block_counts[i] = u
                file_indices.append(i)

        block_size = self.data_block_header.block_size
        bytes_total = blocks_used * block_size

        if (blocks_fragmented == 0 and not force) or blocks_used == 0:
            if progress:
                progress(None, files_total, files_total, bytes_total, bytes_total)
            return True

        def _item_path(index: int) -> str:
            item = self.directory_items[index]
            parts = []
            while item and item.parent is not None:
                parts.append(item.name)
                item = item.parent
            return "/".join(reversed(parts))

        if force:
            file_indices.sort(key=_item_path)

        terminator = (
            0x0000FFFF
            if self.fragmentation_map_header.terminator == 0
            else 0xFFFFFFFF
        )

        mapping: dict[int, int] = {}
        block_owner: dict[int, int] = {}
        next_index = 0
        for idx in file_indices:
            block_entry_index = self.directory_map_entries[idx].first_block_index
            while block_entry_index != self.data_block_header.block_count:
                block_entry = self.block_entries[block_entry_index]
                remaining = block_entry.file_data_size
                data_block_index = block_entry.first_data_block_index
                while remaining > 0 and data_block_index < terminator:
                    mapping[data_block_index] = next_index
                    block_owner[next_index] = idx
                    next_index += 1
                    remaining -= block_size
                    data_block_index = self.fragmentation_map[
                        data_block_index
                    ].next_data_block_index
                block_entry_index = block_entry.next_block_entry_index

        first_block_offset = self.data_block_header.first_block_offset

        visited: set[int] = set()
        processed_targets: set[int] = set()
        remaining_blocks = dict(file_block_counts)
        remaining_files = set(file_indices)
        files_defragmented = 0
        bytes_defragmented = 0
        cancelled = False

        def report(file_index: int | None) -> None:
            nonlocal files_defragmented, bytes_defragmented, cancelled
            if file_index is not None:
                remaining_blocks[file_index] -= 1
                if remaining_blocks[file_index] == 0:
                    remaining_files.discard(file_index)
                files_defragmented = files_total - len(remaining_files)
            bytes_defragmented += block_size
            if progress and not progress(
                self.directory_items[file_index] if file_index is not None else None,
                files_defragmented,
                files_total,
                bytes_defragmented,
                bytes_total,
            ):
                cancelled = True

        for start in list(mapping.keys()):
            if cancelled:
                break
            if start in visited or mapping[start] == start:
                continue
            current = start
            self.stream.seek(first_block_offset + current * block_size)
            temp = self.stream.read(block_size)
            while True:
                target = mapping[current]
                visited.add(current)
                offset = first_block_offset + target * block_size
                self.stream.seek(offset)
                if target in visited or mapping.get(target, target) == target:
                    self.stream.write(temp)
                    processed_targets.add(target)
                    report(block_owner[target])
                    break
                next_temp = self.stream.read(block_size)
                self.stream.seek(offset)
                self.stream.write(temp)
                processed_targets.add(target)
                report(block_owner[target])
                temp = next_temp
                current = target

        if cancelled:
            for old in mapping.keys():
                if old not in visited:
                    mapping[old] = old

        if not cancelled:
            for old, new in mapping.items():
                if old == new:
                    report(block_owner[new])
                    if cancelled:
                        break

        block_count = self.fragmentation_map_header.block_count
        new_fragmentation_map = [
            GCFFragmentationMap(block_count) for _ in range(block_count)
        ]

        for idx in file_indices:
            block_entry_index = self.directory_map_entries[idx].first_block_index
            while block_entry_index != self.data_block_header.block_count:
                block_entry = self.block_entries[block_entry_index]
                remaining = block_entry.file_data_size
                first_old = block_entry.first_data_block_index
                block_entry.first_data_block_index = mapping[first_old]

                data_block_index = first_old
                while remaining > 0 and data_block_index < terminator:
                    new_index = mapping[data_block_index]
                    next_old = self.fragmentation_map[
                        data_block_index
                    ].next_data_block_index
                    if remaining <= block_size or next_old >= terminator:
                        new_next = terminator
                    else:
                        new_next = mapping[next_old]
                    new_fragmentation_map[new_index].next_data_block_index = new_next
                    data_block_index = next_old
                    remaining -= block_size

                block_entry_index = block_entry.next_block_entry_index

        next_index = max(mapping.values(), default=0) + 1
        for i in range(next_index, block_count):
            new_fragmentation_map[i].next_data_block_index = block_count

        self.fragmentation_map = new_fragmentation_map
        self.fragmentation_map_header.first_unused_entry = next_index
        self.fragmentation_map_header.checksum = (
            self.fragmentation_map_header.block_count
            + self.fragmentation_map_header.first_unused_entry
            + self.fragmentation_map_header.terminator
        ) & 0xFFFFFFFF

        block_entries_offset = 44 + 32
        frag_header_offset = (
            block_entries_offset
            + 28 * self.block_entry_header.block_count
        )
        frag_map_offset = frag_header_offset + 16

        for i, entry in enumerate(self.block_entries):
            self.stream.seek(block_entries_offset + i * 28)
            self.stream.write(
                struct.pack(
                    "<7I",
                    entry.entry_flags,
                    entry.file_data_offset,
                    entry.file_data_size,
                    entry.first_data_block_index,
                    entry.next_block_entry_index,
                    entry.previous_block_entry_index,
                    entry.directory_index,
                )
            )

        self.stream.seek(frag_header_offset)
        self.stream.write(
            struct.pack(
                "<4I",
                self.fragmentation_map_header.block_count,
                self.fragmentation_map_header.first_unused_entry,
                self.fragmentation_map_header.terminator,
                self.fragmentation_map_header.checksum,
            )
        )

        for i, fm in enumerate(self.fragmentation_map):
            self.stream.seek(frag_map_offset + i * 4)
            self.stream.write(struct.pack("<I", fm.next_data_block_index))

        self.stream.flush()

        return not cancelled

    # ------------------------------------------------------------------
    # Step 5: File data access and validation
    # ------------------------------------------------------------------
    def get_file_size(self, file_index: int) -> int:
        return self.directory_entries[file_index].item_size

    def get_file_size_on_disk(self, file_index: int) -> int:
        size = 0
        block_index = self.directory_map_entries[file_index].first_block_index
        while block_index != self.data_block_header.block_count:
            block = self.block_entries[block_index]
            size += ((block.file_data_size + self.data_block_header.block_size - 1) // self.data_block_header.block_size) * self.data_block_header.block_size
            block_index = block.next_block_entry_index
        return size

    def read_file(self, file_index: int) -> bytes:
        """Return the contents of the specified file."""
        stream = self.open_stream(file_index)
        try:
            return stream.read()
        finally:
            stream.close()

    def open_stream(self, file_index: int) -> "BinaryIO":
        """Return a stream for the given file index.

        Encrypted files are fully decrypted and decompressed into memory before
        being exposed as a ``BytesIO`` stream.  If ``read_encrypted`` is ``False``
        or the required cryptography backend is unavailable, a ``ValueError`` is
        raised.
        """
        entry = self.directory_entries[file_index]
        if entry.directory_flags & HL_GCF_FLAG_ENCRYPTED:
            if not self.read_encrypted:
                raise ValueError(
                    "File is encrypted; initialize with read_encrypted=True to read it"
                )
            raw = GCFStream(self, file_index).read()
            try:
                key = self._get_encryption_key()
                data = decrypt_gcf_data(raw, key)
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ValueError("pycryptodome is required for encrypted GCF support") from exc
            return BytesIO(data)

        return GCFStream(self, file_index)

    # ------------------------------------------------------------------
    # Decryption
    # ------------------------------------------------------------------
    def decrypt_file(self, file_index: int, key: bytes | None = None) -> int:
        """Decrypt ``file_index`` in-place and return the number of bytes written.

        The routine mirrors ``CGCFFile::DecryptFile`` from HLLib.  File data is
        streamed from the cache, decrypted (and decompressed if necessary) and
        then written back to the underlying archive.  On success the encrypted
        flag is cleared from the directory entry.
        """

        entry = self.directory_entries[file_index]
        if not (entry.directory_flags & HL_GCF_FLAG_ENCRYPTED):
            return entry.item_size

        raw_stream = GCFStream(self, file_index)
        raw = raw_stream.read()

        if key is None:
            key = self._get_encryption_key()

        data = decrypt_gcf_data(raw, key)

        if len(data) != entry.item_size:
            raise ValueError("Decrypted data size mismatch")

        pos = 0
        for _, block_index, length in raw_stream._segments:
            file_offset = (
                self.data_block_header.first_block_offset
                + block_index * self.data_block_header.block_size
            )
            self.stream.seek(file_offset)
            self.stream.write(data[pos : pos + length])
            pos += length

        # Clear the encrypted flag and update the directory entry on disk.
        entry.directory_flags &= ~HL_GCF_FLAG_ENCRYPTED
        if hasattr(self, "_directory_entries_offset"):
            offset = self._directory_entries_offset + file_index * 28 + 12
            self.stream.seek(offset)
            self.stream.write(struct.pack("<I", entry.directory_flags))

        self.stream.flush()
        return len(data)

    def decrypt_item(self, index: int = 0, key: bytes | None = None) -> int:
        """Recursively decrypt ``index`` and its children."""

        entry = self.directory_entries[index]
        if entry.directory_flags & HL_GCF_FLAG_FILE:
            return self.decrypt_file(index, key)

        total = 0
        child = entry.first_index
        while child and child != 0xFFFFFFFF:
            total += self.decrypt_item(child, key)
            child = self.directory_entries[child].next_index
        return total

    def validate_file(
        self,
        file_index: int,
        progress: Callable[[int, int], bool] | None = None,
    ) -> str:
        """Validate a single file within the archive.

        File data is streamed from the underlying cache using :class:`GCFStream`
        to avoid loading the entire contents into memory.  Checksums are
        computed per chunk and compared against the stored checksum table.  The
        optional ``progress`` callback is invoked after each chunk is processed
        with ``(bytes_processed, total_bytes)`` and should return ``True`` to
        continue or ``False`` to cancel the operation.
        """

        entry = self.directory_entries[file_index]

        # Ensure we have all data blocks required for the file.
        size = 0
        block_index = self.directory_map_entries[file_index].first_block_index
        while block_index != self.data_block_header.block_count:
            size += self.block_entries[block_index].file_data_size
            block_index = self.block_entries[block_index].next_block_entry_index
        if size != entry.item_size:
            return "incomplete"

        if entry.checksum_index == 0xFFFFFFFF or not self.checksum_map_entries:
            return "assumed-ok"

        stream = self.open_stream(file_index)
        try:
            map_entry = self.checksum_map_entries[entry.checksum_index]
            remaining = entry.item_size
            total = entry.item_size
            processed = 0
            i = 0
            while remaining > 0 and i < map_entry.checksum_count:
                to_read = min(HL_GCF_CHECKSUM_LENGTH, remaining)
                chunk = stream.read(to_read)
                if len(chunk) != to_read:
                    return "incomplete"
                checksum = (zlib.adler32(chunk) ^ binascii.crc32(chunk)) & 0xFFFFFFFF
                stored = self.checksum_entries[map_entry.first_checksum_index + i].checksum
                if checksum != stored:
                    return "corrupt"
                remaining -= to_read
                processed += to_read
                i += 1
                if progress and not progress(processed, total):
                    return "cancelled"

            if remaining > 0 or i != map_entry.checksum_count:
                return "incomplete"
        finally:
            stream.close()

        return "ok"

    def validate(self, progress: Callable[[int, int], bool] | None = None) -> List[str]:
        errors: List[str] = []
        indices = [
            i
            for i, e in enumerate(self.directory_entries)
            if e.directory_flags & HL_GCF_FLAG_FILE
        ]
        total = len(indices)
        processed = 0
        for i in indices:
            status = self.validate_file(i)
            if status not in ("ok", "assumed-ok"):
                path = self.directory_items[i].path() if self.directory_items else str(i)
                errors.append(f"{path}: {status}")
            processed += 1
            if progress and not progress(processed, total):
                break
        return errors

    # ------------------------------------------------------------------
    # Package and item attribute helpers
    # ------------------------------------------------------------------
    def get_package_attributes(self) -> dict[str, int]:
        if not self.header or not self.data_block_header:
            return {}
        return {
            PACKAGE_ATTRIBUTE_NAMES[0]: self.header.minor_version,
            PACKAGE_ATTRIBUTE_NAMES[1]: self.header.cache_id,
            PACKAGE_ATTRIBUTE_NAMES[2]: self.data_block_header.block_count,
            PACKAGE_ATTRIBUTE_NAMES[3]: self.data_block_header.blocks_used,
            PACKAGE_ATTRIBUTE_NAMES[4]: self.data_block_header.block_size,
            PACKAGE_ATTRIBUTE_NAMES[5]: self.header.last_version_played,
        }

    def get_item_attributes(self, index: int) -> dict[str, object]:
        entry = self.directory_entries[index]
        attrs = {
            ITEM_ATTRIBUTE_NAMES[4]: entry.directory_flags,
        }
        if entry.directory_flags & HL_GCF_FLAG_FILE:
            attrs.update(
                {
                    ITEM_ATTRIBUTE_NAMES[0]: bool(entry.directory_flags & HL_GCF_FLAG_ENCRYPTED),
                    ITEM_ATTRIBUTE_NAMES[1]: bool(entry.directory_flags & HL_GCF_FLAG_COPY_LOCAL),
                    ITEM_ATTRIBUTE_NAMES[2]: not bool(entry.directory_flags & HL_GCF_FLAG_COPY_LOCAL_NO_OVERWRITE),
                    ITEM_ATTRIBUTE_NAMES[3]: bool(entry.directory_flags & HL_GCF_FLAG_BACKUP_LOCAL),
                }
            )
        else:
            attrs.update(
                {
                    ITEM_ATTRIBUTE_NAMES[0]: False,
                    ITEM_ATTRIBUTE_NAMES[1]: False,
                    ITEM_ATTRIBUTE_NAMES[2]: True,
                    ITEM_ATTRIBUTE_NAMES[3]: False,
                }
            )
        blocks_fragmented, blocks_used = self.get_item_fragmentation(index)
        attrs[ITEM_ATTRIBUTE_NAMES[5]] = (
            0.0 if blocks_used == 0 else (blocks_fragmented / blocks_used) * 100.0
        )
        return attrs

    def is_gcf(self) -> bool:
        return True

    @property
    def data_header(self) -> Optional[GCFDataBlockHeader]:
        return self.data_block_header

    def convert_version(
        self,
        target_version: int,
        out_path: str,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None:
        raise NotImplementedError("Conversion not supported")


class GCFStream:
    """Stream interface for reading file data from a ``GCFFile``."""

    def __init__(self, gcf: GCFFile, index: int) -> None:
        self.gcf = gcf
        self.index = index
        self.length = gcf.directory_entries[index].item_size
        self.position = 0
        self.block_size = gcf.data_block_header.block_size

        self._segments: List[tuple[int, int, int]] = []
        self._build_segments()
        self._segment_index = 0

    def _build_segments(self) -> None:
        terminator = (
            0xFFFF if self.gcf.fragmentation_map_header.terminator == 0 else 0xFFFFFFFF
        )
        block_entry_index = self.gcf.directory_map_entries[self.index].first_block_index
        block_entry_terminator = self.gcf.data_block_header.block_count
        offset = 0
        while (
            block_entry_index != block_entry_terminator
            and block_entry_index < len(self.gcf.block_entries)
        ):
            block_entry = self.gcf.block_entries[block_entry_index]
            data_block_index = block_entry.first_data_block_index
            data_block_offset = 0
            while (
                data_block_offset < block_entry.file_data_size
                and data_block_index < terminator
            ):
                length = min(
                    self.block_size, block_entry.file_data_size - data_block_offset
                )
                self._segments.append((offset, data_block_index, length))
                offset += length
                data_block_offset += length
                if data_block_offset < block_entry.file_data_size:
                    data_block_index = self.gcf.fragmentation_map[
                        data_block_index
                    ].next_data_block_index
            block_entry_index = block_entry.next_block_entry_index

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            new_pos = offset
        elif whence == os.SEEK_CUR:
            new_pos = self.position + offset
        elif whence == os.SEEK_END:
            new_pos = self.length + offset
        else:
            raise ValueError("Invalid whence")

        if new_pos < 0 or new_pos > self.length:
            raise ValueError("Attempting to seek outside file bounds")

        self.position = new_pos
        self._segment_index = 0
        while (
            self._segment_index < len(self._segments)
            and self.position
            >= self._segments[self._segment_index][0] + self._segments[self._segment_index][2]
        ):
            self._segment_index += 1
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0 or self.position + size > self.length:
            size = self.length - self.position
        if size <= 0:
            return b""

        remaining = size
        pieces: List[bytes] = []
        while (
            remaining > 0
            and self.position < self.length
            and self._segment_index < len(self._segments)
        ):
            seg_offset, block_index, seg_len = self._segments[self._segment_index]
            offset_in_seg = self.position - seg_offset
            take = min(seg_len - offset_in_seg, remaining)
            file_offset = (
                self.gcf.data_block_header.first_block_offset
                + block_index * self.block_size
                + offset_in_seg
            )
            self.gcf.stream.seek(file_offset)
            pieces.append(self.gcf.stream.read(take))

            self.position += take
            remaining -= take
            if offset_in_seg + take >= seg_len:
                self._segment_index += 1

        return b"".join(pieces)


class CacheFileManifestEntry:
    FLAG_IS_FILE = HL_GCF_FLAG_FILE
    FLAG_IS_EXECUTABLE = 0x00000800
    FLAG_IS_HIDDEN = 0x00000400
    FLAG_IS_READ_ONLY = 0x00000200
    FLAG_IS_ENCRYPTED = HL_GCF_FLAG_ENCRYPTED
    FLAG_IS_PURGE_FILE = 0x00000080
    FLAG_BACKUP_PLZ = HL_GCF_FLAG_BACKUP_LOCAL
    FLAG_IS_NO_CACHE = 0x00000020
    FLAG_IS_LOCKED = 0x00000008
    FLAG_IS_LAUNCH = 0x00000002
    FLAG_IS_USER_CONFIG = 0x00000001


CacheFile = GCFFile

__all__ = [
    "PACKAGE_ATTRIBUTE_NAMES",
    "ITEM_ATTRIBUTE_NAMES",
    "GCFHeader",
    "GCFBlockEntryHeader",
    "GCFBlockEntry",
    "GCFFragmentationMapHeader",
    "GCFFragmentationMap",
    "GCFBlockEntryMapHeader",
    "GCFBlockEntryMap",
    "GCFDirectoryHeader",
    "GCFDirectoryEntry",
    "GCFDirectoryInfo1Entry",
    "GCFDirectoryInfo2Entry",
    "GCFDirectoryCopyEntry",
    "GCFDirectoryLocalEntry",
    "GCFDirectoryMapHeader",
    "GCFDirectoryMapEntry",
    "GCFChecksumHeader",
    "GCFChecksumMapHeader",
    "GCFChecksumMapEntry",
    "GCFChecksumEntry",
    "GCFDataBlockHeader",
    "DirectoryItem",
    "DirectoryFile",
    "DirectoryFolder",
    "GCFFile",
    "GCFStream",
    "CacheFile",
    "CacheFileManifestEntry",
]
