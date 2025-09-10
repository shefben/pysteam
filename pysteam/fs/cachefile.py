
from __future__ import annotations

import struct
import os
import zlib
import copy
from types import SimpleNamespace

from typing import Optional, Callable

from pysteam.fs import DirectoryFolder, DirectoryFile, FilesystemPackage
from math import ceil
from zlib import adler32

# For Python 3 compatibility, use io.BytesIO instead of the removed cStringIO module.
from io import BytesIO

try:  # Optional dependency for AES decryption
    from Crypto.Cipher import AES  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    AES = None

CACHE_CHECKSUM_LENGTH = 0x8000


STEAM_TERMINATOR = "\\"  # Archive path separator
MAX_FILENAME = 0


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
        chunk = data[pos : pos + CACHE_CHECKSUM_LENGTH]
        if len(chunk) < 8:
            break
        comp_size, uncomp_size = struct.unpack_from("<ii", chunk, 0)
        if (
            uncomp_size > CACHE_CHECKSUM_LENGTH
            or comp_size > uncomp_size
            or uncomp_size < -1
            or comp_size < -1
        ):
            dec = _decrypt_aes(chunk, key)
            out.extend(dec)
            pos += CACHE_CHECKSUM_LENGTH
        else:
            enc = chunk[: 8 + comp_size]
            dec = _decrypt_aes(enc, key)
            try:
                out.extend(zlib.decompress(dec[8:8 + comp_size]))
            except zlib.error:
                out.extend(dec[8:8 + comp_size])
            pos += 8 + comp_size
    return bytes(out)


def unpack_dword_list(stream, count):
    """Return ``count`` little-endian DWORDs from ``stream`` as a list.

    The GCF/NCF formats store many arrays of 32-bit unsigned integers.  Older
    versions of this project relied on a helper function named
    ``unpack_dword_list`` to decode these arrays, but the function was missing
    which resulted in a ``NameError`` at runtime when parsing cache files.

    Parameters
    ----------
    stream:
        A binary file-like object positioned at the start of the array.
    count:
        Number of DWORDs to read from the stream.

    Returns
    -------
    list[int]
        The unpacked integers.
    """

    if count <= 0:
        return []

    data = stream.read(4 * count)
    if len(data) != 4 * count:
        raise ValueError(f"Expected {4 * count} bytes, got {len(data)}")
    return list(struct.unpack(f"<{count}L", data))


def pack_dword_list(values):
    """Pack an iterable of integers into little-endian DWORD bytes."""
    values = list(values)
    if not values:
        return b""
    return struct.pack(f"<{len(values)}L", *values)

def raise_parse_error(func):
    def internal(self, *args, **kwargs):
        if not self.is_parsed:
            raise ValueError("Cache file needs to be read first.")

        return func(self, *args, **kwargs)
    return internal

def raise_ncf_error(func):
    def internal(self, *args, **kwargs):
        if self.is_ncf():
            raise ValueError("NCF files do not have contents.")

        return func(self, *args, **kwargs)
    return internal

class CacheFile:

    # Constructor
    def __init__(self):
        self.is_parsed = False
        self.blocks = None
        self.alloc_table = None
        self.block_entry_map = None
        self.manifest = None
        self.checksum_map = None
        self.data_header = None
        self.complete_total = 0
        self.complete_available = 0
        self.ncf_folder_pattern = "common/%(name)s"
        # ``stream`` represents the underlying cache file.  When ``parse`` is
        # given a file path we open and own this handle.  If a file-like object
        # is supplied we duplicate the handle so later extraction/defrag
        # operations still have a live stream even after the caller's handle is
        # closed.
        self.stream = None
        self._stream_owner = False

    # Main methods.

    @classmethod
    def parse(cls, source):
        """Parse ``source`` into a :class:`CacheFile` instance.

        ``source`` may be either a path-like object or an already opened
        binary file handle.  In the latter case the handle may be closed after
        parsing; the cache file keeps its own handle alive to service
        subsequent extraction and conversion operations.
        """

        self = cls()

        if isinstance(source, (str, os.PathLike)):
            stream = open(os.fspath(source), "rb")
            self._stream_owner = True
        else:
            stream = source

        try:
            self.filename = os.path.split(os.path.realpath(stream.name))[1]
        except AttributeError:
            self.filename = None

        # Header
        self.header = CacheFileHeader(self)
        self.header.parse(stream.read(44))
        self.header.validate()

        if self.is_gcf():

            # Block Entries
            self.blocks = CacheFileBlockAllocationTable(self)
            self.blocks.parse(stream)
            self.blocks.validate()

            # Allocation Table
            self.alloc_table = CacheFileAllocationTable(self)
            self.alloc_table.parse(stream)
            self.alloc_table.validate()

            # Older GCF versions include an additional block entry map
            if self.header.format_version < 6:
                self.block_entry_map = CacheFileBlockEntryMap(self)
                self.block_entry_map.parse(stream)

        # Manifest
        self.manifest = CacheFileManifest(self)
        self.manifest.parse(stream)
        self.manifest.validate()

        # Checksum Map
        if self.header.format_version > 1:
            self.checksum_map = CacheFileChecksumMap(self)
            self.checksum_map.parse(stream)
            self.checksum_map.validate()
        else:
            self.checksum_map = None

        if self.is_gcf():
            # Data Header.
            self.data_header = CacheFileSectorHeader(self)
            header_size = 24 if self.header.format_version > 3 else 20
            self.data_header.parse(stream.read(header_size), self.header.format_version)
            self.data_header.validate()

        self.is_parsed = True
        self._read_directory()

        # If the caller supplied the stream we need to reopen the file so we
        # still have a valid handle once the caller closes theirs.
        if not self._stream_owner and hasattr(stream, "name"):
            self.stream = open(stream.name, "rb")
            self._stream_owner = True
        else:
            self.stream = stream

        return self

    def close(self) -> None:
        """Close the underlying file stream if we own it."""
        if self._stream_owner and self.stream is not None:
            try:
                self.stream.close()
            finally:
                self.stream = None

    def convert_version(
        self,
        target_version: int,
        out_path: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Convert this cache file to a different GCF format version.

        The converter rewrites all table headers and recalculates offsets and
        checksums so that the resulting archive adheres to the requested format.
        Only GCF archives are supported.
        """

        if not self.is_parsed:
            raise ValueError("Cache file needs to be read first.")
        if not self.is_gcf():
            raise ValueError("Only GCF archives can be converted")
        if target_version not in (1, 3, 5, 6):
            raise ValueError("Unsupported GCF version: %d" % target_version)

        # Deep copy structures so serialisation does not mutate the source.
        # Exclude the cache file object (which holds an open stream) from the
        # copy operation to avoid pickling errors on file-like objects.
        memo = {id(self): None}
        try:
            memo[id(self.stream)] = None  # type: ignore[attr-defined]
        except Exception:
            pass

        header = copy.deepcopy(self.header, memo)
        blocks = copy.deepcopy(self.blocks, memo)
        alloc_table = copy.deepcopy(self.alloc_table, memo)
        block_entry_map = copy.deepcopy(self.block_entry_map, memo)
        manifest = copy.deepcopy(self.manifest, memo)
        data_header = copy.deepcopy(self.data_header, memo)
        checksum_map = (
            copy.deepcopy(self.checksum_map, memo) if self.checksum_map else None
        )

        # Temporary owner that mirrors the structure expected by the various
        # serialisation routines.
        owner = SimpleNamespace(
            header=header, block_entry_map=block_entry_map, blocks=blocks
        )
        manifest.owner = owner
        if block_entry_map:
            block_entry_map.owner = owner
        data_header.owner = owner
        if checksum_map:
            checksum_map.owner = owner

        header.format_version = target_version
        blocks.owner = owner
        alloc_table.owner = owner

        original_map_entries = list(manifest.manifest_map_entries)

        if target_version < 6:
            if block_entry_map is None:
                block_entry_map = CacheFileBlockEntryMap(owner)
                block_entry_map.entries = list(range(blocks.block_count))
                owner.block_entry_map = block_entry_map
            inverse = {blk: idx for idx, blk in enumerate(block_entry_map.entries)}
            manifest.manifest_map_entries = [inverse.get(i, i) for i in original_map_entries]
        else:
            if block_entry_map is not None:
                mapped: list[int] = []
                for i in original_map_entries:
                    if i == 0xFFFFFFFF or i >= len(block_entry_map.entries):
                        mapped.append(0xFFFFFFFF)
                    else:
                        mapped.append(block_entry_map.entries[i])
                manifest.manifest_map_entries = mapped
            block_entry_map = None
            owner.block_entry_map = None

        # Older directory headers differ significantly from newer manifest
        # layouts.  When targeting version 1 we rewrite the manifest header
        # fields to mirror the legacy structure described in ``GCFDirectoryHeader``
        # from HLLib:
        if target_version == 1:
            manifest.header_version = 4  # uiDummy0 constant
            manifest.application_id = header.application_id
            manifest.application_version = header.application_version
            manifest.compression_block_size = CACHE_CHECKSUM_LENGTH
            manifest.hash_table_keys = []
            manifest.hash_table_indices = [0] * manifest.node_count
            manifest.minimum_footprint_entries = []
            manifest.user_config_entries = []
            manifest.depot_info = 0
            manifest.fingerprint = 0

        # Generate a checksum map when targeting newer formats.
        if target_version > 1:
            if checksum_map is None:
                checksum_map = CacheFileChecksumMap(owner)
                checksum_map.header_version = 1
                checksum_map.checksum_size = 4
                checksum_map.format_code = 1
                checksum_map.version = 1
                checksum_map.entries = []
                checksum_map.checksums = []
                checksum_map.signature = b"\0" * 128
                for entry in self.manifest.manifest_entries:
                    if not (
                        entry.directory_flags & CacheFileManifestEntry.FLAG_IS_FILE
                    ):
                        continue
                    crc = 0
                    remaining = entry.item_size
                    block = entry.first_block
                    while block is not None and remaining > 0:
                        for sector in block.sectors:
                            if remaining <= 0:
                                break
                            self.stream.seek(
                                self.data_header.first_sector_offset
                                + sector.index * self.header.sector_size
                            )
                            chunk = self.stream.read(
                                min(remaining, self.header.sector_size)
                            )
                            crc = zlib.crc32(chunk, crc)
                            remaining -= len(chunk)
                            if remaining <= 0:
                                break
                        block = block.next_block
                    checksum_map.entries.append((1, len(checksum_map.checksums)))
                    checksum_map.checksums.append(crc & 0xFFFFFFFF)
                checksum_map.file_id_count = len(checksum_map.entries)
                checksum_map.checksum_count = len(checksum_map.checksums)
            checksum_map.owner = owner
        else:
            checksum_map = None
            owner.checksum_map = None

        # Recalculate offsets and sizes.
        header.sector_count = blocks.block_count
        alloc_table.sector_count = blocks.block_count
        data_header.sector_count = blocks.block_count
        header.sector_size = self.header.sector_size
        data_header.sector_size = self.header.sector_size

        blocks_bytes = blocks.serialize()
        alloc_bytes = alloc_table.serialize()
        block_entry_bytes = (
            block_entry_map.serialize()
            if target_version < 6 and block_entry_map is not None
            else b""
        )
        manifest_bytes = manifest.serialize()
        checksum_bytes = (
            checksum_map.serialize()
            if target_version > 1 and checksum_map is not None
            else b""
        )

        header_size = 44
        data_header.first_sector_offset = (
            header_size
            + len(blocks_bytes)
            + len(alloc_bytes)
            + len(block_entry_bytes)
            + len(manifest_bytes)
            + len(checksum_bytes)
        )
        data_header_bytes = data_header.serialize()

        total_data = data_header.sectors_used * data_header.sector_size
        header.file_size = (
            data_header.first_sector_offset + len(data_header_bytes) + total_data
        )

        header_bytes = header.serialize()

        with open(out_path, "wb") as out:
            out.write(header_bytes)
            out.write(blocks_bytes)
            out.write(alloc_bytes)
            if block_entry_bytes:
                out.write(block_entry_bytes)
            out.write(manifest_bytes)
            if checksum_bytes:
                out.write(checksum_bytes)
            out.write(data_header_bytes)

            written = 0
            self.stream.seek(self.data_header.first_sector_offset)
            while written < total_data:
                chunk = self.stream.read(min(1024 * 1024, total_data - written))
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if progress:
                    progress(written, total_data)


    def defragment(
        self,
        out_path: str,
        progress: Callable[[int, int], None] | None = None,
        cancel_flag: object | None = None,
    ) -> None:
        """Write a defragmented copy of this GCF archive to ``out_path``.

        The implementation rewrites the allocation table so that all sectors
        for each block are stored sequentially without staging the entire file
        in memory.  A new cache file is written to ``out_path``; the in-memory
        representation of this instance is left untouched.  Only GCF archives
        are supported.

        Parameters
        ----------
        out_path:
            Destination path for the defragmented archive.
        progress:
            Optional callback invoked with ``(bytes_done, bytes_total)`` after
            each sector is copied.
        cancel_flag:
            Optional mutable object that evaluates to ``True`` when the
            operation should be cancelled.  Types such as ``threading.Event`` or
            ``[bool]`` (single-item lists) are supported.
        """

        if not self.is_parsed:
            raise ValueError("Cache file needs to be read first.")
        if not self.is_gcf():
            raise ValueError("Only GCF archives can be defragmented")

        fragmented, used = self._get_item_fragmentation(0)
        if fragmented == 0:
            raise ValueError("Archive is already defragmented.")

        sector_size = self.data_header.sector_size
        terminator = self.alloc_table.terminator

        new_alloc: list[int] = []

        files = [
            m
            for m in self.manifest.manifest_entries
            if (m.directory_flags & CacheFileManifestEntry.FLAG_IS_FILE) != 0
        ]

        # First pass: rebuild allocation tables without touching file data.
        for mentry in files:
            block = mentry.first_block
            while block is not None:
                sectors = list(block.sectors)
                block._first_sector_index = len(new_alloc)
                block.file_data_offset = len(new_alloc) * sector_size
                block.file_data_size = len(sectors) * sector_size

                for i, _ in enumerate(sectors):
                    if i == len(sectors) - 1:
                        new_alloc.append(terminator)
                    else:
                        new_alloc.append(len(new_alloc) + 1)
                block = block.next_block

        # Update tables with new allocation info
        self.alloc_table.entries = new_alloc
        self.alloc_table.sector_count = len(new_alloc)
        self.alloc_table.first_unused_entry = len(new_alloc)
        self.blocks.blocks_used = len(self.blocks.blocks)
        self.blocks.last_block_used = len(self.blocks.blocks) - 1
        self.data_header.sector_count = len(new_alloc)
        self.data_header.sectors_used = len(new_alloc)

        total_bytes = len(new_alloc) * sector_size
        bytes_done = 0

        def _cancelled() -> bool:
            if cancel_flag is None:
                return False
            if isinstance(cancel_flag, (list, tuple)):
                return bool(cancel_flag[0])
            if hasattr(cancel_flag, "is_set"):
                return bool(cancel_flag.is_set())
            return bool(cancel_flag)

        with open(out_path, "wb") as out:
            out.write(self.header.serialize())
            out.write(self.blocks.serialize())
            out.write(self.alloc_table.serialize())
            if self.block_entry_map is not None:
                out.write(self.block_entry_map.serialize())
            out.write(self.manifest.header_data)
            out.write(self.manifest.manifest_stream.getvalue())
            if self.checksum_map is not None:
                out.write(self.checksum_map.serialize())

            self.data_header.first_sector_offset = out.tell() + 24
            out.write(self.data_header.serialize())

            for mentry in files:
                if _cancelled():
                    break
                block = mentry.first_block
                while block is not None and not _cancelled():
                    for sector in block.sectors:
                        out.write(sector.get_data())
                        bytes_done += sector_size
                        if progress:
                            progress(bytes_done, total_bytes)
                        if _cancelled():
                            break
                    block = block.next_block

        if progress and bytes_done < total_bytes:
            progress(bytes_done, total_bytes)

        # Re-open the original stream in case subsequent operations are
        # performed on this instance.
        if self.stream is None and self.filename:
            self.stream = open(self.filename, "rb")


    # Private Methods

    def _read_directory(self):

        if self.is_ncf():
            # Make NCF files "readable" by a configurable folder much like Steam's "common" folder

            name = ".".join(self.filename.split(".")[:-1])
            path = self.ncf_folder_pattern % (dict(name=name, file=self.filename))

            package = FilesystemPackage()
            package.parse(path)

        elif self.is_gcf():
            package = self

        manifest_entry = self.manifest.manifest_entries[0]

        # Fill in root.
        self.root = DirectoryFolder(self, package=package)
        self.root.index = 0
        self.root._manifest_entry = manifest_entry
        self._read_directory_table(self.root)

    def _read_directory_table(self, folder):
        i = folder._manifest_entry.child_index

        while i != 0xFFFFFFFF and i != 0:
            manifest_entry = self.manifest.manifest_entries[i]
            is_file = manifest_entry.directory_flags & CacheFileManifestEntry.FLAG_IS_FILE != 0

            # Create our entry.
            klass = DirectoryFile if is_file else DirectoryFolder
            entry = klass(folder, manifest_entry.name, self)

            entry._manifest_entry = manifest_entry
            entry.item_size = manifest_entry.item_size
            entry.index = manifest_entry.index

            folder.items[entry.name] = entry

            if is_file:
                # Make sure it's a GCF before we read.
                if self.is_gcf():
                    self._read_file_table(entry)

            else:
                self._read_directory_table(entry)

            i = manifest_entry.next_index

    @raise_ncf_error
    def _read_file_table(self, entry):

        # Flags
        # entry.flags = self.blocks[entry.index].entry_flags
        entry.sectors = []
        entry.num_of_blocks = ceil(
            float(entry.size()) / float(self.data_header.sector_size)
        )

        for block in entry._manifest_entry.blocks:
            if block is None:
                entry.sectors = []
                break
            entry.sectors.extend(block.sectors)
            self.complete_available += block.file_data_size

        fragmented, _used = self._get_item_fragmentation(entry.index)
        entry.is_fragmented = fragmented > 0

        entry.is_user_config = entry.index in self.manifest.user_config_entries
        entry.is_minimum_footprint = entry.index in self.manifest.minimum_footprint_entries

    @raise_ncf_error
    def _merge_file_blocks(self, entry):
        terminator = 0xFFFFFFFF if self.alloc_table.is_long_terminator == 1 else 0xFFFF

        # If we are in one block, return plz.
        if not entry.first_block.next_block is not None:
            return

        # Go through the blocks of each file.
        for block in entry.blocks:

            # Get our first sector.
            sector_index = block.first_sector_index

            # From that, find the last sector in the block.
            while self.alloc_table[sector_index] != terminator:
                sector_index = self.alloc_table[sector_index]

            # Set the link from the last sector in the previous block to the first sector in this block.
            self.alloc_table[sector_index] = block.first_sector_index

    # Internal methods.

    def _join_path(self, *args):
        return STEAM_TERMINATOR.join(args)

    @raise_parse_error
    @raise_ncf_error
    def _size(self, file):
        return self.manifest.manifest_entries[file.index].item_size

    @raise_parse_error
    @raise_ncf_error
    def _open_file(self, file, mode, key=None):
        return GCFFileStream(file, self, mode, key)

    @raise_parse_error
    @raise_ncf_error
    def _extract_folder(self, folder, where, recursive, keep_folder_structure, item_filter=None, key=None):

        if keep_folder_structure:
            try:
                os.makedirs(os.path.join(where, folder.sys_path()))
            except os.error:
                pass

        # Loop over the folder and extract files and folders (if recursive)
        for entry in folder:
            # Don't bother recursing (and creating the folder) if no files are left after the filter.
            if entry.is_folder() and recursive and ((item_filter is None) or (len([x for x in entry.all_files() if item_filter(x)]) > 0)):
                self._extract_folder(entry, where, True, keep_folder_structure, item_filter, key)
            elif entry.is_file():
                if (item_filter is None) or item_filter(entry):
                    self._extract_file(entry, where, keep_folder_structure, key)

    @raise_parse_error
    @raise_ncf_error
    def _extract_file(self, file, where, keep_folder_structure, key=None):
        if keep_folder_structure:
            path = os.path.join(where, file.sys_path())
        else:
            path = os.path.join(where, file.name)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if (
            file._manifest_entry.directory_flags
            & CacheFileManifestEntry.FLAG_IS_ENCRYPTED
            and key is None
        ):
            raise ValueError("File is encrypted but no decryption key was provided")

        cacheStream = self._open_file(file, "rb", key=key)
        data = cacheStream.readall()
        cacheStream.close()

        with open(path, "wb") as fsHandle:
            fsHandle.write(data)

    # Public Methods
    def is_ncf(self):
        return self.header.is_ncf()

    def is_gcf(self):
        return self.header.is_gcf()

    @raise_parse_error
    @raise_ncf_error
    def complete_percent(self, range=100):
        return float(self.complete_available) / float(self.complete_total) * float(range)

    @raise_parse_error
    @raise_ncf_error
    def extract(self, where, recursive=True, keep_folder_structure=True, filter=None, key=None):
        self._extract_folder(self.root, where, recursive, keep_folder_structure, filter, key)

    @raise_parse_error
    @raise_ncf_error
    def extract_minimum_footprint(self, where, keep_folder_structure=True, key=None):
        self._extract_folder(
            self.root,
            where,
            True,
            keep_folder_structure,
            lambda x: x.is_minimum_footprint and not (os.path.exists(os.path.join(where, x.sys_path())) and x.is_user_config),
            key,
        )

    def open(self, filename, mode, key=None):
        # Use file.open instead of _open_file as we may be parsing an NCF
        return self.root[filename].open(mode, key)

    def __len__(self):
        return len(self.root)

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, name):
        return self.root[name]

    # ------------------------------------------------------------------
    def is_fragmented(self) -> bool:
        """Return ``True`` if the archive contains fragmented data blocks.

        Uses a direct translation of HLLib's ``CGCFFile::GetItemFragmentation``
        to examine the directory tree and count fragmented versus used data
        blocks.  ``True`` is returned if any blocks are fragmented.  Only
        meaningful for GCF archives; NCF files do not store file data.
        """

        if not self.is_parsed or not self.is_gcf() or not self.manifest:
            return False
        fragmented, _used = self._get_item_fragmentation(0)
        return fragmented > 0

    # ------------------------------------------------------------------
    def _validate_file(self, entry) -> Optional[str]:
        """Internal helper implementing HLLib's validation routine.

        Returns an error string if validation fails or ``None`` if the file
        appears to be valid.  Encrypted files and files without a checksum are
        treated as valid because their contents cannot be verified.
        """

        manifest_entry = entry._manifest_entry

        # Determine how much data is available by walking the block chain.
        size = 0
        block = manifest_entry.first_block
        while block is not None:
            size += block.file_data_size
            block = block.next_block
        if size != manifest_entry.item_size:
            return "size mismatch"

        if (
            manifest_entry.directory_flags
            & CacheFileManifestEntry.FLAG_IS_ENCRYPTED
        ):
            return None  # Can't validate encrypted data, assume OK.

        if (
            self.checksum_map is None
            or manifest_entry.checksum_index == 0xFFFFFFFF
        ):
            return None  # No checksum information available.

        try:
            count, first = self.checksum_map.entries[manifest_entry.checksum_index]
        except Exception:
            return "checksum index out of range"

        stream = entry.open("rb")
        try:
            remaining = manifest_entry.item_size
            i = 0
            while remaining > 0 and i < count:
                to_read = min(CACHE_CHECKSUM_LENGTH, remaining)
                chunk = stream.read(to_read)
                if len(chunk) != to_read:
                    return "size mismatch"
                chk = (adler32(chunk, 0) & 0xFFFFFFFF) ^ (
                    zlib.crc32(chunk) & 0xFFFFFFFF
                )
                if chk != self.checksum_map.checksums[first + i]:
                    return "checksum mismatch"
                remaining -= to_read
                i += 1
            if remaining > 0:
                return "size mismatch"
        finally:
            stream.close()

        return None

    def validate(self, progress=None):
        """Validate file data and return a list of error strings."""

        errors = []
        if not self.is_parsed:
            return ["Cache file not parsed"]

        files = self.root.all_files()
        total = len(files)

        for i, entry in enumerate(files):
            error = self._validate_file(entry)
            if error:
                errors.append(f"{entry.path()}: {error}")
            if progress:
                try:
                    progress(i + 1, total)
                except Exception:
                    pass

        return errors

    def count_complete_files(self) -> tuple[int, int]:
        """Return a tuple of (complete, total) file counts."""
        if not self.is_parsed or not self.manifest:
            return (0, 0)

        files = self.root.all_files()
        complete = 0
        for entry in files:
            manifest_entry = getattr(entry, "_manifest_entry", None)
            if not manifest_entry:
                continue
            size = 0
            block = manifest_entry.first_block
            while block is not None:
                size += block.file_data_size
                block = block.next_block
            if size >= manifest_entry.item_size:
                complete += 1
        return complete, len(files)

    def _get_item_fragmentation(self, item_index: int) -> tuple[int, int]:
        """Return ``(fragmented, used)`` block counts for the given item.

        This is a direct translation of ``CGCFFile::GetItemFragmentation``.  If
        ``item_index`` refers to a folder the counts for all child items are
        accumulated recursively.
        """

        entry = self.manifest.manifest_entries[item_index]
        if (entry.directory_flags & CacheFileManifestEntry.FLAG_IS_FILE) == 0:
            fragmented = 0
            used = 0
            child = entry.child_index
            while child not in (0, 0xFFFFFFFF):
                f, u = self._get_item_fragmentation(child)
                fragmented += f
                used += u
                child = self.manifest.manifest_entries[child].next_index
            return fragmented, used

        terminator = self.alloc_table.terminator
        last_sector = self.data_header.sector_count
        fragmented = 0
        used = 0

        block = entry.first_block
        while block is not None:
            block_size = 0
            sector_index = block._first_sector_index
            while sector_index != terminator and block_size < block.file_data_size:
                if last_sector != self.data_header.sector_count and last_sector + 1 != sector_index:
                    fragmented += 1
                used += 1
                last_sector = sector_index
                sector_index = self.alloc_table[sector_index]
                block_size += self.data_header.sector_size
            block = block.next_block

        return fragmented, used

class CacheFileHeader:

    def __init__(self, owner):
        self.owner = owner
    def parse(self, data):
        (self.header_version,
         self.cache_type,
         self.format_version,
         self.application_id,
         self.application_version,
         self.is_mounted,
         self.dummy1,
         self.file_size,
         self.sector_size,
         self.sector_count,
         self.checksum) = struct.unpack("<11L", data)

    def serialize(self):
        data = struct.pack("<10L", self.header_version, self.cache_type, self.format_version, self.application_id,
                           self.application_version, self.is_mounted, self.dummy1, self.file_size, self.sector_size, self.sector_count)
        self.checksum = sum(data)
        return data + struct.pack("<L", self.checksum)

    def calculate_checksum(self):
        # Calculate Checksum..
        return struct.unpack("<L", self.serialize()[-4:])[0]

    def validate(self):
        # Check the usual stuff.
        if self.header_version != 1:
            raise ValueError("Invalid Cache File Header [HeaderVersion is not 1]")
        if not (self.is_ncf() or self.is_gcf()):
            raise ValueError("Invalid Cache File Header [Not GCF or NCF]")
        if self.is_ncf() and self.format_version != 1:
            raise ValueError("Invalid Cache File Header [Is NCF and version is not 1]")
        elif self.is_gcf() and self.format_version not in (1, 3, 5, 6):
            raise ValueError("Invalid Cache File Header [Is GCF and version is not 1, 3, 5, or 6]")
        # UPDATE: This fails on some files, namely the half-life files.
        #if self.is_mounted != 0:
        #   raise ValueError, "Invalid Cache File Header [Updating is not 0... WTF?]"
        if self.is_ncf() and self.file_size != 0:
            raise ValueError("Invalid Cache File Header [Is NCF and FileSize is not 0]")
        if self.is_ncf() and self.sector_size != 0:
            raise ValueError("Invalid Cache File Header [Is NCF and BlockSize is not 0]")
        if self.is_ncf() and self.sector_count != 0:
            raise ValueError("Invalid Cache File Header [Is NCF and BlockCount is not 0]")
        #if self.checksum != self.calculate_checksum():
        #    raise ValueError, "Invalid Cache File Header [Checksums do not match]"

    def is_ncf(self):
        return self.cache_type == 2
    def is_gcf(self):
        return self.cache_type == 1
    def get_blocks_length(self):
        return self.sector_size * self.sector_count + 32 # Block Size * Block Count + Block Header

class CacheFileBlockAllocationTable:

    def __init__(self, owner):
        self.owner = owner
        self.blocks = []

    def parse(self, stream):

        # Blocks Header
        (self.block_count,
         self.blocks_used,
         self.last_block_used,
         self.dummy1,
         self.dummy2,
         self.dummy3,
         self.dummy4) = struct.unpack("<7L", stream.read(28))
        self.checksum = sum(stream.read(4))

        # Block Entries
        for i in range(self.block_count):
            block = CacheFileBlockAllocationTableEntry(self)
            block.index = i
            block.parse(stream)
            self.blocks.append(block)

    def serialize(self):
        data = struct.pack("<7L", self.block_count, self.blocks_used, self.last_block_used, self.dummy1, self.dummy2, self.dummy3, self.dummy4)
        self.checksum = sum(data)
        return data + struct.pack("<L", self.checksum) + b"".join(x.serialize() for x in self.blocks)

    def calculate_checksum(self):
        return sum(self.serialize()[:24])

    def validate(self):
        if self.owner.header.sector_count != self.block_count:
            raise ValueError("Invalid Cache Block [Sector/BlockCounts do not match]")
        #print self.checksum, self.calculate_checksum()
        #if self.checksum != self.calculate_checksum():
        #    raise ValueError, "Invalid Cache Block [Checksums do not match]"

class CacheFileBlockAllocationTableEntry:

    FLAG_DATA    = 0x200F8000
    FLAG_DATA_2  = 0x200FC000
    FLAG_NO_DATA = 0x200F0000

    def __init__(self, owner):
        self.owner = owner

    def parse(self, stream):
        # Block Entry
        (
            self.entry_flags,
            self.file_data_offset,
            self.file_data_size,
            self._first_sector_index,
            self._next_block_index,
            self._prev_block_index,
            self.manifest_index,
        ) = struct.unpack("<7L", stream.read(28))
        # Maintain backwards compatibility with callers expecting ``flags``.
        self.flags = self.entry_flags

    def _get_sector_iterator(self):
        sector = self.first_sector
        while sector is not None:
            yield sector
            sector = sector.next_sector

    def _get_next_block(self):
        try:
            return self.owner.blocks[self._next_block_index]
        except IndexError:
            return None

    def _set_next_block(self, value):
        if value is None:
            self._next_block_index = 0
        else:
            self._next_block_index = value.index
            value._prev_block_index = self.index

    def _get_prev_block(self):
        try:
            return self.owner.blocks[self._prev_block_index]
        except IndexError:
            return None

    def _set_prev_block(self, value):
        if value is None:
            self._prev_block_index = 0
        else:
            self._prev_block_index = value.index
            value._next_block_index = self.index

    def _get_first_sector(self):
        """Return the first sector for this block or ``None`` if unused."""
        alloc_table = self.owner.owner.alloc_table
        # Block entries that do not reference any data use a sentinel index
        # equal to the allocation table's terminator value.  Creating a
        # ``CacheFileSector`` for these entries would attempt to index past the
        # end of the allocation table and raise ``IndexError``.
        if self._first_sector_index >= alloc_table.terminator:
            return None
        return CacheFileSector(self, self._first_sector_index)

    def _set_first_sector(self, value):
        self._first_sector_index = value.inde

    def _get_is_fragmented(self):
        alloc_table = self.owner.owner.alloc_table
        if self._first_sector_index >= alloc_table.terminator:
            return False
        return (alloc_table[self._first_sector_index] - self._first_sector_index) != -1

    next_block = property(_get_next_block, _set_next_block)
    prev_block = property(_get_prev_block, _set_prev_block)
    first_sector = property(_get_first_sector, _set_first_sector)
    sectors = property(_get_sector_iterator)
    is_fragmented = property(_get_is_fragmented)

    def serialize(self):
        return struct.pack(
            "<7L",
            self.entry_flags,
            self.file_data_offset,
            self.file_data_size,
            self._first_sector_index,
            self._next_block_index,
            self._prev_block_index,
            self.manifest_index,
        )

class CacheFileAllocationTable:

    def __init__(self, owner):
        self.owner = owner
        self.entries = []

    def __getitem__(self, i):
        return self.entries[i]

    def __setitem__(self, i, v):
        self.entries[i] = v

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def parse(self, stream):

        # Block Header
        (
            self.sector_count,
            self.first_unused_entry,
            self.is_long_terminator,
        ) = struct.unpack("<3L", stream.read(12))
        # ``uiChecksum`` in ``GCFFragmentationMapHeader`` is a simple 32-bit
        # sum of the three header fields using unsigned overflow semantics.
        # Older implementations incorrectly summed the raw bytes which caused
        # validation failures on legitimate v1 GCF files.
        (self.checksum,) = struct.unpack("<L", stream.read(4))

        self.terminator = 0xFFFFFFFF if self.is_long_terminator else 0xFFFF
        self.entries = unpack_dword_list(stream, self.sector_count)

    def serialize(self):
        data = struct.pack(
            "<3L",
            self.sector_count,
            self.first_unused_entry,
            self.is_long_terminator,
        )
        # Cache the checksum so subsequent calls to ``serialize`` or
        # ``calculate_checksum`` are in agreement with the on-disk format.
        self.checksum = self.calculate_checksum()
        return data + struct.pack("<L", self.checksum) + pack_dword_list(self.entries)

    def calculate_checksum(self):
        return (
            self.sector_count
            + self.first_unused_entry
            + self.is_long_terminator
        ) & 0xFFFFFFFF

    def validate(self):
        if self.owner.header.sector_count != self.sector_count:
            raise ValueError(
                "Invalid Cache Allocation Table [SectorCounts do not match]"
            )
        # Very old GCF files often contain an incorrect checksum here.  The
        # reference C++ implementation does not enforce this for legacy
        # archives so we only validate for newer formats where the field is
        # known to be reliable.
        if self.owner.header.format_version > 1:
            if self.checksum != self.calculate_checksum():
                raise ValueError(
                    "Invalid Cache Allocation Table [Checksums do not match]"
                )

class CacheFileBlockEntryMap:

    def __init__(self, owner):
        self.owner = owner
        # ``entries`` stores block entry indices in linked-list order so that
        # manifest map entries can be resolved to real block entries.
        self.entries: list[int] = []
        self.first_block_entry_index = 0
        self.last_block_entry_index = 0
        self.dummy0 = 0

    def parse(self, stream):
        # Full header: block count, first & last entry indices, dummy field and
        # checksum.  Older implementations only read the first DWORD which
        # resulted in misaligned reads for v1 archives.
        (
            self.block_count,
            self.first_block_entry_index,
            self.last_block_entry_index,
            self.dummy0,
            self.checksum,
        ) = struct.unpack("<5L", stream.read(20))

        raw_entries = [struct.unpack("<2L", stream.read(8)) for _ in range(self.block_count)]

        # Reconstruct a linear mapping from list position to block entry index
        # by traversing the linked list defined by the raw entries.
        ordered: list[int] = []
        index = self.first_block_entry_index
        visited = set()
        for _ in range(self.block_count):
            if index >= self.block_count or index in visited:
                break
            ordered.append(index)
            visited.add(index)
            index = raw_entries[index][1]

        # Fallback in case of malformed data where not all entries are linked.
        if len(ordered) < self.block_count:
            ordered.extend(i for i in range(self.block_count) if i not in visited)

        self.entries = ordered

    def serialize(self):
        self.block_count = len(self.entries)
        self.first_block_entry_index = self.entries[0] if self.entries else 0
        self.last_block_entry_index = self.entries[-1] if self.entries else 0
        self.dummy0 = 0

        # Build raw linked-list representation from the ordered list.
        raw_entries = [(self.block_count, self.block_count)] * self.block_count
        for pos, entry_index in enumerate(self.entries):
            prev_idx = self.entries[pos - 1] if pos > 0 else self.block_count
            next_idx = self.entries[pos + 1] if pos < self.block_count - 1 else self.block_count
            raw_entries[entry_index] = (prev_idx, next_idx)

        header = struct.pack(
            "<4L",
            self.block_count,
            self.first_block_entry_index,
            self.last_block_entry_index,
            self.dummy0,
        )
        self.checksum = sum(header)
        data = [header, struct.pack("<L", self.checksum)]
        data.extend(struct.pack("<2L", *e) for e in raw_entries)
        return b"".join(data)


class CacheFileManifest:

    FLAG_BUILD_MODE   = 0x00000001
    FLAG_IS_PURGE_ALL = 0x00000002
    FLAG_IS_LONG_ROLL = 0x00000004
    FLAG_DEPOT_KEY    = 0xFFFFFF00

    def __init__(self, owner):
        self.owner = owner
        self.manifest_entries = []
        self.hash_table_keys = []
        self.hash_table_indices = []

        # Contains ManifestIndex
        self.user_config_entries = []

        # Contains ManifestIndex
        self.minimum_footprint_entries = []

        # Contains FirstBlockIndex
        self.manifest_map_entries = []

    def parse(self, stream):
        # Header
        self.header_data = stream.read(56)
        (self.header_version,
         self.application_id,
         self.application_version,
         self.node_count,
         self.file_count,
         self.compression_block_size,
         self.binary_size,
         self.name_size,
         self.hash_table_key_count,
         self.num_of_minimum_footprint_files,
         self.num_of_user_config_files,
         self.depot_info,
         self.fingerprint,
         self.checksum) = struct.unpack("<14L", self.header_data)

        # 56 = size of header
        self.manifest_stream = BytesIO(stream.read(self.binary_size - 56))

        # Manifest Entries
        for i in range(self.node_count):
            entry = CacheFileManifestEntry(self)
            entry.index = i
            # 28 = size of ManifestEntry
            data = self.manifest_stream.read(28)
            entry.parse(data)
            self.manifest_entries.append(entry)
            if (entry.directory_flags & CacheFileManifestEntry.FLAG_IS_FILE) != 0:
                self.owner.complete_total += entry.item_size

        # Name Table
        self.filename_table = self.manifest_stream.read(self.name_size)

        # Info1 / HashTableKeys
        self.hash_table_keys = unpack_dword_list(self.manifest_stream, self.hash_table_key_count)

        # Info2 / HashTableIndices
        self.hash_table_indices = unpack_dword_list(self.manifest_stream, self.node_count)

        # Minimum Footprint Entries
        self.minimum_footprint_entries = unpack_dword_list(self.manifest_stream, self.num_of_minimum_footprint_files)

        # User Config Entries
        self.user_config_entries = unpack_dword_list(
            self.manifest_stream, self.num_of_user_config_files
        )

        # Older GCF v1 directories omit the manifest-map header entirely and
        # store the first-block indices directly after the manifest stream.
        if self.owner.header.format_version <= 1:
            self.map_header_version = 1
            self.map_dummy1 = 0
        else:
            (self.map_header_version, self.map_dummy1) = struct.unpack(
                "<2L", stream.read(8)
            )

        # Manifest Map Entries (FirstBlockIndex)
        self.manifest_map_entries = unpack_dword_list(stream, self.node_count)

    def serialize(self):
        # 56 = size of header
        # 32 = size of ManifestEntry + size of DWORD for HashTableIndices
        self.hash_table_key_count = len(self.hash_table_keys)
        self.num_of_user_config_files = len(self.user_config_entries)
        self.num_of_minimum_footprint_files = len(self.minimum_footprint_entries)
        self.name_size = len(self.filename_table)
        self.binary_size = 56 + 32 * self.node_count + self.name_size + 4 * (
            self.hash_table_key_count
            + self.num_of_user_config_files
            + self.num_of_minimum_footprint_files
        )

        manifest_data_parts = []
        for i in self.manifest_entries:
            manifest_data_parts.append(i.serialize())

        manifest_data_parts.append(self.filename_table)
        manifest_data_parts.append(pack_dword_list(self.hash_table_keys))
        manifest_data_parts.append(pack_dword_list(self.hash_table_indices))
        manifest_data_parts.append(pack_dword_list(self.minimum_footprint_entries))
        manifest_data_parts.append(pack_dword_list(self.user_config_entries))
        if self.owner.header.format_version > 1:
            manifest_data_parts.append(
                struct.pack("<2L", self.map_header_version, self.map_dummy1)
            )
        manifest_data_parts.append(pack_dword_list(self.manifest_map_entries))
        manifest_data = b"".join(manifest_data_parts)

        header_without_checksum = struct.pack(
            "<13L",
            self.header_version,
            self.application_id,
            self.application_version,
            self.node_count,
            self.file_count,
            self.compression_block_size,
            self.binary_size,
            self.name_size,
            self.hash_table_key_count,
            self.num_of_minimum_footprint_files,
            self.num_of_user_config_files,
            self.depot_info,
            self.fingerprint,
        )
        self.header_data = header_without_checksum

        self.checksum = adler32(header_without_checksum + b"\0\0\0\0" + manifest_data, 0) & 0xFFFFFFFF
        return header_without_checksum + struct.pack("<L", self.checksum) + manifest_data

    def validate(self):
        if self.owner.header.application_id != self.application_id:
            raise ValueError("Invalid Cache File Manifest [Application ID mismatch]")
        if self.owner.header.application_version != self.application_version:
            raise ValueError("Invalid Cache File Manifest [Application version mismatch]")
        #if self.checksum != self.calculate_checksum():
        #    raise ValueError, "Invalid Cache File Manifest [Checksum mismatch]"
        if self.owner.header.format_version > 1:
            if self.map_header_version != 1:
                raise ValueError(
                    "Invalid Cache File Manifest [ManifestHeaderMap's HeaderVersion is not 1]"
                )
            if self.map_dummy1 != 0:
                raise ValueError(
                    "Invalid Cache File Manifest [ManifestHeaderMap's Dummy1 is not 0]"
                )

    def calculate_checksum(self):
        # Blank out checksum and fingerprint + hack to get unsigned value.
        data = self.serialize()
        return adler32(data[:48] + b"\0\0\0\0\0\0\0\0" + data[56:], 0) & 0xffffffff

class CacheFileManifestEntry:

    FLAG_IS_FILE        = 0x00004000
    FLAG_IS_EXECUTABLE  = 0x00000800
    FLAG_IS_HIDDEN      = 0x00000400
    FLAG_IS_READ_ONLY   = 0x00000200
    FLAG_IS_ENCRYPTED   = 0x00000100
    FLAG_IS_PURGE_FILE  = 0x00000080
    FLAG_BACKUP_PLZ     = 0x00000040
    FLAG_IS_NO_CACHE    = 0x00000020
    FLAG_IS_LOCKED      = 0x00000008
    FLAG_IS_LAUNCH      = 0x00000002
    FLAG_IS_USER_CONFIG = 0x00000001

    def __init__(self, owner):
        self.owner = owner

    def _get_name(self):
        raw = self.owner.filename_table[self.name_offset:].split(b"\0", 1)[0]
        return raw.decode('utf-8', errors='replace')

    def _set_name(self, value):
        value_bytes = value.encode('utf-8')
        name_end = self.owner.filename_table[self.name_offset:].find(b"\0")
        table = bytearray(self.owner.filename_table)
        table[self.name_offset:self.name_offset + name_end] = value_bytes
        self.owner.filename_table = bytes(table)

    def _get_block_iterator(self):
        block = self.first_block
        while block is not None:
            yield block
            block = block.next_block

    def _get_first_block(self):
        index = self.owner.manifest_map_entries[self.index]
        # Older GCF versions store a block-entry-map indirection
        if self.owner.owner.header.format_version < 6 and self.owner.owner.block_entry_map:
            if index >= len(self.owner.owner.block_entry_map.entries):
                return None
            index = self.owner.owner.block_entry_map.entries[index]
        if index >= len(self.owner.owner.blocks.blocks):
            return None
        return self.owner.owner.blocks.blocks[index]

    def _set_first_block(self, value):
        if self.owner.owner.header.format_version < 6 and self.owner.owner.block_entry_map:
            try:
                mapped = self.owner.owner.block_entry_map.entries.index(value)
            except ValueError:
                mapped = value
            self.owner.manifest_map_entries[self.index] = mapped
        else:
            self.owner.manifest_map_entries[self.index] = value

    def parse(self, data):
        (self.name_offset,
         self.item_size,
         self.checksum_index,
         self.directory_flags,
         self.parent_index,
         self.next_index,
         self.child_index) = struct.unpack("<7L", data)

    blocks = property(_get_block_iterator)
    first_block = property(_get_first_block, _set_first_block)
    name = property(_get_name, _set_name)

    def serialize(self):
        return struct.pack("<7L", self.name_offset, self.item_size, self.checksum_index, self.directory_flags, self.parent_index, self.next_index, self.child_index)

class CacheFileChecksumMap:

    FLAG_IS_SIGNED      = 0x00000001
    FLAG_UNKNOWN        = 0xFFFFFFFE

    def __init__(self, owner):
        self.owner = owner

        # Contains (ChecksumCount, FirstChecksumIndex)
        self.entries = []

        # Contains Checksum
        self.checksums = []

    def parse(self, stream):

        (self.header_version,
         self.checksum_size,
         self.format_code,
         self.version,
         self.file_id_count,
         self.checksum_count) = struct.unpack("<6L", stream.read(24))

        for i in range(self.file_id_count):
            self.entries.append(struct.unpack("<2L", stream.read(8)))

        self.checksums = unpack_dword_list(stream, self.checksum_count)

        self.signature = stream.read(128)

    def serialize(self):
        data = [struct.pack("<6L", self.header_version, self.checksum_size, self.format_code, self.version, self.file_id_count, self.checksum_count)]
        data += [struct.pack("<2L", *i) for i in self.entries]
        data.append(pack_dword_list(self.checksums))
        data.append(self.signature)
        return b"".join(data)

    def validate(self):
        pass
        # NOTE: This check is incorrect on the test file (half-life 2 game dialog.gcf) I have.
        # if self.owner.directory.file_count != self.item_count:
        #     raise ValueError, "Invalid Cache File Checksum Map [ItemCount and FileCount don't match]"

class CacheFileSectorHeader:

    def __init__(self, owner):
        self.owner = owner
        self.format_version = owner.header.format_version

    def parse(self, data, format_version):
        self.format_version = format_version
        if format_version <= 3:
            (
                self.sector_count,
                self.sector_size,
                self.first_sector_offset,
                self.sectors_used,
                self.checksum,
            ) = struct.unpack("<5L", data)
            # Older GCF versions omit the application version field.
            self.application_version = self.owner.header.application_version
        else:
            (
                self.application_version,
                self.sector_count,
                self.sector_size,
                self.first_sector_offset,
                self.sectors_used,
                self.checksum,
            ) = struct.unpack("<6L", data)

    def serialize(self):
        self.checksum = self.calculate_checksum()
        if self.format_version <= 3:
            return struct.pack(
                "<5L",
                self.sector_count,
                self.sector_size,
                self.first_sector_offset,
                self.sectors_used,
                self.checksum,
            )
        return struct.pack(
            "<6L",
            self.application_version,
            self.sector_count,
            self.sector_size,
            self.first_sector_offset,
            self.sectors_used,
            self.checksum,
        )

    def validate(self):
        if (
            self.format_version > 3
            and self.application_version != self.owner.header.application_version
        ):
            raise ValueError(
                "Invalid Cache File Sector Header [ApplicationVersion mismatch]"
            )
        # Some early GCF revisions (notably version 1) report a truncated
        # ``sector_count`` in the data header that does not match the value in
        # the file header.  HLLib tolerates this discrepancy, so we only enforce
        # equality for newer formats where both fields are known to agree.
        if (
            self.format_version > 1
            and self.sector_count != self.owner.header.sector_count
        ):
            raise ValueError(
                "Invalid Cache File Sector Header [SectorCount mismatch]"
            )
        # Legacy GCFs may report a different sector size in the data header
        # than in the file header.  HLLib accepts this mismatch, so only enforce
        # equality for newer format revisions where both fields are known to
        # agree.
        if (
            self.format_version > 1
            and self.sector_size != self.owner.header.sector_size
        ):
            raise ValueError(
                "Invalid Cache File Sector Header [SectorSize mismatch]"
            )
        # Some early (v1) GCF files are known to store an invalid checksum in
        # the data header.  HLLib ignores this discrepancy, so we only enforce
        # checksum validation for newer format revisions.
        if self.format_version > 1 and self.checksum != self.calculate_checksum():
            raise ValueError(
                "Invalid Cache File Sector Header [Checksum mismatch]"
            )

    def calculate_checksum(self):
        # The checksum stored in the data header is a 32-bit unsigned sum of
        # the following fields.  Clamp intermediate results to 32 bits to match
        # the behavior of the original C++ implementation.
        checksum = 0
        for value in (
            self.sector_count,
            self.sector_size,
            self.first_sector_offset,
            self.sectors_used,
        ):
            checksum = (checksum + value) & 0xFFFFFFFF
        return checksum

class CacheFileSector:

    def __init__(self, owner, index):
        self.owner = owner
        self.cache = owner.owner.owner
        self.index = index
        self._next_index = self.cache.alloc_table[index]

    def _get_next_sector(self):
        if self._next_index == self.cache.alloc_table.terminator:
            return None
        return CacheFileSector(self.owner, self._next_index)

    def _set_next_sector(self, value):
        self._next_index = value.index

    def get_data(self):
        size = self.cache.data_header.sector_size
        self.cache.stream.seek(self.cache.data_header.first_sector_offset + size*self.index, os.SEEK_SET)
        return self.cache.stream.read(size)

    next_sector = property(_get_next_sector, _set_next_sector)

class GCFFileStream:

    def __init__(self, entry, owner, mode, key=None):
        self.entry = entry
        self.owner = owner
        self.mode = mode
        self.key = key

        sector_size = self.owner.data_header.sector_size

        sectors = getattr(self.entry, "sectors", None) or []
        if not sectors:
            sectors = []
            for block in self.entry._manifest_entry.blocks:
                if block is None:
                    continue
                sectors.extend(block.sectors)
            self.entry.sectors = sectors

        raw = b"".join(sect.get_data() for sect in sectors)
        if key:
            raw = decrypt_gcf_data(raw, key)

        self.sectors = [raw[i : i + sector_size] for i in range(0, len(raw), sector_size)]
        self.position = 0

    # Iterator protocol.
    def __iter__(self):
        return self

    def __next__(self):
        return self.readline()

    # File protocol.
    def flush(self):
        # Nothing right now...
        pass

    def close(self):
        # Nothing right now...
        pass

    def tell(self):
        return self.position

    def seek(self, offset, origin=None):

        def err():
            raise IOError("Attempting to seek past end of file")

        if origin == os.SEEK_SET or origin is None:
            if offset > self.entry.item_size:
                err()
            self.position = offset

        elif origin == os.SEEK_CUR:
            if offset + self.position > self.entry.item_size:
                err()
            self.position += offset

        elif origin == os.SEEK_END:
            if offset > self.entry.item_size or offset < 0:
                err()
            self.position = self.entry.item_size - offset

    def readall(self):
        return self.read(0)

    def readline(self, size=-1):

        # Our count for the size parameter.
        count = 0
        # Bytes are immutable... use a list
        chars = []

        lastchar = b""

        # Loop over our data one byte at a time looking for line breaks
        while True:
            lastchar = self.read(1)
            # If we get a CR
            if lastchar == b"\r":
                # Strip out a LF if it comes next
                if self.read(1) != b"\n":
                    self.position -= 1
                break
            elif lastchar == b"\n":
                break
            elif count > size and size > 0:
                # FIXME: What does the file module do when we have a size
                # hint? Does it include newline in the count? What about
                # CRLF? Does count as one or two chars?
                break

            # Characters
            chars.append(lastchar)

        line = b"".join(chars)
        if self.is_text_mode():
            return line.decode('utf-8', errors='replace')
        return line

    def readlines(self, sizehint=-1):

        # Our count for the size parameter.
        count = 0
        lines = []

        while True:
            data = self.readline()
            lines.append(data)
            count += len(data)
            # If we have surpassed the sizehint, break
            if count > sizehint and sizehint > 0:
                break

        return lines

    def read(self, size=0):

        if not self.is_read_mode():
            raise AttributeError("Cannot read from file with current mode")

        sector_size = self.owner.data_header.sector_size
        sector_index, offset = divmod(self.position, sector_size)

        if not self.sectors:
            return b""

        # Raise an error if we read past end of file.
        if self.position + size > self.entry.item_size:
            raise IOError("Attempting to read past end of file")

        # One file isn't always in just one block.
        # We have to read multiple blocks sometimes in order to get a file.
        read_pos = 0

        # Bytes are immutable... use a list.
        data = []

        if size < 1:
            size = self.entry.item_size
            # Get all the data by looping over the sectors.
            for sector in self.sectors[sector_index:]:
                take = min(sector_size - offset, size)
                data.append(sector[offset:offset + take])
                size -= take
                if size <= 0:
                    break
                offset = 0
            self.position = self.entry.item_size
        else:
            while read_pos < size:
                take = min(size - read_pos, sector_size - offset)
                data.append(self.sectors[sector_index][offset:offset + take])
                sector_index += 1
                offset = 0
                read_pos += take
            self.position += read_pos

        # TYPE CHANGE!
        # Data - from list to bytes.
        data = b"".join(data)

        if self.is_text_mode():
            return data.decode('utf-8', errors='replace').replace("\r", "")

        return data

    def write(self, data):
        pass

    def is_binary_mode(self):
        return "b" in self.mode

    def is_text_mode(self):
        return "b" not in self.mode

    def is_read_mode(self):
        return "r" in self.mode

    def is_write_mode(self):
        return "w" in self.mode or "r+" in self.mode

    def is_append_mode(self):
        return "a" in self.mode
