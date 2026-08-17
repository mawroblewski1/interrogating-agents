"""
Streaming reader for the Parler dataset zip(s).

- Finds inputs by GLOB (you do NOT need to know the zip filename): point it at a folder.
- Handles COMPLETE zips via the standard library (zipfile).
- Falls back to a raw-deflate reader for TRUNCATED / partial (.crdownload) files,
  so it also works on an incomplete download.
- Streams: never decompresses the whole archive into memory.

Yields parsed JSON records (dicts), one per post/comment line.

NOT A CLI SCRIPT -- this is a library module, imported by Stages 1, 3, and 5 (and
06_bucket_frequency.py), not run directly (no argparse, no __main__ block). Typical
usage from another script:

  import parler_io
  paths = parler_io.find_inputs("/path/to/parler_folder")   # or a single zip file path
  for rec in parler_io.iter_records(paths):
      body = rec.get("body") or ""
      # ... do something with each post/comment record ...
"""
from __future__ import annotations
import os, glob, json, struct, zlib, zipfile


def find_inputs(path: str) -> list[str]:
    """path can be a directory (globs *.zip and *.crdownload) or a single file."""
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "*.zip")))
        found += sorted(glob.glob(os.path.join(path, "*.crdownload")))
        return found
    return [path]


def _iter_zipfile(zpath: str):
    """Complete zip: iterate every *.ndjson entry line by line (decompressed on the fly)."""
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(".ndjson"):
                continue
            with z.open(name) as fh:                # ZipExtFile: iterates by lines (bytes)
                for bline in fh:
                    bline = bline.strip()
                    if not bline:
                        continue
                    try:
                        yield json.loads(bline)
                    except Exception:
                        continue


def _iter_raw(zpath: str):
    """Truncated/partial zip fallback. Parses local file headers sequentially and
    raw-inflates each deflate entry using the compressed size stored in the header.
    Stops cleanly at the truncation point of a partial download."""
    with open(zpath, "rb") as f:
        while True:
            sig = f.read(4)
            if sig != b"PK\x03\x04":
                break
            hdr = f.read(26)
            if len(hdr) < 26:
                break
            (ver, flags, method, mt, md, crc,
             csize, usize, fnlen, extralen) = struct.unpack("<HHHHHIIIHH", hdr)
            f.read(fnlen); f.read(extralen)         # skip name + extra
            # supported: deflate (8) with compressed size in header (Parler: flags bit3 == 0)
            if method != 8 or (flags & 0x08) or csize == 0:
                if csize:
                    f.seek(csize, 1)
                    continue
                break
            remaining = csize
            d = zlib.decompressobj(-15)
            buf = b""
            truncated = False
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    truncated = True
                    break
                remaining -= len(chunk)
                try:
                    buf += d.decompress(chunk)
                except Exception:
                    truncated = True
                    break
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except Exception:
                            pass
            if truncated:
                break                                # partial file: nothing reliable after this
            try:
                buf += d.flush()
            except Exception:
                pass
            for line in buf.split(b"\n"):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        pass


def iter_records(paths):
    """Yield parsed records across all input files, choosing the right reader per file."""
    for zp in paths:
        try:
            yield from _iter_zipfile(zp)
        except zipfile.BadZipFile:
            yield from _iter_raw(zp)
        except Exception:
            yield from _iter_raw(zp)
