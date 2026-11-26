#!/usr/bin/env python
"""Build the Zenodo fetch manifest: which data file is in which archive, with hashes.

    uv run python scripts/make_zenodo_manifest.py          # write the two CSVs
    uv run python scripts/make_zenodo_manifest.py --check  # verify against the zips, no write

The HST Zenodo record (concept DOI 10.5281/zenodo.24052764) holds the release/<name>.zip
archives built by make_release_zips.py; each unzips into repo-relative data/stamps/... paths.
The manifest pins the VERSION record (24052765) in every URL, so the hashes can never shift
under a fetch (concept-record file URLs 404 anyway; a new data version means re-running this
script with the new record id in ZENODO_VERSION_URL). Per-file hashes are computed from the
zip contents -- the bytes a downloader actually gets -- and checked against the working tree
so drift between the stamps on disk and the published archives is reported.

Output (tracked):
    info/zenodo_archives.csv   archive, sha256, size_bytes, url          (one row per zip)
    info/zenodo_files.csv      path, archive, sha256, size_bytes         (one row per data file)

fetch_data.py reads both; the product browser (lensfoundry.github.io build.py) reads
zenodo_archives.csv for its download links.
"""
import argparse
import csv
import hashlib
import os
import sys
import zipfile

ws_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = os.path.join(ws_path, 'release')
ARCHIVES_CSV = os.path.join(ws_path, 'info', 'zenodo_archives.csv')
FILES_CSV = os.path.join(ws_path, 'info', 'zenodo_files.csv')

ZENODO_VERSION_URL = 'https://zenodo.org/records/24052765/files'


def sha256_of(buf, h=None):
    h = h or hashlib.sha256()
    h.update(buf)
    return h


def build(check=False):
    zips = sorted(f for f in os.listdir(RELEASE) if f.endswith('.zip'))
    if not zips:
        raise SystemExit(f'no archives in {RELEASE} (run make_release_zips.py first)')
    arch_rows, file_rows, drift = [], [], []
    for name in zips:
        zpath = os.path.join(RELEASE, name)
        with zipfile.ZipFile(zpath) as z:
            zh = hashlib.sha256()
            with open(zpath, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    zh.update(chunk)
            for member in z.namelist():
                if not member.startswith('data/'):
                    continue   # the archive's own top-level manifest.csv is not fetched
                fh = hashlib.sha256()
                fh.update(z.read(member))
                digest = fh.hexdigest()
                local = os.path.join(ws_path, member)
                if os.path.isfile(local):
                    lh = hashlib.sha256()
                    with open(local, 'rb') as f:
                        for chunk in iter(lambda: f.read(1 << 20), b''):
                            lh.update(chunk)
                    if lh.hexdigest() != digest:
                        drift.append(member)
                file_rows.append(dict(path=member, archive=name, sha256=digest,
                                      size_bytes=z.getinfo(member).file_size))
        arch_rows.append(dict(archive=name, sha256=zh.hexdigest(),
                              size_bytes=os.path.getsize(zpath),
                              url=f'{ZENODO_VERSION_URL}/{name}'))
        print(f'  {name:24s} {len([r for r in file_rows if r["archive"] == name]):5d} files  '
              f'{arch_rows[-1]["size_bytes"] / 1e6:7.1f} MB')
    if drift:
        print(f'WARNING: {len(drift)} working-tree files differ from the published archives:',
              file=sys.stderr)
        for m in drift[:20]:
            print(f'  {m}', file=sys.stderr)
    if check:
        ok_a = ok_f = True
        for path, rows, key in ((ARCHIVES_CSV, arch_rows, 'archive'), (FILES_CSV, file_rows, 'path')):
            if not os.path.isfile(path):
                print(f'MISSING: {path}', file=sys.stderr)
                continue
            with open(path) as f:
                existing = list(csv.DictReader(f))
            want = [[str(r[k]) for k in existing[0]] for r in rows]
            got = [[r[k] for k in existing[0]] for r in existing]
            same = want == got
            print(f'{os.path.basename(path)}: {"current" if same else "STALE (re-run without --check)"}')
            ok_a &= same
        return 0 if ok_a else 1
    for path, rows, fields in ((ARCHIVES_CSV, arch_rows, ['archive', 'sha256', 'size_bytes', 'url']),
                               (FILES_CSV, file_rows, ['path', 'archive', 'sha256', 'size_bytes'])):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {os.path.relpath(path, ws_path)} ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true', help='verify the CSVs against release/, no write')
    sys.exit(build(ap.parse_args().check))
