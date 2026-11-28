#!/usr/bin/env python
"""Refresh the connectivity backstop on every shipped PSF kernel (2026-11-28 recheck).

The 2026-11-04/05 rebuild campaign predates remove_detached_components (added 2026-11-17
for full kernels, 2026-11-18 for trimmed crops), so the shipped full kernels still carry the
star-stack noise/companion blobs the backstop exists to remove -- visible as wing clumps in
the browser PSF panels. This script re-applies the backstop to:

    data/psf/<sample>/<bm>/<cat>/<lens>/<filt>/psf_kernel.fits     (zero detached >=1e-3 px,
                                                                    renormalise, BKSTOPPX card)
    data/psf/.../psf_kernel_err.fits                              (same pixels zeroed)
    data/stamps/<same>/<...>/psf.fits                              (re-trimmed from the cleaned
                                                                    full kernel, header kept)
    data/stamps/.../psf_err.fits                                   (same crop)

Evidence for the rule (recorded in AGENTS.md *The 2026-11-28 PSF wing recheck*): the blobs
are absent from the injected model kernels, have bootstrap error 0.3-2x their amplitude, sit
at scattered non-axial positions with no cross-lens clustering, and the star crops of the
flagged lenses are clean -- so they are contaminants/noise, not PSF structure. (Contrast
JWST/NIRCam, where the wing blobs are confirmed REAL by the star-built check kernel and must
NOT be backstopped.)

Dry run by default; --apply writes. Idempotent.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

ws_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_psf import remove_detached_components, trim_kernel_to_amplitude  # noqa: E402

THR = 1e-3


def process(kernel_path, apply):
    rel = os.path.relpath(kernel_path, ws_path)
    out = dict(path=rel, zeroed=0, rewritten=False)
    with fits.open(kernel_path) as h:
        kernel, hdr = h[0].data.astype(float), h[0].header
    cleaned, n = remove_detached_components(kernel, THR)
    out['zeroed'] = n
    if n == 0:
        return out
    pre_sum = float(kernel.sum())
    cleaned = cleaned / cleaned.sum()
    hdr['BKSTOPPX'] = (n, 'px zeroed by connectivity backstop refresh')
    hdr['BKSTOPTH'] = (THR, 'backstop threshold x peak')
    stamp_dir = os.path.join(ws_path, 'data', 'stamps',
                             os.path.relpath(os.path.dirname(kernel_path),
                                             os.path.join(ws_path, 'data', 'psf')))
    psf_path = os.path.join(stamp_dir, 'psf.fits')
    err_full = os.path.join(os.path.dirname(kernel_path), 'psf_kernel_err.fits')
    err_stamp = os.path.join(stamp_dir, 'psf_err.fits')
    # mask of zeroed pixels, to hit the error maps identically
    dropped = (kernel > 0) & (cleaned == 0)
    sub, size = trim_kernel_to_amplitude(cleaned, THR)
    out['trim_size'] = size
    out['rewritten'] = True
    out['pre_sum'] = pre_sum
    if apply:
        fits.PrimaryHDU(data=cleaned.astype(np.float32), header=hdr).writeto(kernel_path,
                                                                             overwrite=True)
        if os.path.exists(psf_path):
            hdr_t = fits.getheader(psf_path)
            fits.PrimaryHDU(data=sub.astype(np.float32), header=hdr_t).writeto(psf_path,
                                                                               overwrite=True)
        if os.path.exists(err_full):
            with fits.open(err_full) as h:
                e, eh = h[0].data.astype(float), h[0].header
            e[dropped] = 0.0
            eh['BKSTOPPX'] = (n, 'px zeroed (kernel connectivity backstop)')
            fits.PrimaryHDU(data=e.astype(np.float32), header=eh).writeto(err_full,
                                                                          overwrite=True)
            if os.path.exists(err_stamp):
                cy = cx = e.shape[0] // 2
                half = (size - 1) // 2
                e_sub = e[cy - half:cy + half + 1, cx - half:cx + half + 1]
                hdr_e = fits.getheader(err_stamp)
                fits.PrimaryHDU(data=e_sub.astype(np.float32), header=hdr_e).writeto(
                    err_stamp, overwrite=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='write the changes (dry run otherwise)')
    a = ap.parse_args()
    kernels = sorted(glob.glob(os.path.join(ws_path, 'data', 'psf', '*', '*', '*', '*', '*',
                                            'psf_kernel.fits')))
    print(f'{len(kernels)} full kernels; {"APPLYING" if a.apply else "dry run"}')
    rows = [process(p, a.apply) for p in kernels]
    changed = [r for r in rows if r['zeroed'] > 0]
    print(f"{len(changed)} kernels had detached blobs zeroed; "
          f"{sum(r['zeroed'] for r in rows)} px total")
    log = os.path.join(ws_path, 'diagnostics', 'psf_recheck', 'backstop_refresh.json')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w') as f:
        json.dump(dict(apply=bool(a.apply), n_kernels=len(kernels),
                       n_changed=len(changed), rows=changed), f, indent=1)
    print(f'wrote {log}')
    for r in changed:
        rel = r['path'].replace('data/psf/', '').replace('/psf_kernel.fits', '')
        print(f"  {rel:<58} {r['zeroed']:>4}px -> trim {r.get('trim_size')}")


if __name__ == '__main__':
    main()
