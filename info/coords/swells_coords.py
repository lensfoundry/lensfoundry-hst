# SWELLS-only grade-A lenses (not in the SLACS-sourced slacs_coords.py): the six GO-11978
# WFPC2 lenses and the six GO-12292 ACS snapshot lenses added to the swells sample
# 2026-11-09.
# Source: Treu et al. 2011 (SWELLS I, MNRAS 417, 1601) Table 1 (decimal J2000 degrees,
# converted to sexagesimal) for the eight Treu-discovered lenses; Brewer et al. 2012
# (SWELLS III, MNRAS 422, 3574) Table 2 for the four Brewer additions.
# J1331+3638 keeps the Treu+2011 ID as its key; the true position (36.46999 deg) is
# +36:28, matching its MAST target name SDSSJ1331+3628 (see lens_samples.json).
swells_coords = {
    "J0820+4847": ["08:20:12.87", "+48:47:37.1"],  # Treu+2011 Table 1: 125.05363, 48.79364
    "J1037+3517": ["10:37:45.03", "+35:17:31.0"],  # Treu+2011 Table 1: 159.43764, 35.29194
    "J1117+4704": ["11:17:35.38", "+47:04:07.4"],  # Treu+2011 Table 1: 169.39742, 47.06873
    "J1203+2535": ["12:03:56.33", "+25:35:49.1"],  # Treu+2011 Table 1: 180.9847, 25.59697
    "J1331+3638": ["13:31:40.32", "+36:28:12.0"],  # Treu+2011 Table 1: 202.918, 36.46999
    "J1703+2451": ["17:03:41.47", "+24:51:40.0"],  # Treu+2011 Table 1: 255.92278, 24.86111
    "J0915+4211": ["09:15:16.29", "+42:11:52.8"],  # Treu+2011 Table 1: 138.81787, 42.198
    "J1135+3720": ["11:35:06.88", "+37:20:23.9"],  # Treu+2011 Table 1: 173.77867, 37.33997
    "J0822+1828": ["08:22:27.47", "+18:28:55.8"],  # Brewer+2012 Table 2: 125.614474, 18.482175
    "J0930+2855": ["09:30:17.41", "+28:55:00.2"],  # Brewer+2012 Table 2: 142.57256, 28.916732
    "J1021+2028": ["10:21:44.42", "+20:28:39.8"],  # Brewer+2012 Table 2: 155.435084, 20.477728
    "J1111+2234": ["11:11:28.02", "+22:34:50.6"],  # Brewer+2012 Table 2: 167.86676, 22.580729
}
