"""Static geographic metadata for the 29 TransCIF regions.

Geolocation powers the astronomy features (solar elevation / clear-sky
envelope) and local-time calendar features introduced with the
fuel-decomposed architecture (TransCIF-FD).  All dataset timestamps are UTC
(verified empirically: US_CISO solar output peaks at 20:00 UTC == 12:00 PST,
UK solar peaks at 11-12 UTC), so local time is reconstructed as
``utc_hour + tz_offset``.

Coordinates are demand-weighted approximate centroids — sufficient for
hour-scale solar geometry where a ~1 degree error shifts the diurnal cycle
by ~4 minutes.
"""

# name -> (lat_deg, lon_deg, utc_offset_hours_standard_time)
REGION_META = {
    # --- AU NEM (AEST, UTC+10; no DST in QLD, minor DST elsewhere ignored) ---
    "QLD1": (-27.0, 153.0, 10.0),
    "NSW1": (-33.9, 151.2, 10.0),
    "VIC1": (-37.8, 145.0, 10.0),
    "SA1":  (-34.9, 138.6, 10.0),
    # --- US EIA-930 (standard-time offsets; DST ignored, <1h effect) ---
    "US_CISO": (36.8, -119.5, -8.0),
    "US_PJM":  (39.8, -77.5, -5.0),
    "US_MISO": (41.5, -93.0, -6.0),
    "US_ERCO": (31.8, -99.3, -6.0),
    "US_ISNE": (42.8, -71.5, -5.0),
    "US_NYIS": (42.9, -75.3, -5.0),
    "US_FPL":  (27.4, -81.2, -5.0),
    "US_BPAT": (45.4, -119.5, -8.0),
    # --- UK DNOs (UTC+0) ---
    "UK_01_North_Scotland":      (57.6, -4.2, 0.0),
    "UK_02_South_Scotland":      (55.6, -3.6, 0.0),
    "UK_03_North_West_England":  (53.8, -2.7, 0.0),
    "UK_04_North_East_England":  (54.9, -1.6, 0.0),
    "UK_05_Yorkshire":           (53.9, -1.2, 0.0),
    "UK_06_North_Wales_Merseyside": (53.3, -3.4, 0.0),
    "UK_07_South_Wales":         (51.6, -3.5, 0.0),
    "UK_08_West_Midlands":       (52.5, -2.0, 0.0),
    "UK_09_East_Midlands":       (52.9, -0.9, 0.0),
    "UK_10_East_England":        (52.4, 0.2, 0.0),
    "UK_11_South_West_England":  (50.8, -3.5, 0.0),
    "UK_12_South_England":       (51.0, -1.3, 0.0),
    "UK_13_London":              (51.5, -0.1, 0.0),
    "UK_14_South_East_England":  (51.1, 0.5, 0.0),
    "UK_15_England":             (52.8, -1.5, 0.0),
    "UK_16_Scotland":            (56.6, -4.2, 0.0),
    "UK_17_Wales":               (52.3, -3.6, 0.0),
    # National aggregate (excluded from LORO experiments but discovered by
    # the loader; present so metadata lookups never silently fall back).
    "UK_18_GB":                  (52.8, -1.8, 0.0),
}

# Fallback for regions without a metadata entry (e.g. future targets).
DEFAULT_META = (45.0, 10.0, 0.0)


def get_region_meta(region_name):
    """Return (lat, lon, utc_offset) for a region, with a safe default."""
    return REGION_META.get(region_name, DEFAULT_META)
