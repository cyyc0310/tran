"""Recognize evidenced transport truncation without weakening GFS validators."""
from __future__ import annotations

import hashlib

from transcif.data import gfs


def truncated_response(error: Exception) -> dict | None:
    """Inspect an already rejected original response; never alter requests or data."""
    if not isinstance(error, ValueError) or str(error) != 'GFS range length/version evidence mismatch':
        return None
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code is gfs.collect_gfs.__code__:
            values = frame.f_locals
            required = ['part_headers', 'grib_evidence', 'content', 'start', 'end', 'size', 'length', 'run', 'lead', 'name']
            if any(name not in values for name in required):
                return None
            headers, expected, body = values['part_headers'], values['grib_evidence'], values['content']
            length = values['length']
            expected_range = f"bytes {values['start']}-{values['end']}/{values['size']}"
            if (not isinstance(body, bytes) or not 16 <= len(body) < length
                    or headers.get('content-range') != expected_range
                    or headers.get('etag') != expected.get('etag')
                    or headers.get('content-length') != str(length)
                    or body[:4] != b'GRIB' or body[7] != 2
                    or int.from_bytes(body[8:16], 'big') != length):
                return None
            return dict(kind='verified_header_short_read', run=str(values['run']), lead=values['lead'],
                field=values['name'], expected_bytes=length, received_bytes=len(body),
                expected_range=expected_range, expected_etag=expected['etag'], response_headers=headers,
                partial_body_sha256=hashlib.sha256(body).hexdigest(),
                partial_snapshot_accepted=False, validator_modified=False)
        traceback = traceback.tb_next
    return None
