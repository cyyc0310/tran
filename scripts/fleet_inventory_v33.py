"""Pure V33 registered-BA inventory curation and a string-only XLSX reader.

No file paths, network requests, labels, model fitting or output writes live here.
The caller binds archive bytes and exclusively writes the reviewed outputs.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import io
import math
import posixpath
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr
import zipfile


SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
POS_FUELS = ('coal', 'gas', 'oil', 'other')
RELEASE_DATES = {
    2020: '2021-09-09', 2021: '2022-09-22',
    2022: '2023-09-19', 2023: '2024-09-23',
}
SUPPORTED_YEARS = (2020, 2021, 2022, 2023)
OPERABLE_STATUSES = ('OP', 'SB', 'OA', 'OS')
STORAGE_PRIME_MOVERS = frozenset(('BA', 'CE', 'CP', 'FW', 'ES', 'PS'))
PRIME_MOVERS = frozenset('ST GT IC CA CT CS HA HB HK HY BT PV WT WS FC OT'.split())
FUEL_CODES = {
    'coal': frozenset('ANT BIT LIG RC SGC SUB WC'.split()),
    'gas': frozenset(('NG',)),
    'oil': frozenset('DFO JF KER PC PG RFO SGP WO'.split()),
    'nuclear': frozenset(('NUC',)), 'hydro': frozenset(('WAT',)),
    'wind': frozenset(('WND',)), 'solar': frozenset(('SUN',)),
    'other': frozenset('AB BFG BLQ GEO LFG MSW OBG OBL OBS OG OTH PUR SLW TDF WDL WDS WH'.split()),
}
ENERGY_SOURCE_TO_FUEL = {code: fuel for fuel, codes in FUEL_CODES.items() for code in codes}
ENERGY_SOURCE_TO_FUEL['H2'] = 'other'
FUEL_CODES_BY_YEAR = {
    year: frozenset(ENERGY_SOURCE_TO_FUEL) - (frozenset(('H2',)) if year == 2020 else frozenset())
    for year in SUPPORTED_YEARS
}
PLANT_COLUMNS = ('Plant Code', 'Balancing Authority Code')
GENERATOR_COLUMNS = (
    'Plant Code', 'Generator ID', 'Prime Mover', 'Status', 'Nameplate Capacity (MW)',
    *(f'Energy Source {i}' for i in range(1, 7)),
    'Multiple Fuels?', 'Cofire Fuels?', 'Switch Between Oil and Natural Gas?',
)
CAPACITY_COLUMNS = (
    'year', 'region', *(f'capacity_{fuel}_mw' for fuel in FUELS),
    'capacity_pos_mw', 'capacity_total_mw', 'generator_count',
    'zero_capacity_count', 'multifuel_flagged_count', 'anchor_eligible',
)
UNIT_COLUMNS = (
    'year', 'region', 'plant_code', 'generator_id', 'status', 'prime_mover',
    'primary_energy_source', 'fuel', 'capacity_mw', 'secondary_energy_sources',
    'multi_fuels_flag', 'cofire_flag', 'switch_flag', 'in_op',
)
EXCLUDED_COLUMNS = (
    'year', 'region', 'plant_code', 'generator_id', 'status', 'prime_mover',
    'primary_energy_source', 'reason',
)

_NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
_S = '{' + _NS['s'] + '}'
_RID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
_CELL_REFERENCE = re.compile(r'([A-Z]+)([1-9][0-9]*)\Z')
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?\Z')


def _text(row: Mapping[str, str], field: str) -> str:
    value = row.get(field, '')
    if not isinstance(value, str):
        raise ValueError(f'{field} must be an original string')
    return value


def _plant_code(value: str) -> str:
    """Normalize whole positive numeric identifiers without float rounding."""
    text = value.strip()
    if not _NUMBER.fullmatch(text):
        raise ValueError('Plant Code must be a positive integer identifier')
    try:
        number = Decimal(text)
        if not number.is_finite() or number <= 0 or number != number.to_integral_value():
            raise ValueError('Plant Code must be a positive integer identifier')
        return str(int(number))
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError('Plant Code must be a positive integer identifier') from exc


def _capacity(row: Mapping[str, str]) -> float:
    raw = _text(row, 'Nameplate Capacity (MW)').strip()
    if not _NUMBER.fullmatch(raw):
        raise ValueError('Selected nameplate capacity must be a finite nonnegative number')
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Selected nameplate capacity must be finite and nonnegative')
    return value


def _sum_capacity(values: Iterable[float]) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('Aggregate capacity must be finite') from exc
    if not math.isfinite(total):
        raise ValueError('Aggregate capacity must be finite')
    return total


def _aggregate(year: int, region: str, units: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped = {fuel: _sum_capacity(u['capacity_mw'] for u in units if u['fuel'] == fuel) for fuel in FUELS}
    pos = _sum_capacity(grouped[fuel] for fuel in POS_FUELS)
    values = {
        'year': year, 'region': region,
        **{f'capacity_{fuel}_mw': grouped[fuel] for fuel in FUELS},
        'capacity_pos_mw': pos, 'capacity_total_mw': _sum_capacity(grouped.values()),
        'generator_count': len(units),
        'zero_capacity_count': sum(u['capacity_mw'] == 0 for u in units),
        'multifuel_flagged_count': sum(
            bool(u['secondary_energy_sources']) or any(u[k].strip().upper() == 'Y' for k in
                ('multi_fuels_flag', 'cofire_flag', 'switch_flag')) for u in units),
        'anchor_eligible': pos > 0,
    }
    return {name: values[name] for name in CAPACITY_COLUMNS}


def curate_year(
    year: int,
    plant_rows: Iterable[Mapping[str, str]],
    generator_rows: Iterable[Mapping[str, str]],
) -> dict[str, Any]:
    """Curate only the seven registered BAs, validating before capacity access.

    Non-source, unmatched and blank-BA generator rows are counted without reading
    their capacity or classifying their fuel/status. All selected identities are
    checked before any selected status, storage or capacity processing.
    """
    if type(year) is not int or year not in SUPPORTED_YEARS:
        raise ValueError('Unsupported inventory year')
    plant_map: dict[str, str] = {}
    plant_blank = plant_outside = 0
    for row in plant_rows:
        plant = _plant_code(_text(row, 'Plant Code'))
        if plant in plant_map:
            raise ValueError(f'Duplicate plant identifier: {plant}')
        ba = _text(row, 'Balancing Authority Code').strip().upper()
        plant_map[plant] = ba
        plant_blank += not ba
        plant_outside += bool(ba) and ba not in SOURCES

    counts = Counter()
    selected = []
    seen = set()
    for row in generator_rows:
        counts['generator_rows'] += 1
        plant = _plant_code(_text(row, 'Plant Code'))
        if plant not in plant_map:
            counts['unmapped_plant_rows'] += 1
            continue
        ba = plant_map[plant]
        if not ba:
            counts['blank_ba_rows'] += 1
            continue
        if ba not in SOURCES:
            counts['outside_source_rows'] += 1
            continue
        gen = _text(row, 'Generator ID').strip()
        if not gen:
            raise ValueError('Selected Generator ID must be nonempty')
        key = (plant, gen)
        if key in seen:
            raise ValueError(f'Duplicate generator identifier: {plant}/{gen}')
        seen.add(key)
        selected.append((ba, plant, gen, row))

    units, excluded = [], []
    selected_status, selected_energy = Counter(), Counter()
    for region, plant, gen, row in selected:
        status = _text(row, 'Status').strip().upper()
        pm = _text(row, 'Prime Mover').strip().upper()
        energy = _text(row, 'Energy Source 1').strip().upper()
        if status not in OPERABLE_STATUSES:
            raise ValueError(f'Unsupported selected generator status at {plant}/{gen}')
        if pm not in PRIME_MOVERS | STORAGE_PRIME_MOVERS:
            raise ValueError(f'Unknown or planning-only prime mover at {plant}/{gen}')
        if energy not in FUEL_CODES_BY_YEAR[year] | {'MWH'}:
            raise ValueError(f'Unknown or year-ineligible primary energy source at {plant}/{gen}')
        selected_status[status] += 1
        selected_energy[energy] += 1
        metadata = dict(year=year, region=region, plant_code=plant, generator_id=gen,
                        status=status, prime_mover=pm, primary_energy_source=energy)
        if energy == 'MWH' or pm in STORAGE_PRIME_MOVERS:
            reason = ('storage_energy_source_and_prime_mover' if energy == 'MWH' and pm in STORAGE_PRIME_MOVERS
                      else 'storage_energy_source' if energy == 'MWH' else 'storage_prime_mover')
            record = {**metadata, 'reason': reason}
            excluded.append({name: record[name] for name in EXCLUDED_COLUMNS})
            continue
        secondary = []
        for i in range(2, 7):
            code = _text(row, f'Energy Source {i}').strip().upper()
            if code and code != energy and code not in secondary:
                secondary.append(code)
        record = {
            **metadata, 'fuel': ENERGY_SOURCE_TO_FUEL[energy], 'capacity_mw': _capacity(row),
            'secondary_energy_sources': ';'.join(secondary),
            'multi_fuels_flag': _text(row, 'Multiple Fuels?'),
            'cofire_flag': _text(row, 'Cofire Fuels?'),
            'switch_flag': _text(row, 'Switch Between Oil and Natural Gas?'), 'in_op': status == 'OP',
        }
        units.append({name: record[name] for name in UNIT_COLUMNS})

    def order(record: Mapping[str, Any]) -> tuple[int, int, str]:
        return SOURCES.index(record['region']), int(record['plant_code']), record['generator_id']

    units.sort(key=order)
    excluded.sort(key=order)
    op, operable = [], []
    for region in SOURCES:
        sensitive = [u for u in units if u['region'] == region]
        operating = [u for u in sensitive if u['in_op']]
        main = _aggregate(year, region, operating)
        if not operating or main['capacity_total_mw'] <= 0:
            raise ValueError(f'{region} requires a positive nonstorage OP inventory in {year}')
        op.append(main)
        operable.append(_aggregate(year, region, sensitive))

    audit = {
        'year': year, 'scope': 'registered_ba_inventory_proxy',
        'capacity_measure': 'nameplate_mw', 'coverage_certified': False,
        'historical_vintage_certified': False, 'observed_zero_is_physical_absence': False,
        'hard_output_bound': False, 'secondary_fuels_duplicate_capacity': False,
        'plant_rows': len(plant_map), 'plant_blank_ba_rows': int(plant_blank),
        'plant_outside_source_rows': int(plant_outside),
        **{key: counts[key] for key in ('generator_rows', 'unmapped_plant_rows', 'blank_ba_rows', 'outside_source_rows')},
        'selected_rows': len(selected), 'excluded_storage_rows': len(excluded), 'curated_rows': len(units),
        'selected_by_status': {s: selected_status[s] for s in OPERABLE_STATUSES},
        'selected_by_primary_energy_source': dict(sorted(selected_energy.items())),
        'excluded_storage_by_status': {s: sum(u['status'] == s for u in excluded) for s in OPERABLE_STATUSES},
        'curated_by_status': {s: sum(u['status'] == s for u in units) for s in OPERABLE_STATUSES},
        'curated_by_fuel': {f: sum(u['fuel'] == f for u in units) for f in FUELS},
        'op_by_fuel': {f: sum(u['fuel'] == f and u['in_op'] for u in units) for f in FUELS},
    }
    return dict(op=op, operable=operable, units=units, excluded=excluded, audit=audit)


def inventory_for_origin(origin: str | date | datetime) -> dict[str, Any]:
    """Return the unique origin-year-minus-two inventory, with a release guard.

    A calendar date means UTC midnight. A datetime must carry a timezone and is
    converted to UTC before choosing the calendar year. There is no nearest-year
    or future-release fallback.
    """
    try:
        if isinstance(origin, str):
            value = (datetime.combine(date.fromisoformat(origin), datetime.min.time(), timezone.utc)
                     if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', origin)
                     else datetime.fromisoformat(origin))
        elif isinstance(origin, datetime):
            value = origin
        elif isinstance(origin, date):
            value = datetime.combine(origin, datetime.min.time(), timezone.utc)
        else:
            raise ValueError('Origin must be an ISO calendar date or timezone-aware datetime')
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('Datetime origin requires an explicit timezone')
        value = value.astimezone(timezone.utc)
        year = value.year - 2
        if year not in SUPPORTED_YEARS or year not in RELEASE_DATES:
            raise ValueError('Exact origin-year-minus-two inventory year is unavailable')
        release = datetime.combine(date.fromisoformat(RELEASE_DATES[year]), datetime.min.time(), timezone.utc)
        if release > value:
            raise ValueError('Inventory release is later than origin')
        return {'year': year, 'available_at': release.isoformat()}
    except (TypeError, OverflowError) as exc:
        raise ValueError('Invalid inventory origin or release date') from exc


class _SharedStrings:
    """Index shared-string XML as bytes, decoding only requested cell entries."""

    def __init__(self, package: zipfile.ZipFile):
        self.package = package
        self.raw: bytes | None = None
        self.ranges: list[tuple[int, int]] = []
        self.wrapper = b''
        self.cache: dict[int, str] = {}

    def get(self, index: str) -> str:
        if not re.fullmatch(r'[0-9]+', index):
            raise ValueError('Invalid shared-string index')
        number = int(index)
        if self.raw is None:
            self.raw = self.package.read('xl/sharedStrings.xml')
            namespaces = {}
            for event, item in ET.iterparse(io.BytesIO(self.raw), events=('start-ns', 'start')):
                if event == 'start':
                    break
                prefix, uri = item
                namespaces[prefix] = uri
            declarations = ' '.join(('xmlns' + (':' + prefix if prefix else '') + '=' + quoteattr(uri))
                                    for prefix, uri in namespaces.items())
            self.wrapper = ('<lookup ' + declarations + '>').encode('utf-8')
            pattern = rb'<(?:[A-Za-z_][\w.-]*:)?si\b[^>]*(?:/>|>.*?</(?:[A-Za-z_][\w.-]*:)?si\s*>)'
            self.ranges = [(m.start(), m.end()) for m in re.finditer(pattern, self.raw, re.S)]
        if number >= len(self.ranges):
            raise ValueError('Shared-string index is outside the dictionary')
        if number not in self.cache:
            start, end = self.ranges[number]
            root = ET.fromstring(self.wrapper + self.raw[start:end] + b'</lookup>')
            self.cache[number] = _display_text(root[0])
        return self.cache[number]


def _display_text(container: ET.Element) -> str:
    """Concatenate plain/rich display runs, excluding phonetic annotations."""
    parts = []
    for child in container:
        kind = child.tag.rsplit('}', 1)[-1]
        if kind == 't':
            parts.append(child.text or '')
        elif kind == 'r':
            parts.extend(t.text or '' for t in child if t.tag.rsplit('}', 1)[-1] == 't')
    return ''.join(parts)


def _cell_text(cell: ET.Element, strings: _SharedStrings) -> str:
    if cell.find(_S + 'f') is not None:
        raise ValueError('Selected/header formula cells are not accepted')
    kind = cell.attrib.get('t', 'n')
    if kind == 'inlineStr':
        inline = cell.find(_S + 'is')
        return _display_text(inline) if inline is not None else ''
    value = cell.find(_S + 'v')
    raw = value.text or '' if value is not None else ''
    if kind == 's':
        return strings.get(raw)
    if kind not in ('n', 'str', 'b', 'd'):
        raise ValueError('Unsupported/error selected cell type')
    return raw


def _sheet_path(package: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(package.read('xl/workbook.xml'))
    sheets = [s for s in workbook.findall('s:sheets/s:sheet', _NS) if s.attrib.get('name') == sheet_name]
    if len(sheets) != 1:
        raise ValueError('Requested worksheet is missing or duplicated')
    relationships = ET.fromstring(package.read('xl/_rels/workbook.xml.rels'))
    links = [r for r in relationships if r.attrib.get('Id') == sheets[0].attrib.get(_RID)]
    if len(links) != 1 or links[0].attrib.get('TargetMode') == 'External':
        raise ValueError('Worksheet relationship is missing, duplicated or external')
    target = links[0].attrib.get('Target', '')
    if not target or ':' in target or '\\' in target:
        raise ValueError('Invalid worksheet relationship path')
    path = posixpath.normpath(target.lstrip('/') if target.startswith('/') else 'xl/' + target)
    if not path.startswith('xl/'):
        raise ValueError('Worksheet relationship escapes workbook directory')
    return path


def read_workbook_rows(
    xlsx_bytes: bytes, sheet_name: str, columns: Sequence[str],
) -> Iterator[dict[str, str]]:
    """Yield only budget-approved columns as original strings; never float cells.

    EIA title row 1 is ignored and row 2 supplies the required unique headers.
    Sparse cells become empty strings. Unrequested data cells are not decoded,
    including their shared-string entries or formulas. Selected formulas/errors
    are rejected instead of using cached results. No links or macros execute.
    """
    columns = tuple(columns)
    if (not columns or len(set(columns)) != len(columns)
            or not set(columns) <= set(PLANT_COLUMNS + GENERATOR_COLUMNS)):
        raise ValueError('Requested columns must be a nonempty unique budget whitelist')
    if sheet_name not in ('Plant', 'Operable'):
        raise ValueError('Only Plant and Operable worksheets are in scope')
    try:
        with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as package:
            names = package.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate XLSX package members')
            strings = _SharedStrings(package)
            path = _sheet_path(package, sheet_name)
            header = None
            previous_row = 0
            with package.open(path) as stream:
                root = None
                for event, element in ET.iterparse(stream, events=('start', 'end')):
                    if root is None:
                        root = element
                        if root.tag != _S + 'worksheet':
                            raise ValueError('Worksheet relationship does not target a worksheet')
                    if event != 'end' or element.tag != _S + 'row':
                        continue
                    row_text = element.attrib.get('r', '')
                    if not re.fullmatch(r'[1-9][0-9]*', row_text):
                        raise ValueError('Worksheet row requires a positive index')
                    row_number = int(row_text)
                    if row_number <= previous_row:
                        raise ValueError('Duplicate or unordered worksheet row index')
                    previous_row = row_number
                    cells = {}
                    for cell in element.findall('s:c', _NS):
                        match = _CELL_REFERENCE.fullmatch(cell.attrib.get('r', ''))
                        if match is None or int(match[2]) != row_number or match[1] in cells:
                            raise ValueError('Invalid or duplicate worksheet cell reference')
                        cells[match[1]] = cell
                    result = None
                    if row_number == 2:
                        header = {}
                        for coordinate, cell in cells.items():
                            label = _cell_text(cell, strings).strip()
                            if not label:
                                continue
                            if label in header:
                                raise ValueError('Duplicate worksheet header')
                            header[label] = coordinate
                        if not set(columns) <= set(header):
                            raise ValueError('Required worksheet header is missing')
                    elif row_number > 2:
                        if header is None:
                            raise ValueError('Worksheet header row 2 is missing')
                        result = {column: (_cell_text(cells[header[column]], strings)
                                           if header[column] in cells else '') for column in columns}
                    element.clear()
                    root.clear()
                    if result is not None and any(value.strip() for value in result.values()):
                        yield result
            if header is None:
                raise ValueError('Worksheet header row 2 is missing')
    except (ET.ParseError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError('Invalid XLSX package, dictionary or XML structure') from exc
