"""Synthetic inventory, XLSX and access-boundary tests; no real data files."""
from copy import deepcopy
from datetime import date, datetime, timezone, timedelta
import io
import json
import math
import zipfile
from xml.sax.saxutils import escape

import pytest

from scripts import fleet_inventory_v33 as m


REGIONS = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
FUEL_MAP = {
    'coal': 'ANT BIT LIG RC SGC SUB WC', 'gas': 'NG',
    'oil': 'DFO JF KER PC PG RFO SGP WO', 'nuclear': 'NUC',
    'hydro': 'WAT', 'wind': 'WND', 'solar': 'SUN',
    'other': 'AB BFG BLQ GEO LFG MSW OBG OBL OBS OG OTH PUR SLW TDF WDL WDS WH',
}


def generator(plant='1', gen='001', fuel='NG', capacity='10', status='OP', pm='GT', **extra):
    row = {
        'Plant Code': plant, 'Generator ID': gen, 'Prime Mover': pm,
        'Status': status, 'Nameplate Capacity (MW)': capacity,
        **{f'Energy Source {i}': fuel if i == 1 else '' for i in range(1, 7)},
        'Multiple Fuels?': '', 'Cofire Fuels?': '',
        'Switch Between Oil and Natural Gas?': '',
    }
    row.update(extra)
    return row


def fixture_rows():
    plants = [{'Plant Code': str(i), 'Balancing Authority Code': region}
              for i, region in enumerate(REGIONS, 1)]
    generators = [generator(str(i)) for i in range(1, 8)]
    return plants, generators


class ForbiddenCapacity:
    def __float__(self):
        raise AssertionError('forbidden capacity converted')

    def __str__(self):
        raise AssertionError('forbidden capacity decoded')


def test_contract_constants_and_output_schemas():
    assert m.SOURCES == REGIONS
    assert m.FUELS == tuple(FUEL_MAP)
    assert m.RELEASE_DATES == {
        2020: '2021-09-09', 2021: '2022-09-22',
        2022: '2023-09-19', 2023: '2024-09-23',
    }
    plants, gens = fixture_rows()
    gens.append(generator(gen='storage', fuel='MWH', pm='BA', capacity=ForbiddenCapacity()))
    result = m.curate_year(2020, plants, gens)
    assert set(result) == {'op', 'operable', 'units', 'excluded', 'audit'}
    for key, columns in [('op', m.CAPACITY_COLUMNS), ('operable', m.CAPACITY_COLUMNS),
                         ('units', m.UNIT_COLUMNS), ('excluded', m.EXCLUDED_COLUMNS)]:
        assert isinstance(columns, tuple)
        assert len(columns) == len(set(columns))
        assert all(tuple(record) == columns for record in result[key])
    assert not any('capacity' in name for name in m.EXCLUDED_COLUMNS)
    json.dumps(result, allow_nan=False)


def test_scope_filter_precedes_any_capacity_access_and_audit_reconciles():
    plants, gens = fixture_rows()
    plants.extend([{'Plant Code': '8', 'Balancing Authority Code': 'OUTSIDE'},
                   {'Plant Code': '9', 'Balancing Authority Code': '   '}])
    gens.extend(generator(str(i), capacity=ForbiddenCapacity(), fuel='UNKNOWN', pm='UNKNOWN', status='RE')
                for i in (8, 9, 999))
    result = m.curate_year(2020, plants, gens)
    audit = result['audit']
    assert audit['plant_rows'] == 9 and audit['generator_rows'] == 10
    assert audit['outside_source_rows'] == audit['blank_ba_rows'] == audit['unmapped_plant_rows'] == 1
    assert audit['selected_rows'] == audit['curated_rows'] == 7
    assert audit['excluded_storage_rows'] == 0
    assert audit['scope'] == 'registered_ba_inventory_proxy'
    assert audit['coverage_certified'] is False
    assert audit['observed_zero_is_physical_absence'] is False
    assert audit['generator_rows'] == sum(audit[k] for k in
        ('outside_source_rows', 'blank_ba_rows', 'unmapped_plant_rows', 'selected_rows'))
    assert len(result['units']) == 7


@pytest.mark.parametrize('code,fuel', [(code, fuel) for fuel, codes in FUEL_MAP.items() for code in codes.split()])
@pytest.mark.parametrize('year', [2020, 2021, 2022, 2023])
def test_official_primary_codes_map_once(code, fuel, year):
    plants, gens = fixture_rows()
    gens.append(generator(gen='extra', fuel=code, capacity='2.5'))
    result = m.curate_year(year, plants, gens)
    unit = next(x for x in result['units'] if x['generator_id'] == 'extra')
    assert unit['fuel'] == fuel and unit['capacity_mw'] == 2.5
    row = result['op'][0]
    assert row[f'capacity_{fuel}_mw'] == 2.5 + (10 if fuel == 'gas' else 0)
    assert row['capacity_total_mw'] == 12.5


@pytest.mark.parametrize('year', [2021, 2022, 2023])
def test_hydrogen_only_present_from_2021(year):
    plants, gens = fixture_rows()
    gens.append(generator(gen='h2', fuel='H2'))
    assert m.curate_year(year, plants, gens)['op'][0]['capacity_other_mw'] == 10
    with pytest.raises(ValueError, match='energy source'):
        m.curate_year(2020, plants, gens)


@pytest.mark.parametrize('pm,fuel', [('BA', 'NG'), ('CE', 'NG'), ('CP', 'SUN'),
                                    ('FW', 'NG'), ('ES', 'NG'), ('PS', 'WAT'), ('GT', 'MWH')])
def test_storage_exclusion_precedes_capacity_conversion_and_fuel_aggregation(pm, fuel):
    plants, gens = fixture_rows()
    gens.append(generator(gen='storage', fuel=fuel, pm=pm, capacity=ForbiddenCapacity()))
    result = m.curate_year(2020, plants, gens)
    assert len(result['units']) == 7 and len(result['excluded']) == 1
    assert result['excluded'][0]['generator_id'] == 'storage'
    assert result['op'][0]['capacity_total_mw'] == 10
    assert result['audit']['selected_rows'] == 8
    assert result['audit']['excluded_storage_rows'] == 1


@pytest.mark.parametrize('pm', 'ST GT IC CA CT CS HA HB HK HY BT PV WT WS FC OT'.split())
def test_every_permitted_nonstorage_prime_mover(pm):
    plants, gens = fixture_rows()
    gens[0]['Prime Mover'] = pm
    assert m.curate_year(2020, plants, gens)['units'][0]['prime_mover'] == pm


@pytest.mark.parametrize('field,value', [
    ('Status', 'RE'), ('Status', ''), ('Status', 'UNKNOWN'),
    ('Prime Mover', 'CC'), ('Prime Mover', ''), ('Prime Mover', 'UNKNOWN'),
    ('Energy Source 1', ''), ('Energy Source 1', 'UNKNOWN'),
])
def test_selected_unknown_or_disallowed_codes_fail(field, value):
    plants, gens = fixture_rows()
    gens[0][field] = value
    with pytest.raises(ValueError):
        m.curate_year(2020, plants, gens)


@pytest.mark.parametrize('field,value', [('Status', 'RE'), ('Prime Mover', 'UNKNOWN'), ('Energy Source 1', 'UNKNOWN')])
def test_storage_does_not_hide_invalid_selected_metadata(field, value):
    plants, gens = fixture_rows()
    gens.append(generator(gen='storage', fuel='MWH', pm='BA', capacity=ForbiddenCapacity()))
    gens[-1][field] = value
    with pytest.raises(ValueError):
        m.curate_year(2020, plants, gens)


@pytest.mark.parametrize('capacity', ['', ' ', 'missing', '1,000', '1_000', '-0.1', 'NaN', 'inf', '-inf', '1e9999'])
def test_selected_invalid_capacities_fail(capacity):
    plants, gens = fixture_rows()
    gens.append(generator(gen='bad', capacity=capacity))
    with pytest.raises(ValueError, match='capacity'):
        m.curate_year(2020, plants, gens)


def test_zero_capacity_is_kept_and_counted_without_epsilon():
    plants, gens = fixture_rows()
    gens.append(generator(gen='zero', fuel='BIT', capacity='0'))
    result = m.curate_year(2020, plants, gens)
    assert len(result['units']) == 8
    row = result['op'][0]
    assert row['zero_capacity_count'] == 1 and row['generator_count'] == 2
    assert row['capacity_coal_mw'] == 0 and row['capacity_pos_mw'] == 10
    assert row['anchor_eligible'] is True
    gens[0].update({'Energy Source 1': 'NUC'})
    row = m.curate_year(2020, plants, gens)['op'][0]
    assert row['capacity_pos_mw'] == 0 and row['capacity_total_mw'] == 10
    assert row['anchor_eligible'] is False


def test_op_sensitivity_secondary_flags_and_capacity_reconciliation():
    plants, gens = fixture_rows()
    gens[0].update({'Energy Source 2': 'NG', 'Energy Source 3': ' RFO ',
                    'Energy Source 4': 'RFO', 'Energy Source 5': 'NG',
                    'Multiple Fuels?': 'unknown', 'Cofire Fuels?': ' n ',
                    'Switch Between Oil and Natural Gas?': 'no'})
    for status in ('SB', 'OA', 'OS'):
        gens.append(generator(gen=status, fuel='BIT', status=status, capacity='2', **{'Multiple Fuels?': ' y '}))
    result = m.curate_year(2020, plants, gens)
    op, sensitive = result['op'][0], result['operable'][0]
    assert op['capacity_total_mw'] == 10 and op['generator_count'] == 1
    assert sensitive['capacity_total_mw'] == 16 and sensitive['generator_count'] == 4
    assert op['multifuel_flagged_count'] == 1 and sensitive['multifuel_flagged_count'] == 4
    assert len(result['units']) == 10
    first = next(u for u in result['units'] if u['plant_code'] == '1' and u['generator_id'] == '001')
    assert first['secondary_energy_sources'] == 'RFO'
    assert first['multi_fuels_flag'] == 'unknown' and first['cofire_flag'] == ' n '
    assert sum(u['in_op'] for u in result['units']) == 7
    for key in ('op', 'operable'):
        for row in result[key]:
            selected = [u for u in result['units'] if u['region'] == row['region'] and (key == 'operable' or u['in_op'])]
            assert row['capacity_total_mw'] == math.fsum(u['capacity_mw'] for u in selected)
            assert row['capacity_total_mw'] == math.fsum(row[f'capacity_{f}_mw'] for f in m.FUELS)
            assert row['capacity_pos_mw'] == math.fsum(row[f'capacity_{f}_mw'] for f in ('coal', 'gas', 'oil', 'other'))
    assert result['audit']['curated_by_status'] == {'OP': 7, 'SB': 1, 'OA': 1, 'OS': 1}
    assert result['audit']['curated_by_fuel']['coal'] == 3


def test_only_y_flags_or_distinct_nonempty_secondary_set_multifuel_count():
    plants, gens = fixture_rows()
    gens[0].update({'Energy Source 2': 'NG', 'Energy Source 3': 'NG', 'Multiple Fuels?': 'true'})
    assert m.curate_year(2020, plants, gens)['op'][0]['multifuel_flagged_count'] == 0
    gens[0]['Cofire Fuels?'] = ' y '
    assert m.curate_year(2020, plants, gens)['op'][0]['multifuel_flagged_count'] == 1


@pytest.mark.parametrize('newcode', ['1', '001', '1.0', '1e0'])
def test_duplicate_plant_ids_rejected_even_with_same_ba(newcode):
    plants, gens = fixture_rows()
    plants.append({'Plant Code': newcode, 'Balancing Authority Code': 'BPAT'})
    with pytest.raises(ValueError, match='Duplicate plant'):
        m.curate_year(2020, plants, gens)


@pytest.mark.parametrize('pm,status', [('GT', 'OP'), ('BA', 'SB'), ('CC', 'RE')])
def test_duplicate_selected_generators_rejected_before_status_storage_checks(pm, status):
    plants, gens = fixture_rows()
    gens.append(generator(plant='01.0', gen=' 001 ', pm=pm, status=status, capacity=ForbiddenCapacity()))
    with pytest.raises(ValueError, match='Duplicate generator'):
        m.curate_year(2020, plants, gens)


def test_generator_id_preserves_leading_zero_and_case():
    plants, gens = fixture_rows()
    gens.extend([generator(gen='1'), generator(gen='a'), generator(gen='A')])
    assert {u['generator_id'] for u in m.curate_year(2020, plants, gens)['units'] if u['region'] == 'BPAT'} == {'001', '1', 'a', 'A'}


@pytest.mark.parametrize('code', ['', '0', '-1', '1.2', 'NaN', 'INF', 'not-id'])
def test_bad_plant_identifier_fails(code):
    plants, gens = fixture_rows()
    plants[0]['Plant Code'] = code
    with pytest.raises(ValueError, match='Plant Code'):
        m.curate_year(2020, plants, gens)


@pytest.mark.parametrize('mode', ['missing', 'only-zero-op', 'only-standby', 'only-storage'])
def test_each_source_requires_positive_nonstorage_op(mode):
    plants, gens = fixture_rows()
    if mode == 'missing':
        gens.pop()
    elif mode == 'only-zero-op':
        gens[-1]['Nameplate Capacity (MW)'] = '0'
    elif mode == 'only-standby':
        gens[-1]['Status'] = 'SB'
    else:
        gens[-1].update({'Prime Mover': 'BA', 'Energy Source 1': 'MWH'})
    with pytest.raises(ValueError, match='NYIS'):
        m.curate_year(2020, plants, gens)


def test_row_order_invariance_and_input_immutability():
    plants, gens = fixture_rows()
    gens.extend([generator(gen='B', fuel='BIT', capacity='.1'), generator(gen='A', fuel='BIT', capacity='.2')])
    before = deepcopy((plants, gens))
    result = m.curate_year(2020, iter(plants), iter(gens))
    assert (plants, gens) == before
    assert result == m.curate_year(2020, reversed(plants), reversed(gens))


@pytest.mark.parametrize('origin,year', [('2022-01-01', 2020), ('2023-01-01', 2021),
                                       ('2024-12-31', 2022), ('2025-07-01', 2023),
                                       (date(2025, 1, 1), 2023),
                                       (datetime(2025, 1, 1, tzinfo=timezone.utc), 2023)])
def test_inventory_asof_exact_year_and_release(origin, year):
    result = m.inventory_for_origin(origin)
    assert result == {'year': year, 'available_at': m.RELEASE_DATES[year] + 'T00:00:00+00:00'}


@pytest.mark.parametrize('origin', ['2021-12-31', '2026-01-01', 'bad', '2025-01-01T00:00:00', datetime(2025, 1, 1)])
def test_inventory_missing_year_bad_or_naive_origin_rejected(origin):
    with pytest.raises(ValueError):
        m.inventory_for_origin(origin)


def test_origin_timezone_is_normalized_before_calendar_year_selection():
    origin = datetime(2025, 1, 1, 1, tzinfo=timezone(timedelta(hours=8)))
    assert m.inventory_for_origin(origin)['year'] == 2022


def test_future_release_is_rejected_without_fallback(monkeypatch):
    monkeypatch.setattr(m, 'RELEASE_DATES', {**m.RELEASE_DATES, 2023: '2025-02-01'})
    with pytest.raises(ValueError, match='release'):
        m.inventory_for_origin('2025-01-01')


def xlsx(rows, shared=(), sheet='Operable', target='worksheets/sheet1.xml', external=False):
    """Build a tiny synthetic OpenXML package, never an EIA archive."""
    main = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    package = 'http://schemas.openxmlformats.org/package/2006/relationships'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets><sheet name="{sheet}" sheetId="1" r:id="s1"/></sheets></workbook>')
        mode = ' TargetMode="External"' if external else ''
        z.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{package}"><Relationship Id="s1" Target="{escape(target)}"{mode}/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{main}"><sheetData>{rows}</sheetData></worksheet>')
        z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{main}">' + ''.join('<si><t>' + escape(v) + '</t></si>' for v in shared) + '</sst>')
    return stream.getvalue()


def inline(ref, value):
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def test_xlsx_sparse_shared_inline_raw_strings_and_selected_columns_only():
    headers = '<row r="1">' + inline('A1', 'synthetic title') + '</row><row r="2">'
    headers += '<c r="A2" t="s"><v>0</v></c>' + inline('C2', 'Generator ID')
    headers += inline('F2', 'Nameplate Capacity (MW)') + inline('H2', 'Not requested') + '</row>'
    rows = headers + '<row r="3"><c r="A3"><v>1.0</v></c>'
    rows += '<c r="C3" t="inlineStr"><is><r><t> 00</t></r><r><t>1 </t></r></is></c>'
    rows += '<c r="F3" t="s"><v>1</v></c><c r="H3" t="s"><v>99999</v></c></row>'
    rows += '<row r="4"><c r="A4"><v>2</v></c>' + inline('C4', 'A') + '</row>'
    cols = ('Plant Code', 'Generator ID', 'Nameplate Capacity (MW)')
    actual = list(m.read_workbook_rows(xlsx(rows, ['Plant Code', 'DO_NOT_FLOAT']), 'Operable', cols))
    assert actual == [{'Plant Code': '1.0', 'Generator ID': ' 001 ', 'Nameplate Capacity (MW)': 'DO_NOT_FLOAT'},
                      {'Plant Code': '2', 'Generator ID': 'A', 'Nameplate Capacity (MW)': ''}]
    assert all(type(value) is str for row in actual for value in row.values())


@pytest.mark.parametrize('header', [inline('A2', 'Plant Code') + inline('B2', ' Plant Code '), inline('A2', 'Other')])
def test_xlsx_duplicate_or_missing_required_headers_rejected(header):
    with pytest.raises(ValueError, match='header'):
        list(m.read_workbook_rows(xlsx('<row r="2">' + header + '</row>'), 'Operable', ('Plant Code',)))


def test_xlsx_no_formula_execution_or_cached_formula_value_acceptance():
    rows = '<row r="2">' + inline('A2', 'Plant Code') + '</row>'
    rows += '<row r="3"><c r="A3"><f>1+1</f><v>2</v></c></row>'
    with pytest.raises(ValueError, match='formula'):
        list(m.read_workbook_rows(xlsx(rows), 'Operable', ('Plant Code',)))


def test_xlsx_unrequested_formula_is_ignored_and_zero_raw_value_kept():
    rows = '<row r="2">' + inline('A2', 'Plant Code') + inline('B2', 'Unrequested') + '</row>'
    rows += '<row r="3"><c r="A3"><v>0</v></c><c r="B3"><f>1/0</f><v>ERROR</v></c></row>'
    assert list(m.read_workbook_rows(xlsx(rows), 'Operable', ('Plant Code',))) == [{'Plant Code': '0'}]


def test_xlsx_rich_inline_phonetic_annotations_are_not_displayed_data():
    rows = '<row r="2">' + inline('A2', 'Generator ID') + '</row>'
    rows += ('<row r="3"><c r="A3" t="inlineStr"><is><r><t>A</t></r>'
             '<r><t>01</t></r><rPh sb="0" eb="1"><t>phonetic</t></rPh></is></c></row>')
    assert list(m.read_workbook_rows(xlsx(rows), 'Operable', ('Generator ID',))) == [{'Generator ID': 'A01'}]


def test_xlsx_many_sparse_rows_do_not_lose_values_after_xml_clearing():
    rows = '<row r="2">' + inline('C2', 'Generator ID') + '</row>'
    rows += ''.join(f'<row r="{i}">' + inline(f'C{i}', f'ID-{i}') + '</row>' for i in range(3, 503))
    actual = list(m.read_workbook_rows(xlsx(rows), 'Operable', ('Generator ID',)))
    assert actual == [{'Generator ID': f'ID-{i}'} for i in range(3, 503)]


@pytest.mark.parametrize('columns', [(), ('Plant Code', 'Plant Code'), ('Summer Capacity (MW)',)])
def test_xlsx_reader_requires_nonempty_unique_budget_whitelist(columns):
    with pytest.raises(ValueError):
        list(m.read_workbook_rows(xlsx('<row r="2">' + inline('A2', 'Plant Code') + '</row>'), 'Operable', columns))


def test_xlsx_missing_sheet_and_external_or_escaping_relationship_rejected():
    rows = '<row r="2">' + inline('A2', 'Plant Code') + '</row>'
    for data, name in [(xlsx(rows), 'Plant'), (xlsx(rows, target='https://example.invalid', external=True), 'Operable'),
                       (xlsx(rows, target='../../outside.xml'), 'Operable')]:
        with pytest.raises(ValueError):
            list(m.read_workbook_rows(data, name, ('Plant Code',)))


def test_xlsx_duplicate_cells_or_error_values_rejected():
    header = '<row r="2">' + inline('A2', 'Plant Code') + '</row>'
    for data in ['<row r="3">' + inline('A3', '1') + inline('A3', '2') + '</row>',
                 '<row r="3"><c r="A3" t="e"><v>#VALUE!</v></c></row>']:
        with pytest.raises(ValueError):
            list(m.read_workbook_rows(xlsx(header + data), 'Operable', ('Plant Code',)))


def test_unsupported_year_and_aggregate_overflow_fail():
    plants, gens = fixture_rows()
    with pytest.raises(ValueError, match='year'):
        m.curate_year(2019, plants, gens)
    gens[0]['Nameplate Capacity (MW)'] = '1e308'
    gens.append(generator(gen='overflow', capacity='1e308'))
    with pytest.raises(ValueError, match='capacity'):
        m.curate_year(2020, plants, gens)
