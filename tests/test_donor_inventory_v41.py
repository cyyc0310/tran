"""Synthetic whitelist, V33 equivalence and prohibited-field access tests."""
from copy import deepcopy

import pytest

from scripts import donor_inventory_v41 as m
from scripts import fleet_inventory_v33 as old


def generator(plant='1', gen='001', fuel='NG', capacity='10', status='OP', pm='GT', **extra):
    row = dict(zip(old.GENERATOR_COLUMNS, ['']*len(old.GENERATOR_COLUMNS)))
    row.update({'Plant Code': plant, 'Generator ID': gen, 'Energy Source 1': fuel,
                'Nameplate Capacity (MW)': capacity, 'Status': status, 'Prime Mover': pm})
    row.update(extra)
    return row


def fixture(sources=None):
    sources = m.ALL_SOURCES if sources is None else sources
    plants = [{'Plant Code': str(i), 'Balancing Authority Code': source}
              for i, source in enumerate(sources, 1)]
    gens = [generator(str(i)) for i in range(1, len(sources)+1)]
    return plants, gens


class ForbiddenFields(dict):
    def __init__(self, row, allowed):
        super().__init__(row); self.allowed = set(allowed)

    def get(self, key, default=None):
        if key not in self.allowed: raise AssertionError(f'forbidden field access: {key}')
        return super().get(key, default)


def test_constants_and_direct_reuse_without_legacy_mutation():
    assert m.LEGACY_SOURCES == old.SOURCES
    assert m.DONOR_SOURCES == ('SWPP', 'AVA', 'BANC', 'JEA', 'SC', 'NWMT', 'SCL', 'WACM', 'SEC', 'SCEG')
    assert m.ALL_SOURCES == old.SOURCES+m.DONOR_SOURCES
    for name in ('read_workbook_rows', 'inventory_for_origin', 'PLANT_COLUMNS', 'GENERATOR_COLUMNS',
                 'CAPACITY_COLUMNS', 'UNIT_COLUMNS', 'EXCLUDED_COLUMNS'):
        assert getattr(m, name) is getattr(old, name)
    assert old.SOURCES == ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')


@pytest.mark.parametrize('year', [2020, 2021, 2022, 2023])
def test_legacy_mode_entire_result_equals_sealed_v33(year):
    plants, gens = fixture()
    gens += [generator('1', 'standby', 'BIT', '2', 'SB'),
             generator('1', 'storage', 'MWH', 'not-a-capacity', pm='BA'),
             generator('2', 'zero', 'DFO', '0', **{'Energy Source 2': 'NG', 'Multiple Fuels?': ' Y '})]
    expected = old.curate_year(year, plants, gens)
    assert m.curate_year(year, plants, gens, sources=m.LEGACY_SOURCES) == expected


def test_all_seventeen_and_legacy_subsets_match_all_four_tables():
    plants, gens = fixture(); original = deepcopy((plants, gens))
    for i in range(1, 18):
        gens.append(generator(str(i), 'sensitive', 'RFO', '2', 'OS', **{'Energy Source 2': 'NG'}))
        gens.append(generator(str(i), 'storage', 'MWH', 'poison', pm='BA'))
    before = deepcopy((plants, gens))
    result = m.curate_year(2023, plants, gens)
    legacy = old.curate_year(2023, plants, gens)
    for key in ('op', 'operable', 'units', 'excluded'):
        assert [r for r in result[key] if r['region'] in old.SOURCES] == legacy[key]
    assert [r['region'] for r in result['op']] == list(m.ALL_SOURCES)
    assert all(r['capacity_gas_mw'] == 10 for r in result['op'])
    assert all(r['capacity_oil_mw'] == 2 for r in result['operable'])
    assert (plants, gens) == before and len(original[0]) == 17
    assert old.SOURCES == m.LEGACY_SOURCES


@pytest.mark.parametrize('bad', [(), ('PACE',), ('LGEE',), tuple(reversed(old.SOURCES)),
    old.SOURCES+('SWPP',), list(old.SOURCES), set(old.SOURCES), old.SOURCES+('PACE',)])
def test_other_whitelists_fail_before_row_iteration(bad):
    class NoRows:
        def __iter__(self): raise AssertionError('row accessed before whitelist guard')
    with pytest.raises(ValueError, match='sources'):
        m.curate_year(2020, NoRows(), NoRows(), sources=bad)


@pytest.mark.parametrize('year', [2019, 2024, '2020', True])
def test_invalid_year_precedes_row_iteration(year):
    class NoRows:
        def __iter__(self): raise AssertionError('row accessed before year guard')
    with pytest.raises(ValueError, match='year'):
        m.curate_year(year, NoRows(), NoRows())


def test_outside_blank_unmapped_rows_never_access_generator_class_or_capacity():
    plants, gens = fixture()
    for plant, ba in [('101', 'PACE'), ('102', 'LGEE'), ('103', 'UNKNOWN'), ('104', '  ')]:
        plants.append({'Plant Code': plant, 'Balancing Authority Code': ba})
        gens.append(ForbiddenFields(generator(plant, fuel='UNKNOWN', status='RE', capacity='bad'), ['Plant Code']))
    gens.append(ForbiddenFields(generator('999'), ['Plant Code']))
    result = m.curate_year(2020, plants, gens)
    audit = result['audit']
    assert audit['outside_source_rows'] == 3
    assert audit['blank_ba_rows'] == audit['unmapped_plant_rows'] == 1
    assert audit['generator_rows'] == audit['selected_rows']+5
    assert audit['selected_rows'] == audit['curated_rows'] == 17


def test_new_donors_are_excluded_before_any_generator_metadata_in_legacy_mode():
    plants, gens = fixture()
    gens[7:] = [ForbiddenFields(row, ['Plant Code']) for row in gens[7:]]
    result = m.curate_year(2020, plants, gens, sources=m.LEGACY_SOURCES)
    assert len(result['units']) == 7 and result['audit']['outside_source_rows'] == 10


@pytest.mark.parametrize('pm,fuel', [('BA', 'MWH'), ('CP', 'SUN'), ('PS', 'WAT'), ('GT', 'MWH')])
def test_storage_never_accesses_capacity(pm, fuel):
    plants, gens = fixture()
    row = generator('8', 'storage', fuel=fuel, capacity='bad', pm=pm)
    gens.append(ForbiddenFields(row, ['Plant Code', 'Generator ID', 'Status', 'Prime Mover', 'Energy Source 1']))
    result = m.curate_year(2020, plants, gens)
    assert len(result['excluded']) == 1 and len(result['units']) == 17
    assert result['excluded'][0]['region'] == 'SWPP'
    assert not any('capacity' in key for key in result['excluded'][0])


def test_selected_duplicate_keys_fail_before_any_classification_or_capacity():
    plants, gens = fixture()
    gens[7] = ForbiddenFields(gens[7], ['Plant Code', 'Generator ID'])
    gens.append(generator('8'))
    with pytest.raises(ValueError, match='Duplicate generator'):
        m.curate_year(2020, plants, gens)
    plants.append(plants[0].copy())
    with pytest.raises(ValueError, match='Duplicate plant'):
        m.curate_year(2020, plants, gens)


def test_zero_pos_is_preserved_but_zero_total_or_missing_op_fail_entire_call():
    plants, gens = fixture(); gens[7]['Energy Source 1'] = 'NUC'
    gens.append(generator('8', 'zero', 'BIT', '0'))
    row = m.curate_year(2020, plants, gens)['op'][7]
    assert row['capacity_pos_mw'] == 0 and row['anchor_eligible'] is False
    assert row['zero_capacity_count'] == 1 and row['capacity_total_mw'] == 10
    gens[7]['Nameplate Capacity (MW)'] = '0'
    with pytest.raises(ValueError, match='SWPP'): m.curate_year(2020, plants, gens)
    plants, gens = fixture(); gens[7]['Status'] = 'SB'
    with pytest.raises(ValueError, match='SWPP'): m.curate_year(2020, plants, gens)


@pytest.mark.parametrize('field,value', [('Status', 'RE'), ('Prime Mover', 'CC'),
    ('Energy Source 1', 'UNKNOWN'), ('Nameplate Capacity (MW)', '-1'),
    ('Nameplate Capacity (MW)', ''), ('Nameplate Capacity (MW)', 'NaN')])
def test_new_donors_obey_same_fail_closed_metadata_and_capacity_rules(field, value):
    plants, gens = fixture(); gens[7][field] = value
    with pytest.raises(ValueError): m.curate_year(2020, plants, gens)


def test_year_specific_codes_and_multifuel_are_not_reinterpreted():
    plants, gens = fixture(); gens[7]['Energy Source 1'] = 'H2'
    with pytest.raises(ValueError, match='energy source'): m.curate_year(2020, plants, gens)
    assert m.curate_year(2021, plants, gens)['op'][7]['capacity_other_mw'] == 10
    gens[7].update({'Energy Source 1': 'WH', 'Energy Source 2': 'NG', 'Energy Source 3': 'NG',
                    'Multiple Fuels?': ' Y ', 'Cofire Fuels?': 'N'})
    result = m.curate_year(2021, plants, gens)
    unit = result['units'][7]
    assert unit['secondary_energy_sources'] == 'NG' and unit['capacity_mw'] == 10
    assert result['op'][7]['capacity_total_mw'] == 10
    assert result['op'][7]['multifuel_flagged_count'] == 1


def test_order_and_iterable_inputs_preserve_schemas_and_results():
    plants, gens = fixture()
    result = m.curate_year(2023, plants, gens)
    assert result == m.curate_year(2023, iter(plants[::-1]), iter(gens[::-1]))
    for key, columns in [('op', m.CAPACITY_COLUMNS), ('operable', m.CAPACITY_COLUMNS),
                         ('units', m.UNIT_COLUMNS), ('excluded', m.EXCLUDED_COLUMNS)]:
        assert all(tuple(row) == columns for row in result[key])
