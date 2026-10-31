import numpy as np
import pandas as pd
import pytest

from scripts import coarse_fields_v24 as m


def monthly():
    dates = pd.date_range("2023-01-01", periods=12, freq="MS", tz="UTC")
    mix = np.zeros((12, 8))
    mix[:, 0] = np.arange(1, 13)/13
    mix[:, 1] = 1-mix[:, 0]
    end = dates+pd.offsets.MonthBegin(1)
    hours = (end-dates)/pd.Timedelta(hours=1)
    x = pd.DataFrame(dict(region="BPAT", period_start=dates, period_end=end,
        available_at=end+pd.Timedelta(days=30), total_mwh=np.arange(1, 13),
        observed_hours=hours, expected_hours=hours, coverage=1.))
    x[m.SHARES] = mix
    return x


@pytest.mark.parametrize("schema", list(m.SCHEMAS))
def test_sharp_endpoint_witnesses(schema):
    x = np.random.default_rng(17).dirichlet(np.ones(8), size=100)
    lo, hi, a, b = m.bounds(x, schema)
    assert np.all(lo <= x@m.EF+1e-10)
    assert np.all(hi >= x@m.EF-1e-10)
    for group in m.SCHEMAS[schema]:
        np.testing.assert_allclose(a[:, group].sum(1), x[:, group].sum(1), atol=1e-14)
        np.testing.assert_allclose(b[:, group].sum(1), x[:, group].sum(1), atol=1e-14)
    np.testing.assert_array_equal(a@m.EF, lo)
    np.testing.assert_array_equal(b@m.EF, hi)


def test_field_reveal_widths():
    x = np.array([[.5, .2, .1, .2, 0, 0, 0, 0]])
    expected = {"eight": 0, "merged_CGO": 480, "merged_positive_proxy": 480,
                "CGO_plus_coal": 120, "CGO_plus_gas": 120, "CGO_plus_oil": 420}
    for schema, width in expected.items():
        lo, hi, _, _ = m.bounds(x, schema)
        np.testing.assert_allclose(hi-lo, width, atol=1e-10)


def test_annual_is_energy_weighted_and_preserves_hours():
    x = monthly()
    annual, support = m.annualize(x)
    expected = np.average(x[m.SHARES], weights=x.total_mwh, axis=0)
    np.testing.assert_allclose(annual[m.SHARES].to_numpy()[0], expected)
    assert not np.allclose(expected, x[m.SHARES].mean())
    assert support.complete.all()
    assert annual.observed_hours.iloc[0] == 8760
    assert annual.available_at.iloc[0] == x.available_at.max()


def test_missing_months_are_not_filled():
    annual, support = m.annualize(monthly().iloc[:-1])
    assert annual.empty
    assert not support.complete.any()
    assert support.missing_months.iloc[0] == "12"


@pytest.mark.parametrize("problem", ["duplicate", "release", "simplex", "coverage", "region"])
def test_invalid_inputs_fail(problem):
    x = monthly()
    if problem == "duplicate":
        x = pd.concat([x, x.iloc[:1]])
    elif problem == "release":
        x.loc[0, "available_at"] = x.loc[0, "period_start"]
    elif problem == "simplex":
        x.loc[0, m.SHARES[0]] = -.1
    elif problem == "coverage":
        x.loc[0, "coverage"] = .5
    else:
        x.loc[0, "region"] = "UNSEEN"
    with pytest.raises(ValueError):
        m.validate(x)


def test_region_then_macro_weights():
    x = monthly()
    y = x.iloc[:1].copy()
    y["region"] = "CISO"
    y[m.SHARES] = [0, 0, 0, 1, 0, 0, 0, 0]
    mb = m.diagnose(pd.concat([x, y], ignore_index=True), "month")
    ab = m.diagnose(m.annualize(x)[0], "proxy_year")
    region, macro = m.summaries(mb, ab)
    row = macro.loc[(macro.cohort == "monthly_2023_common") & (macro.schema == "merged_CGO")].iloc[0]
    expected = region.loc[(region.cohort == "monthly_2023_common") & (region.schema == "merged_CGO"), "mean_identification_width"].mean()
    assert row.mean_identification_width == expected


def test_manifest_mutation_is_rejected(tmp_path):
    path = tmp_path/"freeze.json"
    path.write_text("{}")
    original = m.sha(path)
    frozen = dict(input_hashes={}, code_hashes={})
    m.verify_frozen(frozen, tmp_path, original)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="manifest changed"):
        m.verify_frozen(frozen, tmp_path, original)
