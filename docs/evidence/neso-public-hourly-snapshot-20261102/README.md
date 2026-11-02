# NESO public generation-mix snapshot — 2026-11-02

Retrieved: `2026-11-02T03:19:34Z`  
Source: NESO Historic GB Generation Mix, public CKAN datastore resource `f93d1835-75bc-43e5-84ad-12472b180a98`  
Official source page: https://www.neso.energy/data-portal/historic-generation-mix/historic_gb_generation_mix  
Public query API: https://api.neso.energy/api/3/action/datastore_search_sql  
Resource metadata last modified (queried 2026-11-02): `2026-11-02T02:21:36.613942`  
Query range: `[2026-11-01T00:00:00, 2026-12-01T00:00:00)` UTC.

## Captured response

`neso_generation_mix_2026_11_api_response.json` is the verbatim JSON response for the November-to-date query. It contains 51 contiguous half-hour records, from `2026-11-01T00:00:00` through `2026-11-02T01:00:00`, with all 36 returned fields. The query is included in the API response. SHA-256:

`4639c6a39c09335323ac87e77709a4af62c99fb04038ae037b9e85d12f33a57d`

## Scope

This is a snapshot of the public resource as it existed at retrieval time. The data are historical generation-mix outturns, not forecasts and not an as-issued monthly report. NESO states that the historical series is updated hourly and may change during data cleansing. This file therefore starts a prospective public snapshot record; it cannot reconstruct earlier API states or support a claim that the observations were available at their original settlement times. Preserve each future retrieval separately with its retrieval timestamp and hash.

## October local-month query for the prospective forecast

A separate query used Europe/London October boundaries, translated to `[2026-09-30T23:00:00Z, 2026-11-01T00:00:00Z)`, so it includes both duplicated local-clock hours during the autumn daylight-saving transition. The response `neso_generation_mix_2026_10_localmonth_asof_20261102.json` contains 1,490 unique, contiguous half-hour rows and is the input snapshot used by the t−2 prospective forecast. Its SHA-256 is `6629e553468c5d06286263e9951355e5880477158059f71d7e499d7b30d5c140`. Retrieval was complete by the declared forecast origin `2026-11-02T03:23:29Z`.

The earlier file `neso_generation_mix_2026_10_asof_20261102.json` queried the UTC calendar month `[2026-10-01T00:00:00Z, 2026-11-01T00:00:00Z)` and contains 1,488 rows; it is retained as a separate exact API response and is not used for the full London local-month input.
