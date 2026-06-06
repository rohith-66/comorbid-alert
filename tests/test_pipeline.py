"""
ComorbidAlert Week 1 — Test Suite
===================================
Run with: pytest tests/ -v
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.transform.clean import clean_and_validate, _validate_fips
from src.transform.join import join_on_fips, _pivot_places


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_places_df():
    return pd.DataFrame([
        {"fips": "01001", "county_name": "Autauga", "stateabbr": "AL",
         "statedesc": "Alabama", "measureid": "DIABETES",
         "data_value": 12.5, "totalpopulation": 58805, "source": "cdc_places"},
        {"fips": "01001", "county_name": "Autauga", "stateabbr": "AL",
         "statedesc": "Alabama", "measureid": "OBESITY",
         "data_value": 38.2, "totalpopulation": 58805, "source": "cdc_places"},
        {"fips": "01003", "county_name": "Baldwin", "stateabbr": "AL",
         "statedesc": "Alabama", "measureid": "DIABETES",
         "data_value": 11.8, "totalpopulation": 231767, "source": "cdc_places"},
    ])


@pytest.fixture
def sample_census_df():
    return pd.DataFrame([
        {"fips": "01001", "total_population": 58805, "median_household_income": 58143,
         "poverty_rate_pct": 13.2, "unemployment_rate_pct": 4.1, "source": "census_acs"},
        {"fips": "01003", "total_population": 231767, "median_household_income": 64432,
         "poverty_rate_pct": 11.7, "unemployment_rate_pct": 3.8, "source": "census_acs"},
    ])


@pytest.fixture
def sample_brfss_df():
    return pd.DataFrame([
        {"mmsa_code": "10700", "current_smoker_pct": 18.3, "exercise_past30d_pct": 72.1,
         "avg_sleep_hours": 7.1, "source": "brfss"},
    ])


# ─────────────────────────────────────────────────────────────────────────────
# clean.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanAndValidate:

    def test_places_drops_invalid_prevalence(self):
        df = pd.DataFrame([
            {"fips": "01001", "measureid": "DIABETES", "data_value": 12.5},
            {"fips": "01003", "measureid": "OBESITY",  "data_value": 150.0},
            {"fips": "01005", "measureid": "CHD",      "data_value": -5.0},
        ])
        result = clean_and_validate(df, "cdc_places")
        assert len(result) == 1
        assert result.iloc[0]["data_value"] == 12.5

    def test_census_removes_sentinel_values(self):
        df = pd.DataFrame([
            {"fips": "01001", "total_population": 58805, "median_household_income": -666666666},
            {"fips": "01003", "total_population": 231767, "median_household_income": 64432},
        ])
        result = clean_and_validate(df, "census_acs")
        assert result.loc[result["fips"] == "01001", "median_household_income"].isna().all()
        assert result.loc[result["fips"] == "01003", "median_household_income"].iloc[0] == 64432

    def test_validates_fips_format(self):
        df = pd.DataFrame([
            {"fips": "01001", "data_value": 10.0},
            {"fips": "BADFF", "data_value": 20.0},
            {"fips": "99999", "data_value": 30.0},
        ])
        result = _validate_fips(df, "test")
        assert set(result["fips"]) == {"01001", "99999"}

    def test_excludes_territories(self):
        df = pd.DataFrame([
            {"fips": "01001"},
            {"fips": "72001"},
            {"fips": "78010"},
        ])
        result = _validate_fips(df, "test")
        assert len(result) == 1
        assert result.iloc[0]["fips"] == "01001"

    def test_empty_dataframe_passthrough(self):
        result = clean_and_validate(pd.DataFrame(), "cdc_places")
        assert result.empty


# ─────────────────────────────────────────────────────────────────────────────
# join.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPivotPlaces:

    def test_pivot_creates_measure_columns(self, sample_places_df):
        pivot = _pivot_places(sample_places_df)
        assert "places_diabetes" in pivot.columns
        assert "places_obesity" in pivot.columns
        assert "fips" in pivot.columns

    def test_pivot_deduplicates_counties(self, sample_places_df):
        pivot = _pivot_places(sample_places_df)
        assert pivot["fips"].nunique() == len(pivot)

    def test_pivot_empty_input(self):
        result = _pivot_places(pd.DataFrame())
        assert "fips" in result.columns


class TestJoinOnFips:

    def test_census_is_spine(self, sample_places_df, sample_brfss_df, sample_census_df):
        with patch("src.transform.join._load_mmsa_crosswalk", return_value=pd.DataFrame()):
            result = join_on_fips(sample_places_df, sample_brfss_df, sample_census_df)
        assert len(result) == len(sample_census_df)

    def test_fips_column_preserved(self, sample_places_df, sample_brfss_df, sample_census_df):
        with patch("src.transform.join._load_mmsa_crosswalk", return_value=pd.DataFrame()):
            result = join_on_fips(sample_places_df, sample_brfss_df, sample_census_df)
        assert "fips" in result.columns
        assert all(result["fips"].str.len() == 5)

    def test_places_measures_present(self, sample_places_df, sample_brfss_df, sample_census_df):
        with patch("src.transform.join._load_mmsa_crosswalk", return_value=pd.DataFrame()):
            result = join_on_fips(sample_places_df, sample_brfss_df, sample_census_df)
        assert "places_diabetes" in result.columns

    def test_state_fips_derived(self, sample_places_df, sample_brfss_df, sample_census_df):
        with patch("src.transform.join._load_mmsa_crosswalk", return_value=pd.DataFrame()):
            result = join_on_fips(sample_places_df, sample_brfss_df, sample_census_df)
        assert result["state_fips"].iloc[0] == "01"


# ─────────────────────────────────────────────────────────────────────────────
# s3_writer.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestS3Writer:

    @patch("src.storage.s3_writer.boto3.client")
    def test_writes_parquet_and_metadata(self, mock_boto, sample_census_df):
        from src.storage.s3_writer import write_versioned_parquet

        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        s3_path = write_versioned_parquet(
            sample_census_df, year=2023, run_id="20230101T000000Z",
            bucket="test-bucket", prefix="test"
        )

        assert s3_path.startswith("s3://test-bucket/test/year=2023/run_id=")
        assert mock_s3.put_object.call_count >= 3

    @patch("src.storage.s3_writer.boto3.client")
    def test_latest_pointer_updated(self, mock_boto, sample_census_df):
        from src.storage.s3_writer import write_versioned_parquet

        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        write_versioned_parquet(
            sample_census_df, year=2023, run_id="20230101T120000Z",
            bucket="test-bucket", prefix="test"
        )

        keys_written = [call.kwargs.get("Key", "") for call in mock_s3.put_object.call_args_list]
        assert any("_latest.json" in k for k in keys_written)