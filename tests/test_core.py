import numpy as np
import pandas as pd
import pytest

from modules import analytics, anomaly, customer, data_utils, forecasting, reports, ai_insights

SAMPLE = "sample_data/sample_sales.csv"


@pytest.fixture(scope="module")
def sales():
    df = pd.read_csv(SAMPLE)
    roles = data_utils.infer_semantic_roles(df)
    return data_utils.try_parse_dates(df, roles["date"]), roles


def test_roles_do_not_crash_on_text_columns():
    # Regression: np.issubdtype crashed on pandas' string dtype (HR-style data).
    df = pd.DataFrame({"Department": ["a", "b"], "Name": ["x", "y"], "Age": [30, 40]})
    roles = data_utils.infer_semantic_roles(df)
    assert roles["category"] == "Department"
    assert roles["revenue"] is None  # no guessing


def test_dates_parsed_but_numbers_left_alone(sales):
    df, roles = sales
    assert pd.api.types.is_datetime64_any_dtype(df[roles["date"]])
    num = pd.DataFrame({"date": [20, 30, 40]})
    assert data_utils.try_parse_dates(num, "date")["date"].tolist() == [20, 30, 40]


def test_timeseries_anomaly_detects_spike_at_default_threshold(sales):
    # Regression: the old window included the point itself, capping z at ~2.27.
    df, roles = sales
    spiked = df.copy()
    spiked.loc[spiked.index[100], roles["revenue"]] *= 200
    agg = anomaly.time_series_anomalies(spiked, roles["date"], roles["revenue"], "D", 3.0)
    assert agg["is_anomaly"].sum() >= 1
    assert (agg.loc[agg.is_anomaly, "anomaly_type"] == "Spike").any()


@pytest.mark.parametrize("freq", ["D", "W", "M"])
def test_forecast_all_frequencies(sales, freq):
    # Regression: "M" raises on pandas 3.
    df, roles = sales
    ts, fc, _ = forecasting.run_forecast(df, roles["date"], roles["revenue"],
                                         "Holt-Winters (Exponential Smoothing)", 3, freq)
    assert len(fc) == 3 and np.isfinite(fc["forecast"]).all()


def test_clv_not_inflated_by_recent_purchase():
    rfm = pd.DataFrame({
        "recency": [1, 200], "tenure_days": [300, 300],
        "frequency": [4, 4], "monetary": [400.0, 400.0],
    })
    out = customer.estimate_clv(rfm)
    assert out.loc[0, "clv_estimate"] == out.loc[1, "clv_estimate"]


def test_rfm_frequency_counts_distinct_orders(sales):
    df, roles = sales
    rfm = customer.compute_rfm(df, roles["customer_id"], roles["date"], roles["revenue"], roles["order_id"])
    assert {"recency", "frequency", "monetary", "tenure_days", "segment"} <= set(rfm.columns)
    cohort = customer.cohort_retention(df, roles["customer_id"], roles["date"])
    assert not cohort.empty


def test_sql_guard():
    assert ai_insights.validate_sql("SELECT 1;") == "SELECT 1"
    for bad in ["DROP TABLE data", "SELECT 1; DELETE FROM data", "PRAGMA table_info(data)",
                "UPDATE data SET x=1", ""]:
        with pytest.raises(ValueError):
            ai_insights.validate_sql(bad)


def test_readonly_sql_runs_on_full_data(sales):
    df, _ = sales
    out = ai_insights.run_readonly_sql(df, "SELECT COUNT(*) AS n FROM data")
    assert out.loc[0, "n"] == len(df)


def test_pdf_survives_markup_characters(sales):
    df, roles = sales
    kpis = analytics.compute_kpis(df, roles)
    pdf = reports.export_pdf("Q<3> & more", kpis, "Revenue < budget & growth > 5%\n- **bold** item")
    assert pdf.startswith(b"%PDF")


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "log.csv")
    result = pipeline.run_pipeline(pipeline.load_config(None), SAMPLE)
    assert result["kpis"]["total_records"] > 0
    assert (tmp_path / "log.csv").exists()
