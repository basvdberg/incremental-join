"""Regression: TimestampType incremental columns must use calendar-day semantics post-join."""

from __future__ import annotations

from datetime import date, datetime

from pyspark.sql import SparkSession

from inc_join import inc_join


def test_same_calendar_day_different_times_yields_same_time_join_type(
    spark: SparkSession,
) -> None:
    """DiffArrivalTime and JoinType must use date truncation when time_uom is day.

    Same calendar day at 09:00 (A) and 23:00 (B) must classify as same_time with
    DiffArrivalTime == 0 after post-join F.to_date rebinding (see inc_join step 8).

    Spark compares ``timestamp <= date`` using the date at start-of-day, so an
    ``output_window_end`` of ``date(2025, 3, 6)`` would incorrectly exclude
    ``2025-03-06 23:00:00`` from the pre-join filters; use the following day as the
    inclusive end for this fixture.
    """
    schema = "TrxId INT, RecDate TIMESTAMP"
    day = date(2025, 3, 6)
    df_a = spark.createDataFrame(
        [(1, datetime(2025, 3, 6, 9, 0, 0))],
        schema,
    )
    df_b = spark.createDataFrame(
        [(1, datetime(2025, 3, 6, 23, 0, 0))],
        schema,
    )
    out = inc_join(
        df_a,
        df_b,
        how="inner",
        join_cols="TrxId",
        look_back_time=0,
        max_waiting_time=1,
        output_window_start=day,
        output_window_end=date(2025, 3, 7),
    )
    row = out.filter(out["TrxId"] == 1).collect()[0]
    assert row["DiffArrivalTime"] == 0
    assert row["JoinType"] == "same_time"
