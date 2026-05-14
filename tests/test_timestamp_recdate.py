"""Regression: TimestampType incremental columns must use calendar-day semantics post-join."""

from __future__ import annotations

from datetime import date, datetime, timezone

from pyspark.sql import SparkSession

from inc_join import inc_join


def test_same_calendar_day_different_times_yields_same_time_join_type(
    spark: SparkSession,
) -> None:
    """DiffArrivalTime and JoinType must use date truncation when time_uom is day.

    Same calendar day at 09:00 (A) and 23:00 (B) must classify as same_time with
    DiffArrivalTime == 0 after post-join F.to_date rebinding (see inc_join step 8).

    Timestamp-fixture best practice
    -------------------------------
    Spark's timestamp semantics (parsing literals, ``F.to_date``, ``F.datediff``
    on timestamps) are governed by ``spark.sql.session.timeZone``. ``conftest.py``
    pins that to ``UTC``. We pair it here with **timezone-aware** Python
    datetimes (``tzinfo=timezone.utc``), so the values do not silently pick up
    the driver's local zone via ``createDataFrame``. Aligning the Python tzinfo
    with the session time zone makes the wall-clock date in the fixture match
    the calendar date Spark uses for ``to_date``/``datediff``, removing any
    dependency on the host or CI machine's local time zone.

    The ``output_window_end`` uses the day after the fixture date because Spark
    compares ``timestamp <= date`` at start-of-day; a same-day end would
    incorrectly exclude ``23:00:00``.
    """
    schema = "TrxId INT, RecDate TIMESTAMP"
    day = date(2025, 3, 6)
    df_a = spark.createDataFrame(
        [(1, datetime(2025, 3, 6, 9, 0, 0, tzinfo=timezone.utc))],
        schema,
    )
    df_b = spark.createDataFrame(
        [(1, datetime(2025, 3, 6, 23, 0, 0, tzinfo=timezone.utc))],
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
