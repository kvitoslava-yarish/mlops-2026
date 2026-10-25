from kyiv_alerts.batch import batch_for
from kyiv_alerts.etl.clean import clean_batch
from kyiv_alerts.etl.validate import validate_raw

B = batch_for("2025-06")


def test_rules_are_counted(make_raw):
    raw = make_raw(
        ("Kyiv City", "oblast", "2025-06-01 00:00:00", "2025-06-01 00:30:00"),
        ("Kyiv City", "oblast", "2025-06-01 00:00:00", "2025-06-01 00:30:00"),  # duplicate
        ("Kyiv City", "oblast", "2025-06-02 00:00:00", "2025-06-01 23:00:00"),  # end < start
        ("Kyiv City", "oblast", "2025-06-03 00:00:00", ""),  # unfinished
    )
    episodes, c = clean_batch(raw, B)
    assert (c["rows_raw"], c["rows_duplicate"], c["rows_bad_time"], c["rows_unfinished"]) == (
        4,
        1,
        1,
        1,
    )
    assert c["rows_clean"] == len(episodes) == 1
    assert c["invalid_share"] == round(2 / 3, 4)


def test_raions_merge_into_one_oblast_episode(make_raw):
    raw = make_raw(
        ("Sumska oblast", "raion", "2025-06-05 10:00:00", "2025-06-05 10:40:00"),
        ("Sumska oblast", "raion", "2025-06-05 10:30:00", "2025-06-05 11:00:00"),  # overlaps
        ("Sumska oblast", "raion", "2025-06-05 11:00:00", "2025-06-05 11:10:00"),  # touches
        ("Sumska oblast", "raion", "2025-06-05 12:00:00", "2025-06-05 12:05:00"),  # separate
    )
    episodes, c = clean_batch(raw, B)
    assert len(episodes) == 2
    first = episodes.iloc[0]
    assert str(first.started_at) == "2025-06-05 10:00:00+00:00"
    assert str(first.finished_at) == "2025-06-05 11:10:00+00:00"
    assert first.n_records == 3 and first.duration_min == 70
    assert c["rows_merged"] == 2


def test_guessed_end_is_kept_but_flagged(make_raw):
    raw = make_raw(("Kyiv City", "oblast", "2025-06-01 00:00:00", "2025-06-01 00:30:00", True))
    episodes, _ = clean_batch(raw, B)
    assert bool(episodes.iloc[0].end_is_guess)


def test_bad_batch_passes_validate_but_not_cleaning(make_raw):
    """The make bad-batch corruption: schema fine, ends before starts."""
    raw = make_raw(
        *[("Kyiv City", "oblast", f"2025-06-{d:02d} 01:00:00", f"2025-06-{d:02d} 00:55:00")
          for d in range(1, 4)],
        ("Kyiv City", "oblast", "2025-06-10 01:00:00", "2025-06-10 01:30:00"),
    )  # fmt: skip
    validate_raw(raw, B)
    _, c = clean_batch(raw, B)
    assert c["invalid_share"] == 0.75
