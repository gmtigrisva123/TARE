"""Checks on the readers in `tare.dataset`.

The timestamps and label records below are copied verbatim from
group_02/experiment_03/subject_02 of the release. They are here as format
examples for the parsers, the way any parser test carries a sample of what
it has to read. No rating is invented and nothing here produces a result.

Run with:
    python -m pytest tests -q
"""

import json

import pytest

from tare import config, dataset


# --- clocks ---------------------------------------------------------------

def test_image_clock_reads_the_name():
    """Image names are HH_MM_SS_fraction, which is where t_i comes from."""
    assert dataset.image_clock("images/12_41_17_013271.png") == pytest.approx(
        12 * 3600 + 41 * 60 + 17 + 0.013271
    )


def test_label_clock_reads_the_datetime_field():
    """Label change points write the same instant with colons."""
    assert dataset.label_clock("12:41:17:013271") == pytest.approx(
        dataset.image_clock("12_41_17_013271.png")
    )


def test_fraction_is_read_by_its_width():
    """The fractional field is not always six digits.

    Image names in this release use six, and some label stamps use three,
    so the parser divides by the width it sees rather than assuming
    microseconds.
    """
    assert dataset.label_clock("00:00:01:5") == pytest.approx(1.5)
    assert dataset.label_clock("00:00:01:500") == pytest.approx(1.5)
    assert dataset.label_clock("00:00:01:500000") == pytest.approx(1.5)


def test_a_name_that_is_not_a_timestamp_is_refused():
    with pytest.raises(ValueError):
        dataset.image_clock("images/frame_0001.png")


# --- change points --------------------------------------------------------

LABELER_01_HEAD = [
    {"datetime": "12:41:17:013271", "attention": "3", "emotion": "9"},
    {"datetime": "12:41:21:014006", "attention": "2"},
    {"datetime": "12:41:23:014279", "attention": "3"},
    {"datetime": "12:41:29:015264", "attention": "2", "emotion": "1"},
    {"datetime": "12:41:48:024577", "emotion": "5"},
]
SESSION_START = dataset.image_clock("12_41_17_013271.png")


def test_emotion_only_records_do_not_move_the_rating(tmp_path):
    """A change point may carry only an emotion.

    The last record above changes the emotion at 12:41:48 and says nothing
    about attention, so the attention rating stays where the record before
    it left it.
    """
    path = tmp_path / "labeler_01.json"
    path.write_text(json.dumps(LABELER_01_HEAD))

    points = dataset._read_change_points(str(path), SESSION_START)
    assert points == {0: 3.0, 4: 2.0, 6: 3.0, 12: 2.0}
    assert 31 not in points           # the emotion-only record at +31 s


def test_forward_fill_then_back_fill():
    """III-E fills a change-point series to one rating per second."""
    filled = dataset._fill_forward_then_back({2: 3.0, 5: 2.0}, last_second=7)
    assert [filled[s] for s in range(8)] == [3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0]
    assert filled[0] == 3.0 and filled[1] == 3.0      # back-filled head


def test_fill_runs_to_the_end_of_the_session():
    """A rating holds until the session ends, not until the last change point.

    One labeler in the cohort writes a single entry at second zero for a
    five-minute session, meaning they never revised their rating. Stopping
    the fill at the last change point would leave the tail of the session
    unrated, and since (1) needs all four experts that would drop those
    frames for everyone.
    """
    filled = dataset._fill_forward_then_back({0: 3.0}, last_second=298)
    assert len(filled) == 299
    assert filled[298] == 3.0


def test_a_later_change_in_the_same_second_wins(tmp_path):
    path = tmp_path / "labeler_01.json"
    path.write_text(json.dumps([
        {"datetime": "12:41:17:100000", "attention": "3"},
        {"datetime": "12:41:17:900000", "attention": "5"},
    ]))
    assert dataset._read_change_points(str(path), SESSION_START) == {0: 5.0}


def test_a_file_with_no_attention_is_refused(tmp_path):
    path = tmp_path / "labeler_01.json"
    path.write_text(json.dumps([{"datetime": "12:41:17:013271", "emotion": "9"}]))
    with pytest.raises(ValueError):
        dataset._read_change_points(str(path), SESSION_START)


# --- the target -----------------------------------------------------------

def test_target_is_the_mean_of_four_experts():
    series = [{3: 2.0}, {3: 3.0}, {3: 3.0}, {3: 4.0}]
    assert dataset._target_at(series, 3) == pytest.approx(3.0)


def test_a_second_missing_from_one_expert_has_no_target():
    """Equation (1) averages four experts, so three is not averaged."""
    series = [{3: 2.0}, {3: 3.0}, {3: 3.0}, {}]
    assert dataset._target_at(series, 3) is None


# --- sampling -------------------------------------------------------------

def test_first_frame_of_each_bin_is_kept():
    width = config.SAMPLING_BIN_SECONDS
    relative = [(i * 0.1, f"{i}.png") for i in range(int(3 * width / 0.1) + 1)]
    kept, empty, n_bins = dataset._sample_bins(relative)

    assert n_bins == 4 and empty == []
    assert [path for _, path in kept] == ["0.png", "15.png", "30.png", "45.png"]


def test_a_gap_leaves_an_empty_bin():
    """Dropped frames produce bins with nothing in them.

    III-E flags such a bin as missing and never fills it from a neighbour,
    so it contributes no frame at all.
    """
    relative = [(0.0, "a.png"), (0.1, "b.png"), (4.0, "c.png")]
    kept, empty, n_bins = dataset._sample_bins(relative)

    assert n_bins == 3
    assert empty == [1]                                  # 1.5 s to 3.0 s
    assert [path for _, path in kept] == ["a.png", "c.png"]


def test_sampling_is_idempotent():
    """Sampling the kept frames again returns the same frames.

    The Colab extraction step keeps only the first frame of each bin and
    throws the rest away, then the loader runs the same rule over what is
    left. These agree because each kept frame still starts its own bin and
    the session still begins at the same instant, so pre-filtering the
    images does not change which frames get scored, nor how many bins come
    up empty.
    """
    rng = __import__("random").Random(0)
    clock, relative = 0.0, []
    for i in range(400):
        clock += rng.choice([0.1, 0.1, 0.1, 0.3, 1.6, 2.9])   # with dropouts
        relative.append((clock, f"{i}.png"))

    once, empty_once, bins_once = dataset._sample_bins(relative)
    twice, empty_twice, bins_twice = dataset._sample_bins(once)

    assert once == twice
    assert empty_once == empty_twice
    assert bins_once == bins_twice
    assert empty_once, "the dropouts should leave some bin empty"


# --- session order --------------------------------------------------------

def test_sessions_order_by_clock_not_by_activity_number():
    """The three activities were filmed as 03, then 08, then 05.

    Equation (6) calibrates on the earliest session in time, so the order
    has to come from the clock. Numeric order would pick the wrong second
    and third session, though it happens to agree on the first.
    """
    starts = {
        "03": dataset.image_clock("12_41_17_013271.png"),
        "05": dataset.image_clock("13_13_32_045000.png"),
        "08": dataset.image_clock("12_46_36_054000.png"),
    }
    by_clock = sorted(config.ACTIVITIES, key=lambda a: starts[a])
    assert by_clock == ["03", "08", "05"]
    assert by_clock[0] == config.LECTURE_ACTIVITY
