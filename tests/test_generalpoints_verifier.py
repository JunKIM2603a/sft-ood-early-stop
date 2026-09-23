from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sft_ood_early_stop.generalpoints import score_response


def test_correct_formula():
    score = score_response(
        '{"formula": "(8-6)*3*4"}',
        [8, 6, 3, 4],
        24,
    )
    assert score.correct
    assert score.formula_valid


def test_equal_sign_is_accepted():
    score = score_response(
        '{"formula": "(8-6)*3*4=24"}',
        [8, 6, 3, 4],
        24,
    )
    assert score.correct


def test_wrong_number_multiset_fails():
    score = score_response(
        '{"formula": "(8-6)*4*4"}',
        [8, 6, 3, 4],
        24,
    )
    assert not score.correct
    assert score.reason == "wrong_number_multiset"


def test_wrong_target_is_valid_but_incorrect():
    score = score_response(
        '{"formula": "8+6+4+3"}',
        [8, 6, 4, 3],
        24,
    )
    assert not score.correct
    assert score.formula_valid
    assert score.reason == "wrong_target"


def test_exponent_is_illegal():
    score = score_response(
        '{"formula": "2**3+8+6"}',
        [2, 3, 8, 6],
        24,
    )
    assert not score.correct
    assert not score.formula_valid
    assert score.reason == "illegal_formula"


def test_formula_can_be_extracted_from_fenced_text():
    fence = chr(96) * 3
    response = (
        fence
        + "json\n"
        + '{"cards": ["8", "6", "3", "4"], "formula": "(8-6)*3*4"}'
        + "\n"
        + fence
    )
    assert score_response(response, [8, 6, 3, 4], 24).correct
