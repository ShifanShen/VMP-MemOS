from pathlib import Path

SCRIPT_PATH = Path("scripts/run_vmp_v54_experiment.sh")


def test_v54_reuses_frozen_v532_dev_candidates() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "lme_dev_vmp_v532_candidates_seed42" in script
    assert 'STAGE="${STAGE:-dev_rerank}"' in script
    assert "dev_candidates)" not in script


def test_v54_uses_paired_symbolic_span_protocol_and_gate() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "vmp_v54_symbolic_span_selector_v1" in script
    assert "vmp_v54_symbolic_span_boundary_v1" in script
    assert '--prompt-version "${SELECTOR_PROMPT_VERSION}"' in script
    assert (
        '--expected-selector-prompt-version "${SELECTOR_PROMPT_VERSION}"' in script
    )
    assert (
        '--expected-boundary-prompt-version "${BOUNDARY_PROMPT_VERSION}"' in script
    )


def test_v54_keeps_test_behind_dev_gate() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    test_candidates = script.split("test_candidates)", maxsplit=1)[1].split(
        "test_rerank)", maxsplit=1
    )[0]
    test_rerank = script.split("test_rerank)", maxsplit=1)[1]

    assert "check_dev_gate" in test_candidates
    assert "check_dev_gate" in test_rerank
