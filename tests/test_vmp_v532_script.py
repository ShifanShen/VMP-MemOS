from pathlib import Path

SCRIPT_PATH = Path("scripts/run_vmp_v532_experiment.sh")


def test_dev_candidate_stage_explicitly_allows_dev_model_selection() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    run_candidates = script.split("run_candidates() {", maxsplit=1)[1].split(
        "run_rerank() {", maxsplit=1
    )[0]

    assert 'if [[ "${split_name}" == "dev" ]]' in run_candidates
    assert "dev_args=(--allow-dev-model-selection)" in run_candidates
    assert '"${dev_args[@]}"' in run_candidates
