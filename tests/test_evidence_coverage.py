"""Deterministic tests for VMP-v6 atomic evidence coverage."""

from vmp_memos.llm.evidence_coverage import (
    AtomicEvidenceFact,
    CandidateEvidenceProfile,
    build_candidate_evidence_profile,
    build_question_evidence_plan,
    extract_owned_span_ids,
    select_evidence_coverage,
)


def test_question_plan_distinguishes_count_from_temporal_duration() -> None:
    count = build_question_evidence_plan(
        "How many magazine subscriptions do I currently have?"
    )
    duration = build_question_evidence_plan(
        "How many weeks ago did I attend the Nordstrom sale?"
    )

    assert count.operator == "count"
    assert count.entity_diversity_required is True
    assert count.temporal_coverage_required is True
    assert duration.operator == "temporal"
    assert duration.temporal_coverage_required is True


def test_owned_span_parser_accepts_label_prefixed_grounded_text() -> None:
    observed = extract_owned_span_ids(
        [
            "[X:S01]USER:YESTERDAY,IATTENDEDAFRIENDSANDFAMILYSALE",
            "X:S02",
            "B1:S01",
        ],
        owner="X",
        allowed={"X:S01", "X:S02"},
    )

    assert observed == ["X:S01", "X:S02"]


def test_ungrounded_relevance_claim_cannot_influence_coverage() -> None:
    plan = build_question_evidence_plan("Which museum did I visit?")

    profile = build_candidate_evidence_profile(
        {"candidate_relevant": True, "facts": []},
        candidate_label="C06",
        session_id="s6",
        rank=6,
        plan=plan,
        allowed_span_ids={"X:S01"},
        excerpt="user: I visited a museum.",
    )

    assert profile.candidate_relevant is False
    assert profile.facts == []
    assert profile.extraction_failures == ["relevance_without_grounded_fact"]


def test_profile_accepts_scalar_need_and_span_and_infers_date_anchor() -> None:
    """Qwen may serialize one-item arrays as scalars; grounding must survive it."""

    plan = build_question_evidence_plan(
        "How many weeks ago did I attend the friends and family sale at Nordstrom?"
    )
    profile = build_candidate_evidence_profile(
        {
            "candidate_relevant": True,
            "facts": [
                {
                    "entity": "Nordstrom friends and family sale",
                    "relation": "event_date",
                    "value": "2022/11/18",
                    "temporal_anchor": None,
                    "supports_needs": "N2",
                    "evidence_spans": "X:S01",
                    "confidence": "high",
                }
            ],
        },
        candidate_label="C09",
        session_id="sale-session",
        rank=9,
        plan=plan,
        allowed_span_ids={"X:S01"},
        excerpt="user: Yesterday, I attended a friends and family sale at Nordstrom.",
    )

    assert profile.candidate_relevant is True
    assert profile.extraction_failures == []
    assert len(profile.facts) == 1
    assert profile.facts[0].supports_needs == ["N2"]
    assert profile.facts[0].evidence_spans == ["X:S01"]
    assert profile.facts[0].temporal_anchor == "2022/11/18"


def test_count_coverage_combines_two_distinct_challengers() -> None:
    plan = build_question_evidence_plan(
        "How many magazine subscriptions do I currently have?"
    )
    profiles = [
        _profile(1, "locked-a", "owns", "baseline preference", entity="profile"),
        _profile(2, "locked-b", "owns", "baseline preference", entity="account"),
        _profile(3, "locked-c", "uses", "baseline preference", entity="reader"),
        _profile(4, "boundary-a", "mentions", "magazines", entity="reading"),
        _profile(5, "boundary-b", "mentions", "subscriptions", entity="reading"),
        _profile(6, "noise", "discusses", "unrelated cycling", entity="cycling"),
        _profile(
            7,
            "national-geographic",
            "has_subscription",
            "active",
            entity="National Geographic",
            needs=["N1", "N2"],
            lexical_overlap=0.9,
        ),
        _profile(8, "noise-2", "discusses", "unrelated food", entity="food"),
        _profile(9, "noise-3", "discusses", "unrelated travel", entity="travel"),
        _profile(
            10,
            "architectural-digest",
            "has_subscription",
            "active",
            entity="Architectural Digest",
            needs=["N1", "N2"],
            lexical_overlap=0.9,
        ),
    ]

    selection = select_evidence_coverage(
        plan,
        profiles,
        output_top_k=5,
        protected_top_n=3,
        min_gain=0.1,
    )

    assert selection.selected_candidate_labels == [
        "C01",
        "C02",
        "C03",
        "C07",
        "C10",
    ]
    assert selection.promoted_candidate_labels == ["C07", "C10"]
    assert selection.gain > 0.1


def test_temporal_coverage_promotes_grounded_event_anchor() -> None:
    plan = build_question_evidence_plan(
        "How many weeks ago did I attend the friends and family sale at Nordstrom?"
    )
    profiles = [
        _profile(rank, f"noise-{rank}", "mentions", "Nordstrom shopping", entity="shopping")
        for rank in range(1, 9)
    ]
    profiles.append(
        _profile(
            9,
            "sale-event",
            "attended",
            "friends and family sale at Nordstrom",
            entity="Nordstrom sale",
            needs=["N1", "N2"],
            temporal_anchor="yesterday",
            lexical_overlap=1.0,
        )
    )
    profiles.append(
        _profile(10, "noise-10", "mentions", "unrelated event", entity="event")
    )

    selection = select_evidence_coverage(
        plan,
        profiles,
        output_top_k=5,
        protected_top_n=3,
        min_gain=0.1,
    )

    assert "C09" in selection.selected_candidate_labels
    assert selection.promoted_candidate_labels == ["C09"]


def _profile(
    rank: int,
    fact_id: str,
    relation: str,
    value: str,
    *,
    entity: str,
    needs: list[str] | None = None,
    temporal_anchor: str | None = None,
    lexical_overlap: float = 0.1,
) -> CandidateEvidenceProfile:
    return CandidateEvidenceProfile(
        candidate_label=f"C{rank:02d}",
        session_id=f"s{rank}",
        rank=rank,
        candidate_relevant=bool(needs),
        lexical_overlap=lexical_overlap,
        facts=[
            AtomicEvidenceFact(
                fact_id=fact_id,
                entity=entity,
                relation=relation,
                value=value,
                temporal_anchor=temporal_anchor,
                supports_needs=needs or [],
                evidence_spans=["X:S01"],
                confidence="high" if needs else "low",
            )
        ],
    )
