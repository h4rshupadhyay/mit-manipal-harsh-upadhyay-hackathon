import json
import math
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from zoneinfo import ZoneInfo

import pytest

from risk_engine.backtest import metrics as m
from risk_engine.backtest.splits import FoldGroup, InnerFold
from risk_engine.domain import EventClass, FactorShock, ShockType, ShockUnit

G = EventClass.GEOPOLITICAL
C = EventClass.CREDIT_DEFAULT
START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_classification_counts_abstentions_as_false_negatives_and_discloses_classes() -> None:
    result = m.classification_metrics([G, G, C, C], [G, C, C, None])
    assert result.macro_f1 == pytest.approx(7 / 12)
    assert result.evaluated_classes == (G, C)
    assert result.non_abstention_coverage == 0.75
    assert result.confusion[0] == (1, 0, 1, 0, 0, 0, 0, 0, 0)
    assert result.confusion[2] == (0, 0, 1, 0, 0, 0, 0, 0, 1)
    assert result.per_class[0].precision == 1
    assert result.per_class[0].recall == 0.5
    assert result.per_class[2].f1 == 0.5
    assert result.per_class[1].f1 is None


def test_all_abstentions_have_zero_f1_without_inventing_precision() -> None:
    result = m.classification_metrics([G], [None])
    assert result.macro_f1 == 0
    assert result.non_abstention_coverage == 0
    assert result.per_class[0].precision is None
    assert result.per_class[0].recall == 0


def test_entity_accuracy_counts_missing_links_as_errors() -> None:
    assert m.entity_link_accuracy(["a", "b", "c"], ["a", None, "b"]) == pytest.approx(1 / 3)


def test_joint_confidence_calibration_uses_literal_binary_losses_and_reliability_bins() -> None:
    result = m.confidence_metrics([True, False, True, False], [0.8, 0.4, 0.6, 0.2], [0, 0.5, 1])
    assert result.brier_score == pytest.approx(0.1)
    assert result.log_loss == pytest.approx(0.3669845875401002)
    assert result.reliability[0].count == 2
    assert result.reliability[0].mean_confidence == pytest.approx(0.3)
    assert result.reliability[0].accuracy == 0
    assert result.reliability[1].mean_confidence == pytest.approx(0.7)
    assert result.reliability[1].accuracy == 1


def test_reliability_bins_are_left_closed_with_last_right_closed_and_empty_bins_explicit() -> None:
    result = m.confidence_metrics([False, True, True], [0, 0.5, 1], [0, 0.25, 0.5, 1])
    assert [bin_.count for bin_ in result.reliability] == [1, 0, 2]
    assert result.reliability[1].accuracy is None
    assert result.reliability[1].mean_confidence is None
    assert result.reliability[2].mean_confidence == 0.75
    assert m.confidence_metrics([False, True], [0, 1], [0, 1]).log_loss == 0
    with pytest.raises(ValueError, match="certain"):
        m.confidence_metrics([True], [0], [0, 1])


def test_severity_uses_average_tied_ranks_and_occupied_bucket_means() -> None:
    result = m.severity_metrics([1, 1, 3, 4], [1, 2, 2, 5], [1, 3, 2, 4], "USD")
    assert result.rank_correlation == pytest.approx(0.6324555320336759)
    assert result.ordinal_mae == 0.75
    assert result.bucket_monotonicity == 1
    assert [(b.impact_score, b.count, b.mean_realized_loss) for b in result.buckets] == [
        (1, 2, 2),
        (3, 1, 2),
        (4, 1, 4),
    ]
    assert result.loss_unit == "USD"
    decreasing = m.severity_metrics([1, 2, 3], [1, 2, 3], [1, 3, 2], "USD")
    assert decreasing.bucket_monotonicity == 0.5


def test_constant_ranks_and_single_bucket_are_explicitly_undefined() -> None:
    result = m.severity_metrics([2, 2], [2, 3], [4, 4], "USD")
    assert result.rank_correlation is None
    assert result.bucket_monotonicity is None
    assert result.ordinal_mae == 0.5


def test_interval_coverage_includes_endpoints_and_preserves_units_and_nominal_level() -> None:
    result = m.interval_metrics([1, 2, 3], [2, 4, 5], [1, 5, 5], "USD", 0.9)
    assert result.coverage == pytest.approx(2 / 3)
    assert result.mean_width == pytest.approx(5 / 3)
    assert result.unit == "USD"
    assert result.nominal_coverage == 0.9


@pytest.mark.parametrize(
    "call",
    [
        lambda: m.classification_metrics([], []),
        lambda: m.classification_metrics([G], [G, C]),
        lambda: m.classification_metrics(["bad"], [G]),
        lambda: m.entity_link_accuracy([""], [None]),
        lambda: m.confidence_metrics([1], [0.5], [0, 1]),
        lambda: m.confidence_metrics([True], [float("nan")], [0, 1]),
        lambda: m.confidence_metrics([True], [1.1], [0, 1]),
        lambda: m.confidence_metrics([True], [0.5], [0, 0.5, 0.5, 1]),
        lambda: m.confidence_metrics([True], [0.5], [0.1, 1]),
        lambda: m.severity_metrics([0], [1], [1], "USD"),
        lambda: m.severity_metrics([True], [1], [1], "USD"),
        lambda: m.severity_metrics([1], [1], [float("inf")], "USD"),
        lambda: m.interval_metrics([2], [1], [1], "USD", 0.9),
        lambda: m.interval_metrics([1], [2], [1], "", 0.9),
        lambda: m.interval_metrics([1], [2], [1], "USD", 1.1),
    ],
)
def test_numerical_metrics_reject_incomplete_or_inconsistent_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()


def shock(factor: str, value: float, unit: ShockUnit = ShockUnit.DECIMAL) -> FactorShock:
    return FactorShock(
        factor_id=factor,
        value=value,
        shock_type=ShockType.RELATIVE,
        unit=unit,
        horizon_days=2,
    )


def scenario(**updates):
    values = dict(
        case_id="case-1",
        occurred_at=START,
        snapshot_id="snapshot-1",
        configuration_version="config-1",
        realized=(shock("equity", 1), shock("fx", -1)),
        joint_samples=(
            (shock("equity", -2), shock("fx", -2)),
            (shock("equity", 2), shock("fx", 0)),
        ),
    )
    values.update(updates)
    return m.ScenarioObservation(**values)


def test_scenario_direction_uses_sample_mean_and_marginal_empirical_crps() -> None:
    # equity: mean 0 has wrong direction; CRPS=(3+1)/2 - 8/(2*4)=1.
    # fx: mean -1 correct; CRPS=(1+1)/2 - 4/(2*4)=0.5.
    result = m.scenario_metrics([scenario()])
    assert [(f.factor_id, f.direction_accuracy, f.crps) for f in result.factors] == [
        ("equity", 0, 1),
        ("fx", 1, 0.5),
    ]
    assert result.factors[0].unit == ShockUnit.DECIMAL
    assert result.factors[0].horizon_days == 2
    assert result.model_dump_json() == m.scenario_metrics([scenario()]).model_dump_json()


def test_scenario_zero_direction_and_single_joint_sample_are_defined() -> None:
    result = m.scenario_metrics(
        [
            scenario(
                realized=(shock("equity", 0),),
                joint_samples=((shock("equity", 0),),),
            )
        ]
    )
    assert result.factors[0].direction_accuracy == 1
    assert result.factors[0].crps == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"joint_samples": ()},
        {"joint_samples": ((shock("equity", 1),),)},
        {"joint_samples": ((shock("equity", 1, ShockUnit.PERCENT), shock("fx", 1)),)},
        {
            "joint_samples": (
                (shock("equity", 1).model_copy(update={"horizon_days": 3}), shock("fx", 1)),
            )
        },
        {"realized": (shock("equity", 1), shock("equity", 2))},
        {"realized": (shock("fx", -1), shock("equity", 1))},
    ],
)
def test_scenario_rejects_incomplete_or_inconsistent_joint_vectors(updates) -> None:
    with pytest.raises(ValueError):
        m.scenario_metrics([scenario(**updates)])


def valuation(case_id="case-1", **updates):
    values = dict(
        case_id=case_id,
        occurred_at=START,
        comparison_scope="declared-supported-instruments",
        snapshot_id="snapshot-1",
        configuration_version="config-1",
        predicted_pnl=Decimal("-8"),
        realized_pnl=Decimal("-10"),
        currency="USD",
        horizon_days=2,
        supported_gross_value=Decimal("80"),
        total_gross_value=Decimal("100"),
    )
    values.update(updates)
    return m.ValuationObservation(**values)


def test_valuation_pnl_error_and_supported_value_share_preserve_currency() -> None:
    result = m.valuation_metrics(
        [
            valuation(),
            valuation(
                "case-2",
                predicted_pnl=Decimal("-16"),
                realized_pnl=Decimal("-20"),
                supported_gross_value=Decimal("20"),
            ),
        ]
    )
    assert result.pnl_mae == Decimal("3")
    assert result.supported_value_share == Decimal("0.5")
    assert result.currency == "USD"
    assert result.horizon_days == 2


def test_zero_supported_value_does_not_imply_zero_risk_or_zero_error() -> None:
    result = m.valuation_metrics([valuation(supported_gross_value=Decimal("0"))])
    assert result.supported_value_share == 0
    assert result.pnl_mae == 2


@pytest.mark.parametrize(
    "updates",
    [
        {"supported_gross_value": Decimal("101")},
        {"supported_gross_value": Decimal("-1")},
        {"total_gross_value": Decimal("0")},
        {"predicted_pnl": Decimal("NaN")},
    ],
)
def test_valuation_rejects_invalid_values(updates) -> None:
    with pytest.raises(ValueError):
        m.valuation_metrics([valuation(**updates)])


def alert(case_id: str, priority: float, material: bool, loss: str, flagged=True, **updates):
    values = dict(
        case_id=case_id,
        occurred_at=START,
        is_alert=flagged,
        material_event=material,
        action_priority=priority,
        realized_loss=Decimal(loss),
    )
    values.update(updates)
    return m.AlertObservation(**values)


def alert_window(**updates):
    values = dict(
        start=START,
        end=START + timedelta(days=2),
        currency="USD",
        budget=1,
        snapshot_id="snapshot-1",
        configuration_version="config-1",
    )
    values.update(updates)
    return m.AlertWindow(**values)


def test_alert_metrics_use_explicit_elapsed_days_and_loss_capture_at_priority_budget() -> None:
    rows = [
        alert("a", 9, True, "10"),
        alert("b", 8, False, "0"),
        alert("c", 7, True, "20"),
        alert("d", 10, True, "30", flagged=False),
    ]
    result = m.alert_metrics(rows, alert_window())
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.alerts_per_day == 1.5
    assert result.loss_captured_at_budget == Decimal("0.1666666666666666666666666667")
    assert result.budget_case_ids == ("a",)
    assert result.currency == "USD"


def test_alert_budget_ties_by_id_and_window_is_half_open_in_utc() -> None:
    rows = [alert("b", 9, True, "20"), alert("a", 9, True, "10")]
    window = alert_window(end=START + timedelta(hours=12))
    assert m.alert_metrics(rows, window).budget_case_ids == ("a",)
    assert m.alert_metrics(rows, window).alerts_per_day == 4
    local = START.astimezone(timezone(timedelta(hours=5, minutes=30)))
    assert m.alert_metrics(rows, alert_window(start=local)).alerts_per_day == 1
    with pytest.raises(ValueError, match="window"):
        m.alert_metrics([alert("end", 1, True, "1", occurred_at=window.end)], window)


def test_undefined_alert_denominators_and_zero_budget_are_explicit() -> None:
    result = m.alert_metrics([alert("a", 1, False, "0", flagged=False)], alert_window(budget=0))
    assert result.precision is None
    assert result.recall is None
    assert result.loss_captured_at_budget is None
    assert result.alerts_per_day == 0
    assert result.budget_case_ids == ()
    assert (
        m.alert_metrics([alert("a", 1, True, "10")], alert_window(budget=0)).loss_captured_at_budget
        == 0
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: m.scenario_metrics([]),
        lambda: m.scenario_metrics([scenario(), scenario()]),
        lambda: m.valuation_metrics([]),
        lambda: m.valuation_metrics([valuation(), valuation()]),
        lambda: m.valuation_metrics([valuation(), valuation("b", currency="INR")]),
        lambda: m.valuation_metrics([valuation(), valuation("b", horizon_days=3)]),
        lambda: m.alert_metrics([], alert_window()),
        lambda: m.alert_metrics(
            [alert("a", 1, True, "1"), alert("a", 1, True, "1")], alert_window()
        ),
        lambda: alert_window(end=START),
        lambda: alert_window(start=START.replace(tzinfo=None)),
        lambda: alert_window(budget=True),
        lambda: alert("a", float("inf"), True, "1"),
        lambda: alert("a", 1, True, "-1"),
    ],
)
def test_scenario_valuation_and_alerts_reject_ambiguous_input(call) -> None:
    with pytest.raises(ValueError):
        call()


def group(identity: str, day: int) -> FoldGroup:
    return FoldGroup(
        cluster_id=identity, event_time=START + timedelta(days=day), source_item_ids=(identity,)
    )


INNER = (
    InnerFold(fold_id="inner-1", train=(group("a", 0),), embargo=(), validation=(group("b", 1),)),
    InnerFold(fold_id="inner-2", train=(group("b", 1),), embargo=(), validation=(group("c", 2),)),
)


def candidate(identity: str, scores: tuple[float, ...], complexity_count: int, **updates):
    values = dict(
        candidate_id=identity,
        configuration_version=f"config-{identity}",
        convention_version="one-standard-error-v1",
        evaluation_scope="inner-development",
        snapshot_id="snapshot-1",
        split_version="nested-grouped-chronological-v1",
        outer_fold_id="outer-1",
        objective_name="macro_f1",
        objective_unit="dimensionless",
        direction="maximize",
        complexity_dimensions=("rule_count",),
        complexity=(complexity_count,),
        fold_results=tuple(
            m.InnerDevelopmentResult(fold=fold, score=score)
            for fold, score in zip(INNER, scores, strict=False)
        ),
    )
    values.update(updates)
    return m.CandidateDevelopmentResult(**values)


def test_selects_simplest_at_inclusive_best_standard_error_boundary() -> None:
    # Best mean .75; sample SD sqrt(.125), SE .25. Challenger at .5 qualifies.
    candidates = [
        candidate("best", (0.5, 1), 3),
        candidate("simple", (0.5, 0.5), 1),
        candidate("noisy", (0.1, 0.7), 0),
    ]
    result = m.select_within_one_standard_error(candidates)
    assert result.selected_candidate_id == "simple"
    assert result.best_candidate_id == "best"
    assert result.best_mean == 0.75
    assert result.best_standard_error == 0.25
    assert result.eligibility_threshold == 0.5
    assert result.eligible_candidate_ids == ("best", "simple")
    assert result.summaries[0].mean_score == 0.75
    assert result.summaries[0].standard_error == 0.25
    assert result.candidates == tuple(sorted(candidates, key=lambda c: c.candidate_id))
    reverse = m.select_within_one_standard_error(list(reversed(candidates)))
    assert reverse.model_dump_json() == result.model_dump_json()
    assert m.SelectionResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("direction,sign", [("maximize", 1), ("minimize", -1)])
def test_selection_preserves_v1_finite_score_boundary_exactly(direction, sign) -> None:
    result = m.select_within_one_standard_error(
        [
            candidate("best", (sign * 0.01, sign * 0.06), 1, direction=direction),
            candidate("simple", (sign * 0.01, sign * 0.01), 0, direction=direction),
        ]
    )
    assert result.best_mean == sign * 0.034999999999999996
    assert result.best_standard_error == 0.024999999999999998
    assert result.eligibility_threshold == sign * 0.009999999999999998
    assert result.eligible_candidate_ids == ("best", "simple")
    assert result.selected_candidate_id == "simple"


def test_selection_restores_existing_v1_json_with_exact_finite_score_summary() -> None:
    result = m.select_within_one_standard_error([candidate("best", (0.2, 0.3), 1)])
    # These literal summaries were emitted by the original finite-deviation v1 path.
    payload = result.model_dump(mode="json")
    payload["best_mean"] = 0.25
    payload["best_standard_error"] = 0.04999999999999999
    payload["eligibility_threshold"] = 0.2
    payload["summaries"] = [
        {"candidate_id": "best", "mean_score": 0.25, "standard_error": 0.04999999999999999}
    ]
    restored = m.SelectionResult.model_validate_json(json.dumps(payload))
    assert restored.best_standard_error == 0.04999999999999999
    assert restored.model_dump(mode="json") == payload


def test_minimization_uses_best_standard_error_and_lexicographic_complexity() -> None:
    values = [
        candidate("best", (1, 3), 3, direction="minimize", objective_name="ordinal_mae"),
        candidate("simple", (3, 3), 1, direction="minimize", objective_name="ordinal_mae"),
        candidate("bad", (4, 4), 0, direction="minimize", objective_name="ordinal_mae"),
    ]
    result = m.select_within_one_standard_error(values)
    assert result.selected_candidate_id == "simple"
    assert result.best_mean == 2
    assert result.best_standard_error == 1
    assert result.eligibility_threshold == 3
    assert result.eligible_candidate_ids == ("best", "simple")


def test_score_complexity_and_candidate_ties_are_deterministic_with_zero_se() -> None:
    result = m.select_within_one_standard_error(
        [
            candidate("z", (0.8, 0.8), 1),
            candidate("a", (0.8, 0.8), 1),
            candidate("b", (0.7, 0.7), 0),
        ]
    )
    assert result.selected_candidate_id == "a"
    assert result.best_candidate_id == "a"
    assert result.best_standard_error == 0
    assert result.eligible_candidate_ids == ("a", "z")
    tuple_order = m.select_within_one_standard_error(
        [
            candidate(
                "first",
                (0.8, 0.8),
                1,
                complexity=(1, 9),
                complexity_dimensions=("rules", "parameters"),
            ),
            candidate(
                "second",
                (0.8, 0.8),
                1,
                complexity=(2, 0),
                complexity_dimensions=("rules", "parameters"),
            ),
        ]
    )
    assert tuple_order.selected_candidate_id == "first"


@pytest.mark.parametrize(
    "updates",
    [
        {"evaluation_scope": "final"},
        {"evaluation_scope": "outer-test"},
        {"convention_version": "unknown"},
        {"complexity": (-1,)},
        {"complexity": (True,)},
        {"complexity": ()},
        {"complexity": (1, 2)},
        {"fold_results": ()},
        {"fold_results": (None,)},
        {"direction": "unknown"},
    ],
)
def test_selection_records_reject_non_development_and_incomplete_inputs(updates) -> None:
    with pytest.raises(ValueError):
        candidate("a", (0.5, 1), 1, **updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"snapshot_id": "other"},
        {"objective_name": "other"},
        {"objective_unit": "USD"},
        {"outer_fold_id": "other"},
        {"direction": "minimize"},
        {"complexity_dimensions": ("different_meaning",)},
    ],
)
def test_selection_rejects_comparing_unmatched_candidate_evidence(updates) -> None:
    with pytest.raises(ValueError, match="match"):
        m.select_within_one_standard_error(
            [
                candidate("a", (0.5, 1), 1),
                candidate("b", (0.5, 1), 1, **updates),
            ]
        )


def test_selection_matches_complete_fold_evidence_not_only_ids() -> None:
    changed = INNER[1].model_copy(update={"validation": (group("changed", 2),)})
    second = candidate(
        "b",
        (0.5, 1),
        1,
        fold_results=(
            m.InnerDevelopmentResult(fold=INNER[0], score=0.5),
            m.InnerDevelopmentResult(fold=changed, score=1),
        ),
    )
    with pytest.raises(ValueError, match="match"):
        m.select_within_one_standard_error([candidate("a", (0.5, 1), 1), second])


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "duplicate",
        "one-fold",
        "duplicate-fold",
        "reverse",
        "nonfinite",
        "copied-scope",
        "copied-fold",
    ],
)
def test_selection_revalidates_and_rejects_invalid_samples(kind: str) -> None:
    with pytest.raises(ValueError):
        first = candidate("a", (0.5, 1), 1)
        values = [first]
        if kind == "empty":
            values = []
        elif kind == "duplicate":
            values.append(first)
        elif kind == "one-fold":
            values = [candidate("a", (0.5,), 1)]
        elif kind == "duplicate-fold":
            values = [first.model_copy(update={"fold_results": (first.fold_results[0],) * 2})]
        elif kind == "reverse":
            values = [first.model_copy(update={"fold_results": first.fold_results[::-1]})]
        elif kind == "nonfinite":
            values = [candidate("a", (float("nan"), 1), 1)]
        elif kind == "copied-scope":
            values = [first.model_copy(update={"evaluation_scope": "final"})]
        elif kind == "copied-fold":
            bad = INNER[0].model_copy(update={"validation": INNER[0].train})
            bad_result = first.fold_results[0].model_copy(update={"fold": bad})
            values = [
                first.model_copy(update={"fold_results": (bad_result, first.fold_results[1])})
            ]
        m.select_within_one_standard_error(values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("selected_candidate_id", "best"),
        ("best_candidate_id", "simple"),
        ("best_mean", 0.5),
        ("best_standard_error", 0.3),
        ("eligibility_threshold", 0.4),
        ("eligible_candidate_ids", ["simple"]),
    ],
)
def test_restored_selection_rejects_contradictory_derived_claims(field, value) -> None:
    result = m.select_within_one_standard_error(
        [
            candidate("best", (0.5, 1), 3),
            candidate("simple", (0.5, 0.5), 1),
        ]
    )
    payload = result.model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValueError, match="evidence"):
        m.SelectionResult.model_validate(payload)


def test_restored_selection_rejects_tampered_summaries_and_reordered_audit_evidence() -> None:
    result = m.select_within_one_standard_error(
        [
            candidate("best", (0.5, 1), 3),
            candidate("simple", (0.5, 0.5), 1),
        ]
    )
    payload = result.model_dump(mode="json")
    payload["summaries"][0]["standard_error"] = 0
    with pytest.raises(ValueError, match="evidence"):
        m.SelectionResult.model_validate(payload)
    payload = result.model_dump(mode="json")
    payload["candidates"].reverse()
    with pytest.raises(ValueError, match="canonical"):
        m.SelectionResult.model_validate(payload)


def test_currency_metrics_are_independent_of_ambient_decimal_precision() -> None:
    rows = [alert("a", 1, True, "1"), alert("b", 0, False, "2", flagged=False)]
    with localcontext() as context:
        context.prec = 6
        result = m.alert_metrics(rows, alert_window())
        pnl = m.valuation_metrics(
            [valuation(supported_gross_value=Decimal("1"), total_gross_value=Decimal("3"))]
        )
    assert result.loss_captured_at_budget == Decimal("0.3333333333333333333333333333")
    assert pnl.supported_value_share == Decimal("0.3333333333333333333333333333")


@pytest.mark.parametrize(
    "scores,mean,standard_error",
    [
        ((-1e200, 1e200), 0, 1e200),
        ((-sys.float_info.max, sys.float_info.max), 0, sys.float_info.max),
        ((sys.float_info.max, sys.float_info.max), sys.float_info.max, 0),
        ((0, 0), 0, 0),
    ],
)
def test_large_finite_selection_scores_have_finite_scale_safe_standard_error(
    scores, mean, standard_error
) -> None:
    result = m.select_within_one_standard_error([candidate("large", scores, 1)])
    assert result.best_mean == pytest.approx(mean)
    assert result.best_standard_error == pytest.approx(standard_error)


@pytest.mark.parametrize("direction,sign", [("maximize", 1), ("minimize", -1)])
def test_finite_selection_scores_scale_before_centering_and_restore(direction, sign) -> None:
    folds = (
        *INNER,
        InnerFold(
            fold_id="inner-3",
            train=(group("c", 2),),
            embargo=(),
            validation=(group("d", 3),),
        ),
    )
    maximum = sys.float_info.max
    scores = (-sign * maximum, sign * maximum, sign * maximum)
    values = [
        candidate(
            identity,
            (),
            complexity,
            direction=direction,
            fold_results=tuple(
                m.InnerDevelopmentResult(fold=fold, score=score)
                for fold, score in zip(folds, fold_scores, strict=True)
            ),
        )
        for identity, complexity, fold_scores in [("large", 1, scores), ("simple", 0, (0, 0, 0))]
    ]
    result = m.select_within_one_standard_error(values)
    assert result.best_mean == pytest.approx(sign * 5.992310449541052e307)
    assert result.best_standard_error == pytest.approx(1.1984620899082105e308)
    assert result.eligibility_threshold == pytest.approx(-sign * 5.992310449541052e307)
    assert result.best_candidate_id == "large"
    assert result.selected_candidate_id == "simple"
    assert result.eligible_candidate_ids == ("large", "simple")
    assert all(
        math.isfinite(value)
        for summary in result.summaries
        for value in (summary.mean_score, summary.standard_error)
    )
    reordered = m.select_within_one_standard_error(values[::-1])
    assert reordered.model_dump_json() == result.model_dump_json()
    assert m.SelectionResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "factory,record",
    [
        (scenario, "ScenarioObservation"),
        (valuation, "ValuationObservation"),
    ],
)
def test_case_observation_boundaries_require_aware_time(factory, record: str) -> None:
    payload = factory().model_dump()
    payload.pop("occurred_at", None)
    with pytest.raises(ValueError):
        getattr(m, record).model_validate(payload)
    payload["occurred_at"] = START.replace(tzinfo=None)
    with pytest.raises(ValueError):
        getattr(m, record).model_validate(payload)


def test_valuation_requires_explicit_comparison_scope_and_rejects_mixed_scopes() -> None:
    payload = valuation().model_dump()
    payload.pop("comparison_scope", None)
    with pytest.raises(ValueError):
        m.ValuationObservation.model_validate(payload)
    with pytest.raises(ValueError, match="scope"):
        m.valuation_metrics([valuation(), valuation("b", comparison_scope="different-positions")])


def test_alert_window_chronology_uses_utc_across_repeated_local_dst_hour() -> None:
    zone = ZoneInfo("America/New_York")
    start = datetime(2020, 11, 1, 1, 30, tzinfo=zone, fold=0)
    end = datetime(2020, 11, 1, 1, 30, tzinfo=zone, fold=1)
    window = alert_window(start=start, end=end)
    result = m.alert_metrics([alert("a", 1, True, "1", occurred_at=start)], window)
    assert result.alerts_per_day == 24


def dst_group(identity: str, fold: int) -> FoldGroup:
    return FoldGroup(
        cluster_id=identity,
        event_time=datetime(2020, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=fold),
        source_item_ids=(identity,),
    )


def dst_development_fold(identity: str, validation: FoldGroup) -> InnerFold:
    return InnerFold(
        fold_id=identity,
        train=(group(f"train-{identity}", 0),),
        embargo=(),
        validation=(validation,),
    )


def test_development_selection_rejects_reversed_utc_validation_blocks_in_repeated_hour() -> None:
    late = dst_development_fold("late", dst_group("a", 1))  # 06:30 UTC
    early = dst_development_fold("early", dst_group("z", 0))  # 05:30 UTC
    with pytest.raises(ValueError, match="chronological"):
        value = candidate(
            "a",
            (0.8, 0.8),
            1,
            fold_results=(
                m.InnerDevelopmentResult(fold=late, score=0.8),
                m.InnerDevelopmentResult(fold=early, score=0.8),
            ),
        )
        m.select_within_one_standard_error([value])


def test_development_selection_accepts_true_utc_order_and_restores_across_repeated_hour() -> None:
    early = dst_development_fold("early", dst_group("z", 0))  # 05:30 UTC
    late = dst_development_fold("late", dst_group("a", 1))  # 06:30 UTC
    value = candidate(
        "a",
        (0.8, 0.8),
        1,
        fold_results=(
            m.InnerDevelopmentResult(fold=early, score=0.8),
            m.InnerDevelopmentResult(fold=late, score=0.8),
        ),
    )
    result = m.select_within_one_standard_error([value])
    assert result.selected_candidate_id == "a"
    assert result.candidates[0].fold_results[0].fold.validation[0].event_time == datetime(
        2020,
        11,
        1,
        5,
        30,
        tzinfo=timezone.utc,
    )
    assert result.candidates[0].fold_results[1].fold.validation[0].event_time == datetime(
        2020,
        11,
        1,
        6,
        30,
        tzinfo=timezone.utc,
    )
    assert m.SelectionResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("partition", ["train", "embargo", "validation", "boundary"])
def test_development_result_rejects_reversed_utc_order_within_supplied_partitions(
    partition,
) -> None:
    payload = dict(
        fold_id="dst",
        train=(group("train", 0),),
        embargo=(),
        validation=(group("validation", 366),),
    )
    reversed_groups = (dst_group("a", 1), dst_group("z", 0))
    if partition == "boundary":
        payload.update(train=(reversed_groups[0],), validation=(reversed_groups[1],))
    else:
        payload[partition] = reversed_groups
    with pytest.raises(ValueError, match="chronological"):
        m.InnerDevelopmentResult(fold=payload, score=0.8)


def test_development_result_accepts_true_utc_partition_order_and_json_restores() -> None:
    payload = dict(
        fold_id="dst",
        train=(dst_group("z", 0),),
        embargo=(),
        validation=(dst_group("a", 1),),
    )
    value = m.InnerDevelopmentResult(fold=payload, score=0.8)
    assert m.InnerDevelopmentResult.model_validate_json(value.model_dump_json()) == value
