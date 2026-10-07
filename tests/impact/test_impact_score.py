"""Hand-calculated fixtures exercise algorithms, not empirical calibration quality."""

import hashlib
import json
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal, localcontext

import pytest

from risk_engine.domain import MarketObservation, MarketSnapshot, ShockType, ShockUnit
from risk_engine.impact.analogues import AnalogueRepository
from risk_engine.impact.module import (
    FactorBinding,
    FrozenImpactCalibration,
    GovernedHypothetical,
    ImpactEstimator,
    TrainingEvidence,
)
from risk_engine.impact.reference_basket import BasketConstruction, ReferenceBasketBuilder
from tests.impact.test_analogues import AS_OF, PAST, attributes, config, event, historical
from tests.impact.test_reference_basket import CALIBRATION, rows, spec


def market():
    return MarketSnapshot(
        snapshot_id="fixture-market",
        as_of=CALIBRATION,
        provider="fixture",
        schema_version="v1",
        normalization_version="v1",
        observations=tuple(
            MarketObservation(
                factor_id=factor,
                observed_at=CALIBRATION,
                value=100,
                unit=ShockUnit.INDEX_POINT,
                provider="fixture",
                vintage="v1",
            )
            for factor in ("EQUITY-US", "EQUITY-INDIA")
        ),
    )


def bindings():
    common = tuple(
        FactorBinding(
            historical_factor_id=source,
            valuation_factor_id=target,
            measurement_dimension=dimension,
            source_unit=unit,
            shock_type=kind,
            valuation_unit=unit,
            value_multiplier=Decimal(1),
        )
        for source, target, dimension, unit, kind in (
            ("equity", "EQUITY-US", "return", ShockUnit.DECIMAL, ShockType.RELATIVE),
            ("rate", "RATE-USD", "yield", ShockUnit.BASIS_POINT, ShockType.BASIS_POINT),
            ("spread", "CREDIT-SPREAD", "spread", ShockUnit.BASIS_POINT, ShockType.BASIS_POINT),
            ("fx", "EUR-USD", "currency", ShockUnit.PERCENT, ShockType.RELATIVE),
            ("commodity", "COMMODITY", "price", ShockUnit.PERCENT, ShockType.RELATIVE),
            (
                "volatility",
                "VOLATILITY",
                "volatility",
                ShockUnit.VOLATILITY_POINT,
                ShockType.VOLATILITY,
            ),
        )
    )
    return (
        *common,
        FactorBinding(
            historical_factor_id="equity-india",
            valuation_factor_id="EQUITY-INDIA",
            measurement_dimension="return",
            source_unit=ShockUnit.DECIMAL,
            shock_type=ShockType.RELATIVE,
            valuation_unit=ShockUnit.DECIMAL,
            value_multiplier=Decimal(1),
        ),
    )


def analogue(event_id, loss):
    record = historical(event_id)
    data = record.model_dump()
    equity = data["scenario"]["windows"][0]["shocks"][0]
    equity["value"] = -float(loss) / 1000
    india = dict(equity, factor_id="equity-india")
    india["source_observations"] = tuple(
        dict(row, series_id="equity-india") for row in equity["source_observations"]
    )
    data["scenario"]["windows"][0]["shocks"] = (
        *data["scenario"]["windows"][0]["shocks"],
        india,
    )
    data["factor_roles"] = (*data["factor_roles"], {"factor_id": "equity-india", "role": "equity"})
    return type(record).model_validate(data)


def calibration(**changes):
    basket = ReferenceBasketBuilder().build(rows(), spec(BasketConstruction.EQUAL_NOTIONAL))
    data = dict(
        version="impact-v1",
        calibrated_at=AS_OF - timedelta(days=1),
        training_start=PAST - timedelta(days=1),
        training_end=PAST + timedelta(days=1),
        reference_basket=basket,
        window={"start": 0, "end": 0},
        factor_bindings=bindings(),
        quantile_convention="nearest-rank-lower-ties",
        uncertainty_convention="observed-min-max",
        source_terms="synthetic test fixtures",
        training=tuple(
            TrainingEvidence(
                analogue=analogue(f"train-{i}", i * 10), base_market=market(), partition="training"
            )
            for i in range(1, 11)
        ),
    )
    data.update(changes)
    return FrozenImpactCalibration.model_validate(data)


def repository(loss=40):
    return AnalogueRepository(
        tuple(analogue(f"matched-{i}", loss) for i in range(3)),
        config(),
        (attributes("query"),),
    )


def estimator(loss=40, **options):
    return ImpactEstimator(repository(loss), calibration(), market(), **options)


def test_revalues_complete_vectors_and_reports_median_loss():
    repository = AnalogueRepository(
        tuple(analogue(f"matched-{i}", loss) for i, loss in enumerate((20, 40, 90))),
        config(),
        (attributes("query"),),
    )
    estimate = ImpactEstimator(repository, calibration(), market()).estimate(event(), AS_OF)
    assert estimate.expected_reference_loss == Decimal("40")
    assert estimate.loss_lower_bound == Decimal("20")
    assert estimate.loss_upper_bound == Decimal("90")
    assert estimate.impact_score == 4
    assert estimate.analogue_count == 3
    assert estimate.method.value == "empirical"


def test_audit_retains_loss_population_and_pool_support_without_source_text():
    result = estimator().estimate(event(), AS_OF)
    audit = json.loads(result.calibration_version)
    assert list(map(Decimal, audit["cutpoints"])) == list(map(Decimal, range(10, 91, 10)))
    assert audit["support_count"] == 3
    assert audit["matching_validation_status"] == "bootstrap_unvalidated"
    assert [Decimal(row["loss"]) for row in audit["training_losses"]] == list(
        map(Decimal, range(10, 101, 10))
    )
    assert "training" not in audit["calibration"]
    assert len(audit["audit_sha256"]) == 64
    digest = audit.pop("audit_sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_calibration_summary_matches_actual_impact_audit():
    fitted = estimator()
    summary = fitted.calibration_summary
    result = fitted.estimate(event(), AS_OF)
    audit = json.loads(result.calibration_version)

    assert summary.calibration_version == audit["calibration"]["version"]
    assert summary.calibration_hash == audit["calibration_sha256"]
    assert summary.reference_basket_version == result.reference_basket_version
    assert summary.reference_basket_hash == audit["calibration"]["reference_basket_sha256"]
    assert summary.matching_version == audit["matching_version"]
    assert summary.training_event_ids == tuple(row["event_id"] for row in audit["training_losses"])
    assert summary.cutpoints == tuple(Decimal(point) for point in audit["cutpoints"])
    assert summary.currency == result.loss_currency == "USD"
    assert summary.horizon_days == 1
    assert summary.quantile_convention == "nearest-rank-lower-ties"
    assert summary.frozen_at.isoformat() == audit["calibration"]["calibrated_at"].replace(
        "Z", "+00:00"
    )
    object.__setattr__(summary, "cutpoints", ())
    assert fitted.calibration_summary.cutpoints == tuple(
        Decimal(point) for point in audit["cutpoints"]
    )


@pytest.mark.parametrize(
    ("loss", "score"), [(0, 1), (10, 1), (11, 2), (90, 9), (100, 10), (200, 10), (-20, 1)]
)
def test_decile_extremes_and_lower_bin_ties(loss, score):
    assert estimator(loss).estimate(event(), AS_OF).impact_score == score


def test_thin_history_abstains_without_governed_fallback():
    repository = AnalogueRepository((), config(), ())
    with pytest.raises(ValueError, match="governed hypothetical"):
        ImpactEstimator(repository, calibration(), market()).estimate(event(), AS_OF)


@pytest.mark.parametrize(
    "changes",
    [
        {"calibrated_at": PAST},
        {"training_end": PAST},
        {"training_start": CALIBRATION - timedelta(days=1)},
        {
            "training": (
                TrainingEvidence(
                    analogue=historical("synthetic", evidence_kind="synthetic"),
                    base_market=market(),
                    partition="training",
                ),
            )
        },
    ],
)
def test_rejects_nontraining_or_unavailable_calibration_evidence(changes):
    with pytest.raises(ValueError):
        calibration(**changes)


def test_validation_partition_cannot_fit_cutpoints():
    with pytest.raises(ValueError):
        TrainingEvidence(
            analogue=analogue("held-out", 100), base_market=market(), partition="validation"
        )


def test_training_and_scoring_must_share_frozen_market_content():
    changed = market().model_copy(update={"snapshot_id": "different"})
    with pytest.raises(ValueError, match="frozen base market"):
        ImpactEstimator(repository(), calibration(), changed)
    training = list(calibration().training)
    training[1] = training[1].model_copy(update={"base_market": changed})
    with pytest.raises(ValueError, match="frozen base market"):
        calibration(training=tuple(training))


def test_scoring_timestamp_and_training_overlap_are_rejected():
    fitted = estimator()
    with pytest.raises(ValueError, match="timezone-aware"):
        fitted.estimate(event(), AS_OF.replace(tzinfo=None))
    with pytest.raises(ValueError, match="calibration unavailable"):
        fitted.estimate(event(), AS_OF - timedelta(days=1))
    with pytest.raises(ValueError, match="overlaps"):
        fitted.estimate(event().model_copy(update={"event_id": "train-1"}), AS_OF)


def test_unknown_factor_and_wrong_conversion_reject_atomically():
    for changes in ({"valuation_factor_id": "unknown"}, {"value_multiplier": Decimal(100)}):
        with pytest.raises(ValueError):
            FactorBinding.model_validate(dict(bindings()[0].model_dump(), **changes))


def test_current_holdings_and_confidence_do_not_change_score():
    fitted = estimator()
    original = fitted.estimate(event(), AS_OF)
    changed = fitted.estimate(event().model_copy(update={"classification_confidence": 0.1}), AS_OF)
    assert changed == original
    # Current holdings have no input seam. Mutating a caller's basket after
    # injection must not mutate the estimator's frozen internal Reference Basket.
    supplied = calibration()
    isolated = ImpactEstimator(repository(), supplied, market())
    supplied.reference_basket.positions[0].factor_exposures["EQUITY-US"] = 999
    assert isolated.estimate(event(), AS_OF) == original


def test_governed_hypothetical_is_labeled_and_retains_actual_support():
    from risk_engine.domain import FactorShock, ProvenanceMethod, StressScenario

    fallback = GovernedHypothetical(
        scenario=StressScenario(
            scenario_id="fallback",
            risk_signal_id="query",
            method=ProvenanceMethod.HYPOTHETICAL,
            calibration_version="governed-v1",
            shocks=tuple(
                FactorShock(
                    factor_id=factor,
                    shock_type=ShockType.RELATIVE,
                    value=-0.08,
                    unit=ShockUnit.DECIMAL,
                    horizon_days=1,
                )
                for factor in ("EQUITY-US", "EQUITY-INDIA")
            ),
        ),
        base_market=market(),
        window={"start": 0, "end": 0},
        available_at=AS_OF - timedelta(days=1),
        governance_version="governance-v1",
        source_terms="project-authored hypothetical",
        source_sha256="a" * 64,
    )
    repository = AnalogueRepository((analogue("thin", 40),), config(), ())
    result = ImpactEstimator(repository, calibration(), market(), fallback).estimate(event(), AS_OF)
    assert result.method is ProvenanceMethod.HYPOTHETICAL
    assert result.expected_reference_loss == Decimal(80)
    assert result.impact_score == 8
    assert result.analogue_count == 0
    assert result.analogue_ids == ()
    assert json.loads(result.calibration_version)["support_count"] == 1
    audit = json.loads(result.calibration_version)
    assert Decimal(audit["hypothetical_loss"]["loss"]) == Decimal(80)
    assert "stress-engine-v2" in audit["hypothetical_loss"]["valuation_rule_version"]


def test_incomplete_basket_manifest_cannot_hide_missing_source_metadata():
    original = calibration().reference_basket
    manifest = json.loads(original.version)
    del manifest["return_snapshot_id"]
    with pytest.raises(ValueError, match="manifest"):
        calibration(reference_basket=original.model_copy(update={"version": json.dumps(manifest)}))


def test_even_median_is_independent_of_callers_decimal_precision():
    repository = AnalogueRepository(
        tuple(analogue(f"matched-{i}", loss) for i, loss in enumerate((11, 12, 13, 14))),
        config(),
        (attributes("query"),),
    )
    fitted = ImpactEstimator(repository, calibration(), market())
    with localcontext() as context:
        context.prec = 1
        result = fitted.estimate(event(), AS_OF)
    assert result.expected_reference_loss == Decimal("12.5")


def test_incomplete_joint_mapping_is_rejected_instead_of_zero_loss():
    with pytest.raises(ValueError, match="entire joint vector"):
        ImpactEstimator(repository(), calibration(factor_bindings=bindings()[:-1]), market())


def test_incomplete_basket_exposure_is_rejected_instead_of_zero_loss():
    original = calibration().reference_basket
    data = original.model_dump()
    data["positions"][0]["factor_exposures"] = {"EQUITY-EUROPE": 1}
    with pytest.raises(ValueError, match="factor coverage"):
        ImpactEstimator(
            repository(),
            calibration(reference_basket=type(original).model_validate(data)),
            market(),
        )


def test_partial_supported_value_share_is_rejected():
    original = calibration().reference_basket
    data = original.model_dump()
    data["positions"][1]["asset_type"] = "loan"
    with pytest.raises(ValueError, match="valuation coverage"):
        ImpactEstimator(
            repository(),
            calibration(reference_basket=type(original).model_validate(data)),
            market(),
        )


def test_registered_window_must_exist_in_every_training_vector():
    with pytest.raises(ValueError, match="registered event window"):
        ImpactEstimator(repository(), calibration(window={"start": 0, "end": 1}), market())


def test_revaluation_preserves_historical_joint_source_bundle():
    record = analogue("matched", 40)
    before = record.model_dump_json()
    repository = AnalogueRepository(
        (record, analogue("two", 40), analogue("three", 40)), config(), (attributes("query"),)
    )
    ImpactEstimator(repository, calibration(), market()).estimate(event(), AS_OF)
    assert record.model_dump_json() == before


def test_selected_count_is_separate_from_eligible_pool_support():
    matching = config().model_copy(update={"nearest_neighbors": 3})
    pool = AnalogueRepository(
        tuple(analogue(f"matched-{i}", 40) for i in range(5)), matching, (attributes("query"),)
    )
    result = ImpactEstimator(pool, calibration(), market()).estimate(event(), AS_OF)
    assert result.analogue_count == 3
    assert json.loads(result.calibration_version)["support_count"] == 5


def test_one_registered_window_is_used_without_selecting_factor_tails():
    record = analogue("multi", 40)
    data = record.model_dump()
    first = data["scenario"]["windows"][0]
    for shock in first["shocks"]:
        for field in ("source_observations", "source_benchmark_observations"):
            initial = shock[field][0]
            shock[field] = (
                initial,
                dict(
                    initial,
                    session_date=initial["session_date"] + timedelta(days=1),
                    observed_at=initial["observed_at"] + timedelta(days=1),
                ),
            )
    second = deepcopy(first)
    second["window"] = {"start": 0, "end": 1}
    for shock in second["shocks"]:
        shock["value"] = -0.9 if shock["factor_id"].startswith("equity") else 10
    data["scenario"]["windows"] = (first, second)
    complete = PAST + timedelta(days=1, hours=4)
    data["window_evidence"] = (
        *data["window_evidence"],
        dict(
            window=second["window"], sessions=(PAST.date(), complete.date()), completed_at=complete
        ),
    )
    data["source_evidence"][0]["available_at"] = complete
    data["available_at"] = complete + timedelta(hours=1)
    multi = type(record).model_validate(data)
    pool = AnalogueRepository(
        (multi, analogue("two", 40), analogue("three", 40)), config(), (attributes("query"),)
    )
    result = ImpactEstimator(pool, calibration(), market()).estimate(event(), AS_OF)
    assert result.expected_reference_loss == Decimal(40)
    assert result.loss_upper_bound == Decimal(40)


def test_relative_fx_and_commodity_returns_revalue_complete_upstream_vectors():
    original = calibration()
    mapped = []
    for binding in original.factor_bindings:
        data = binding.model_dump()
        if binding.historical_factor_id in {"fx", "commodity"}:
            data["measurement_dimension"] = "return"
        mapped.append(FactorBinding.model_validate(data))

    def returns(record):
        data = record.model_dump()
        for shock in data["scenario"]["windows"][0]["shocks"]:
            if shock["factor_id"] in {"fx", "commodity"}:
                shock["measurement_dimension"] = "return"
                for field in ("source_observations", "source_benchmark_observations"):
                    for row in shock[field]:
                        row["measurement_dimension"] = "return"
        return type(record).model_validate(data)

    training = tuple(
        TrainingEvidence(
            analogue=returns(row.analogue), base_market=row.base_market, partition="training"
        )
        for row in original.training
    )
    pool = AnalogueRepository(
        tuple(returns(analogue(f"matched-{i}", 40)) for i in range(3)),
        config(),
        (attributes("query"),),
    )
    result = ImpactEstimator(
        pool, calibration(factor_bindings=tuple(mapped), training=training), market()
    ).estimate(event(), AS_OF)
    assert result.expected_reference_loss == Decimal(40)
    assert result.impact_score == 4


@pytest.mark.parametrize(
    ("factor", "unit"), [("EUR-USD", ShockUnit.CURRENCY), ("COMMODITY", ShockUnit.ABSOLUTE)]
)
def test_absolute_fx_and_commodity_cannot_be_mislabeled_returns(factor, unit):
    with pytest.raises(ValueError, match="measurement dimension"):
        FactorBinding(
            historical_factor_id="series",
            valuation_factor_id=factor,
            measurement_dimension="return",
            source_unit=unit,
            valuation_unit=unit,
            shock_type=ShockType.ABSOLUTE,
            value_multiplier=Decimal(1),
        )


@pytest.mark.parametrize("notionals", [("1000", "500"), ("600", "400")])
def test_basket_manifest_rejects_total_and_same_total_allocation_tampering(notionals):
    basket = calibration().reference_basket
    data = basket.model_dump()
    for position, notional in zip(data["positions"], notionals, strict=True):
        position["notional"] = Decimal(notional)
    with pytest.raises(ValueError, match="manifest"):
        calibration(reference_basket=type(basket).model_validate(data))


@pytest.mark.parametrize("construction", list(BasketConstruction))
def test_all_registered_basket_allocations_validate_and_reject_redistribution(construction):
    basket = ReferenceBasketBuilder().build(rows(), spec(construction))
    frozen = calibration(reference_basket=basket)
    result = ImpactEstimator(repository(), frozen, market()).estimate(event(), AS_OF)
    assert result.expected_reference_loss == Decimal(40)
    assert result.impact_score == 4
    data = basket.model_dump()
    data["positions"][0]["notional"] += Decimal(1)
    data["positions"][1]["notional"] -= Decimal(1)
    with pytest.raises(ValueError, match="manifest"):
        calibration(reference_basket=type(basket).model_validate(data))
    with localcontext() as context:
        context.prec = 1
        # Validate an existing immutable artifact under hostile caller arithmetic.
        assert FrozenImpactCalibration.model_validate(frozen.model_dump()) == frozen


def test_actual_high_precision_builder_output_can_be_revalued_without_rounding_guess():
    candidate = spec(BasketConstruction.INVERSE_VOLATILITY).model_copy(
        update={"total_notional": Decimal("100000000000000.01")}
    )
    with localcontext() as context:
        context.prec = 50
        basket = ReferenceBasketBuilder().build(rows(), candidate)
    fitted = ImpactEstimator(repository(), calibration(reference_basket=basket), market())
    result = fitted.estimate(event(), AS_OF)
    assert result.expected_reference_loss == Decimal("4000000000000.0004")


@pytest.mark.parametrize("declared", [None, "unknown-arithmetic-v9"])
def test_undeclared_or_unsupported_basket_arithmetic_is_rejected(declared):
    basket = calibration().reference_basket
    manifest = json.loads(basket.version)
    if declared is None:
        manifest.pop("allocation_arithmetic_version", None)
    else:
        manifest["allocation_arithmetic_version"] = declared
    with pytest.raises(ValueError, match="manifest"):
        calibration(reference_basket=basket.model_copy(update={"version": json.dumps(manifest)}))


def frozen_erc_calibration():
    basket = ReferenceBasketBuilder().build(
        rows(), spec(BasketConstruction.EQUAL_RISK_CONTRIBUTION)
    )
    return calibration(reference_basket=basket)


@pytest.mark.parametrize("precision", [3, 28, 50])
def test_frozen_erc_parse_and_valuation_never_optimize_or_rewrite(precision, monkeypatch):
    import risk_engine.impact.module as impact_module
    import risk_engine.impact.reference_basket as basket_module

    original = frozen_erc_calibration()
    before = original.model_dump_json()
    expected = ImpactEstimator(repository(), original, market()).calibration_summary

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen validation attempted ERC optimization")

    monkeypatch.setattr(basket_module, "_equal_risk_weights", forbidden)
    monkeypatch.setattr(basket_module, "minimize", forbidden)
    monkeypatch.setattr(impact_module, "_equal_risk_weights", forbidden, raising=False)
    with localcontext() as context:
        context.prec = precision
        restored = FrozenImpactCalibration.model_validate_json(before)
        actual = ImpactEstimator(repository(), restored, market()).calibration_summary
    assert actual == expected
    assert restored.model_dump_json() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "redistribution",
        "nonfinite",
        "asymmetric",
        "indefinite",
        "zero_variance",
        "negative_weight",
        "wrong_total",
    ],
)
def test_frozen_erc_direct_validation_rejects_invalid_math(mutation, monkeypatch):
    import risk_engine.impact.module as impact_module

    original = frozen_erc_calibration()
    data = original.model_dump()
    basket = data["reference_basket"]
    manifest = json.loads(basket["version"])
    if mutation == "redistribution":
        basket["positions"][0]["notional"] += Decimal(100)
        basket["positions"][1]["notional"] -= Decimal(100)
    elif mutation == "nonfinite":
        manifest["covariance"][0][0] = float("nan")
    elif mutation == "asymmetric":
        manifest["covariance"][0][1] *= 2
    elif mutation == "indefinite":
        # Symmetric positive diagonal, but a negative eigenvalue: invalid sample covariance.
        manifest["covariance"] = [[1.0, 2.0], [2.0, 1.0]]
    elif mutation == "zero_variance":
        manifest["covariance"] = [[0.0, 0.0], [0.0, 0.0]]
    elif mutation == "negative_weight":
        basket["positions"][0]["notional"] = Decimal(-1)
        basket["positions"][1]["notional"] = Decimal(1001)
    else:
        basket["positions"][0]["notional"] += Decimal(1)
    basket["version"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid frozen input reached optimizer")

    monkeypatch.setattr(impact_module, "_equal_risk_weights", forbidden, raising=False)
    with pytest.raises(ValueError):
        FrozenImpactCalibration.model_validate(data)
    assert not calls


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_frozen_erc_accepts_singular_scaled_covariance_and_tiny_valid_redistribution(
    scale, monkeypatch
):
    import risk_engine.impact.module as impact_module

    original = frozen_erc_calibration()
    data = original.model_dump()
    manifest = json.loads(data["reference_basket"]["version"])
    manifest["covariance"] = [[value * scale for value in row] for row in manifest["covariance"]]
    # This is explicitly allowed mathematical tolerance, not bitwise solver reproduction.
    data["reference_basket"]["positions"][0]["notional"] += Decimal("0.00001")
    data["reference_basket"]["positions"][1]["notional"] -= Decimal("0.00001")
    data["reference_basket"]["version"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen validation attempted optimizer")

    monkeypatch.setattr(impact_module, "_equal_risk_weights", forbidden, raising=False)
    restored = FrozenImpactCalibration.model_validate(data)
    assert (
        restored.reference_basket.positions[0].notional
        == data["reference_basket"]["positions"][0]["notional"]
    )
