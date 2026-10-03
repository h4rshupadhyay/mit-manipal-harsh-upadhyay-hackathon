# Financial Risk Intelligence

This context defines the language used by the prototype that turns unstructured market information into explainable portfolio stress scenarios.

## Language

**Source Item**:
A timestamped piece of unstructured text received from a named news or social-media source, together with its provenance.
_Avoid_: Feed event, raw event

**Risk Signal**:
A structured, entity-linked interpretation of a Source Item containing sentiment, event classification, expected market severity, confidence, provenance, and a human-readable rationale.
_Avoid_: Prediction, alert

**Event Class**:
One of eight stable top-level risk categories: Geopolitical, Macroeconomic/Monetary, Credit/Default, Regulatory/Legal, Operational/Cyber, Corporate Action, Climate/Natural Disaster, or Other/Uncertain. More specific child labels may be retained without changing the top-level contract.
_Avoid_: Topic, arbitrary generated label

**Impact Score**:
A 1–10 severity decile derived by applying matched historical joint market shocks to a fixed Reference Basket and ranking the expected loss against a frozen historical distribution. It is conditional on the event interpretation being credible and remains independent of model Confidence and the current Synthetic Portfolio.
_Avoid_: Confidence score, probability

**Reference Basket**:
A fixed, versioned collection of representative factor exposures used to convert historical joint market shocks into a portfolio-independent Market Impact Score.
_Avoid_: Synthetic Portfolio, benchmark index

**Abnormal Return**:
The return of an exposed security or market segment after removing the return attributable to its selected benchmark over the same period.
_Avoid_: Raw return, portfolio loss

**Historical Analogue**:
A past event matched to a new Risk Signal by event class, region, exposure type, and other measurable modifiers, with associated post-event market movements. The association does not by itself establish that the event caused those movements.
_Avoid_: Training example, scenario template

**Confidence**:
A calibrated probability that the engine linked the correct entity and assigned the correct Event Class to a Source Item. It is kept separate from expected market severity and analogue support.
_Avoid_: Impact, severity

**Synthetic Portfolio**:
A fictional collection of wholesale-banking loans, bonds, and derivatives with explicit sector, geography, rating, maturity, and market-factor exposures.
_Avoid_: Customer portfolio, transaction dataset

**Portfolio Materiality**:
The stressed loss attributable to a Risk Signal on the current Synthetic Portfolio, expressed in currency and as a percentage of portfolio value. Automatic action compares it with a configured economic floor rather than testing whether it is merely nonzero.
_Avoid_: Impact Score, Confidence

**Action Priority**:
A portfolio-specific ordering signal derived from Market Impact Score, Portfolio Materiality, and Confidence. It determines analyst attention but is not exposed as the required market-impact field.
_Avoid_: Impact Score, automatic decision

**Joint Shock Vector**:
The observed or hypothetical set of simultaneous equity, rate, spread, FX, commodity, and volatility movements for one scenario, with explicit units and horizon.
_Avoid_: Independently combined worst-case shocks

**Stress Scenario**:
A Joint Shock Vector selected in response to a Risk Signal and applied consistently to the Reference Basket or Synthetic Portfolio.
_Avoid_: Risk Signal, event

**Stress Test**:
The calculation that applies a Stress Scenario to the Synthetic Portfolio and attributes changes in value to assets, sectors, geographies, and risk factors.
_Avoid_: Backtest, forecast

**Backtest**:
A temporally out-of-sample evaluation of whether historical Risk Signals ranked subsequently realized abnormal market movements and selected appropriate factor shocks without using future information.
_Avoid_: Stress Test, model training
