# Fix: Options Recommender (0 results) + AI Debate (schema echo)

## Context

After fixing the yfinance Invalid Crumb 401 errors, two pre-existing issues surfaced:

1. **Options recommender returns 0 recommendations** — The bid-ask spread filter (`max_bid_ask_spread_pct=0.10`) eliminates every contract. Materials sector stocks have wide spreads (15-40%+), so 0 of 23 contracts pass. The filter is too strict for less-liquid sectors.

2. **AI debate fails** — llama3.1:8b echoes the JSON schema definition back instead of generating data. The error: `input_value={'description': 'Response...', 'type': 'object'}`. The schema hint appended to the prompt confuses the 8B model.

## Fix 1: Relax Options Liquidity Filter

**File:** `src/option_alpha/config.py` (line 37)

- Change `max_bid_ask_spread_pct` default from `0.10` to `0.30` (30%)
  - 10% is unrealistic for anything outside mega-cap tech
  - 30% is a reasonable default for mid-cap options
  - Users can still override via config.json or env var

**File:** `config.json`

- Update the saved `max_bid_ask_spread_pct` value from `0.1` to `0.3`

**Tests:** No test changes needed — tests use explicit values, not defaults.

## Fix 2: Fix Ollama Structured Output Prompt

**File:** `src/option_alpha/ai/clients.py` (lines 139-151)

The problem: the schema hint includes the full Pydantic JSON schema with `description`, `type`, `properties` etc. llama3.1:8b treats this as the expected output and echoes it back.

**Fix:** Replace the full JSON schema dump with a concrete example of the expected output. Small models respond much better to examples than to abstract schemas.

Change the `OllamaClient.complete()` schema hint from:
```python
schema = response_model.model_json_schema()
schema_hint = (
    f"\n\nRespond with a JSON object matching this schema:\n"
    f"```json\n{json.dumps(schema, indent=2)}\n```"
)
```

To: generate a concrete example from the model's fields with placeholder values instead of dumping the abstract schema. This avoids the model echoing back `{"description": "...", "type": "object"}`.

For `AgentResponse`, the example would look like:
```json
{"role": "bull", "analysis": "Your analysis here", "key_points": ["point 1", "point 2"], "conviction": 7}
```

For `TradeThesis`:
```json
{"symbol": "TICKER", "direction": "bullish", "conviction": 7, "entry_rationale": "...", "risk_factors": ["risk 1"], "recommended_action": "Buy TICKER 100C 30DTE"}
```

**Implementation:** Add a helper `_build_example_hint(response_model)` that generates a simple example JSON from the model's field names, types, and defaults — not the full JSON schema.

**Tests:** Update `tests/test_ai.py` structured output test to verify the new hint format.

## Verification

1. `pytest tests/test_recommender.py tests/test_ai.py tests/test_chains.py tests/test_earnings.py -v` — all pass
2. `pytest tests/ -x -q` — full suite, no regressions
3. Run `python -m option_alpha` and verify:
   - Options phase produces >0 recommendations
   - AI debate phase produces structured responses (not schema echoes)
