# Contributing to SentinelML

Thanks for contributing to SentinelML.

## Development setup

1. Create and activate a virtual environment.
2. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

3. Run the test suite:

```bash
pytest -q
```

4. Run the monitoring pipeline when the required data/model artifacts are available:

```bash
python -m src.run_pipeline
```

## Contribution scope

Good contributions include:

- drift and data-quality checks
- prediction-risk validation
- tests for monitoring edge cases
- API reliability improvements
- clearer operational documentation

Keep changes focused. Prefer one logical change per pull request, add tests for behavior changes, and avoid committing generated model/data artifacts unless they are intentionally part of the change.

## Validation

Before submitting a change:

- run `pytest -q`
- verify imports work from the repository root
- confirm monitoring thresholds or scoring changes are documented
- avoid silently changing the meaning of existing report fields

## Reporting bugs

Include the command you ran, the observed output, the expected behavior, and a minimal reproducible example when possible.
