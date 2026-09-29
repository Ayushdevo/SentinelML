# Security Policy

## Supported code

Security fixes should target the current default branch.

## Reporting a vulnerability

Please do not open a public issue for vulnerabilities that expose credentials, private datasets, model artifacts containing sensitive information, or a remotely exploitable API weakness.

When reporting a security issue, include:

- affected component or endpoint
- reproduction steps
- impact
- relevant environment details
- a suggested mitigation, if known

Do not include real secrets, access tokens, private customer data, or credentials in reports, tests, screenshots, or logs.

## ML-specific security notes

SentinelML consumes model and data artifacts from local paths. Treat those artifacts as untrusted unless their provenance is known. In production deployments, validate artifact sources, restrict filesystem permissions, and avoid exposing internal model-health reports without appropriate access controls.
