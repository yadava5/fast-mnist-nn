## Summary
<!-- One or two sentences. What changes? -->

Closes #<issue>

## Why
<!-- The underlying problem or goal. Link to ADR/RFC if applicable. -->

## Changes
<!-- Bulleted list of concrete changes per file or per module. -->

## Test plan
<!-- Exact commands run; results observed. Prefer reproducible repros. -->
- [ ] `ctest --test-dir build` passes
- [ ] `npm run build` + `npm run lint` pass
- [ ] New tests added for new behavior
- [ ] Benchmark numbers attached if a hot path changed

## Perf impact
<!-- If a kernel changed, paste before/after from `scripts/bench.sh` with median + MAD + CI. -->

## Screenshots / recordings
<!-- If UI changed. -->

## Breaking changes
<!-- API, CLI, file format, HTTP contract. "None" if none. -->

## Rollback plan
<!-- How to revert. -->

## Checklist
- [ ] Conventional Commit title
- [ ] CI green (CodeQL, sanitizers, perf, coverage)
- [ ] CodeRabbit approval (optional for now)
- [ ] Docs updated (README, ADR, or CHANGELOG)
- [ ] No secrets, PII, or large binaries added
