# Implementation decisions

- Existing local UCI `day.csv` is the source. Avoid a redundant download; record its checksum and attribution.
- The host offers Python 3.13 rather than 3.12. Development and the Docker image use Python 3.13 with the measured, pinned dependency versions. This is a documented runtime deviation from the plan.
- Use a JSON metadata manifest and a joblib model payload for the historical release. SQLite stores only the independent replay.
- Use the plan's exact partitions, candidates, parameters, and selection rule without test-driven changes.
