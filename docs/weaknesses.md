# Weaknesses and open issues

This page is completed from the final results; the items below are known *a priori* and are
quantified with MLflow numbers once the runs exist.

* **Nine subjects, little diversity.** One person in test means a single subject's habits can
  swing the test score; LOSO quantifies this spread.
* **Lab protocol, not free living.** Activities were performed in a fixed order and setting.
* **Determinism is tied to one GPU type** (`rtx2080ti`) and one software stack.
* **Interim MLflow store.** SQLite on a shared network disk; concurrent writers are retried but
  not ideal. To be migrated to the university server.
