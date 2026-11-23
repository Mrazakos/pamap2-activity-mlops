# Experiment plan and reasoning

Written **before** any model was trained. Each round changes one group of settings, keeps
everything else fixed, and selects on **validation macro-F1** only. The test subjects are
evaluated once, for the final models. Results and the decisions actually taken are added under
each round as the runs finish (numbers copied from MLflow, run ids linked).

Fixed for all rounds unless stated: `data-v2`, window 5.12 s with 50 % overlap, `all` sensors,
seed 42, AdamW, early stopping on val macro-F1 (patience 8, max 40 epochs), GPU type `rtx2080ti`.

## Baseline

* **Hypothesis.** Hand-crafted statistical and spectral features with a RandomForest already
  separate most activities; this is the bar the CNN has to beat.
* **Runs.** RandomForest `n_estimators ∈ {100, 300, 500}` × `max_depth ∈ {None, 20}` (cheap, CPU),
  plus LogisticRegression as a sanity check (a linear model on the same features).

## Round A: learning rate (1D CNN)

* **Hypothesis.** LR is the most sensitive optimiser setting; too high diverges or oscillates,
  too low under-fits within 40 epochs.
* **Runs.** `train.lr ∈ {1e-2, 1e-3, 3e-4, 1e-4}`, everything else fixed.
* **Next.** The best LR is fixed for all later rounds.
* **Status.** Submitted 2026-11-23 as Slurm job array 10466, MLflow parent run
  `35dcc4d8c7fd4eabb888adaaca0ba767`.

## Round B: window length, then capacity and regularisation

* **Hypothesis 1.** 5.12 s contains several gait cycles; 2.56 s gives twice as many windows and a
  quicker response but less context. Runs: `windows.window_seconds ∈ {2.56, 5.12}`.
* **Hypothesis 2.** With only 7 training subjects, the model over-fits subject idiosyncrasies;
  more dropout or fewer channels may help, and class weighting helps the rare classes.
  Runs: `model.dropout ∈ {0.3, 0.5}`, `model.conv_channels ∈ {[64,128,128], [32,64,64]}`,
  `train.class_weighting ∈ {true, false}` (chosen one at a time from the best so far).

## Round C: the smartwatch question

* **Hypothesis.** A wrist-only device loses most on posture (sitting/standing) and stairs, which
  the ankle and chest sensors capture; heart rate partly compensates for intensity.
* **Runs.** Best config with `data.sensor_subset ∈ {all, wrist_hr, wrist}` (+ `chest_hr` if budget allows).

## Additional model: CNN-GRU

* **Hypothesis.** A recurrent layer over the CNN feature sequence models the order of movement
  phases (e.g. stairs up vs down) better than global average pooling.
* **Runs.** `model.name=cnn_gru` with the best Round A/B settings, compared with the CNN.

## Data round: data-v1 vs data-v2

* **Hypothesis.** Naive cleaning (interpolating across long gaps, windows across activity
  boundaries with majority labels, saturating ±6 g channels and invalid orientation kept) hurts
  generalisation; the full pipeline should score higher on validation.
* **Runs.** The best config on `data.version=v1` and `v2` (v1 is fetched from its DVC tag).

## Final

* Best configuration promoted into `configs/params.yaml`, produced with `dvc repro`, evaluated
  once on the test subjects, then `verify-repro` for the final model and the baseline, and LOSO
  for both.
