# Task 1 CHANGELOG: numerical stability and configuration
(Static edits only; nothing was run or tested.)

## benchmark_config.py
- New module constant `GRAD_CLIP_NORM = 1.0` (global L2 norm; `None` disables).
- `TrainConfig.grad_clip_norm` defaults to it. `ANN_TRAIN`, `RSNN_TRAIN`, `RNN_TRAIN` inherit it, so ann, rsnn, rnn,
  rsnn_wide and rnn_wide share one threshold (wide models reuse the same train configs). Recorded in config.json.

## training.py
- `fit(..., grad_clip_norm)`: `clip_grad_norm_` between `backward()` and `optimizer.step()`.
- `pretrain_then_finetune` passes `cfg.grad_clip_norm` to both pretraining and fine-tuning (the only training path).
- New `NonFiniteTrainingError`. Once per epoch (no per-step host sync) train loss, validation loss and max pre-clip
  gradient norm are checked; on NaN/Inf it raises with the tag (model, subject, repeat, phase), epoch, losses, grad norm,
  lr, dropout, batch size, seed and the last 3 finite epochs. Nothing is replaced or skipped.
- History entries gain `grad_norm_max` and `clip_fraction` (existing keys unchanged; plotting code unaffected).
- `predict_logits` raises on non-finite logits instead of letting argmax mask them.

## ann_to_snn.py
- Conversion raises if any calibration scale (lambda) is non-finite. The SNN itself is not trained, so no clipping applies.

## Unchanged
Architectures, preprocessing, folds, epochs, lr, Adam, early stopping, loss, conversion method, evaluation.
Note: clipping can slightly change ANN/RSNN/RNN trajectories versus the earlier run (permitted).
