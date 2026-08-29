# ADR: Recipe Mentor — Real Pipelines (MIMII Fetch, Features, Quantization)

Status: Implemented. Both `maize_train.py` and `mimii_anomaly.py` verified
end to end against real data, including a genuine (sub-chance, not tuned
away) MIMII result -- see the final section below, added after the fetch
completed and the full training run ran.
Scope: `recipe_mentor/pipelines/`.
Covers: 2026-08-29, continuing `ADR_Recipe_Mentor_Vertex_Backend_2026-08-29.md`.

## Objective

Give both project cards a real pipeline, not a described one: dataset in,
model trained, exported to ONNX, quantized statically to INT8, verified
against FP32 on held-out data, real numbers written back into the passport.
This ADR covers the decisions specific to getting real data and real
training working, following on from the passport/Vertex decisions already
recorded.

## Decision: fetch a bounded MIMII slice via HTTP range requests, not the full zip

MIMII's own Zenodo record (3384388) publishes one zip per machine
type/SNR. Even the smallest, `6_dB_valve.zip`, is 6.9GB, and contains all
four physical valve units bundled together. Downloading it whole for a
demo subset was the wrong move on both time and honesty grounds — a
"bounded subset" that starts by pulling 6.9GB isn't bounded.

Checked, not assumed: a `Range: bytes=0-1023` request against the file
returns `206 Partial Content`. Zenodo's storage backend supports partial
reads, so `remotezip` can list the zip's central directory and extract
individual entries by path without touching the rest of the archive.
`pipelines/fetch_mimii.py` pulls exactly 240 files this way (~615MB
estimated, well under the 6.9GB source), not the full corpus.

**Split by source, at the strongest available grain.** MIMII's own
directory structure exposes four physically distinct valve units
(`id_00`, `id_02`, `id_04`, `id_06`) per machine type. Recipe step 3 ("split
by source, never by window") is honored by drawing train and the
held-out-normal test slice from `id_00`, and validation plus a cross-unit
anomaly test slice from `id_02` — a different physical machine entirely,
not just different files from the same one. This is a stronger split than
the recipe's own minimum bar (file-level, stratified) and was chosen
because the data made it available essentially for free.

## Decision: pure NumPy + soundfile features, frame count solved for, not assumed

`pipelines/features.py` implements STFT and the mel filterbank directly in
NumPy — no librosa, per the recipe's own step 2 rule (and per SiloSense's
own precedent). One design choice worth recording: the number of STFT
frames a fixed-duration clip produces is a function of sample count, FFT
size, and hop length, not a free parameter — so rather than pick a clip
duration and accept whatever frame count falls out, `TARGET_FRAMES = 256`
was fixed first and `CLIP_SAMPLES` solved backward from it
(`(256-1)*hop + n_fft`). That gives a conv autoencoder's three stride-2
downsampling steps exact integer output shapes with no cropping or padding
inside the model itself — the kind of small decision that avoids a class of
shape-mismatch bugs entirely rather than debugging them one at a time.

## Decision: `common_quant.py` is shared verbatim by both pipelines

Full Spectrum's own text claims some project cards "reuse the [prior
project's] pipeline, unchanged, retargeted." `common_quant.py` is the part
of this build that makes that claim literally true rather than descriptive:
`export_onnx`, `quantize_static_int8`, and `verify_fp32_vs_int8` are the
exact same functions for the maize CNN and the MIMII autoencoder, with only
the model, the input tensor shape, and the scoring function differing at
the call site. Both models are convolutional on purpose — recipe step 6's
whole point (dynamic quantization only touches MatMul/Gemm and leaves
convolutions in FP32) only means anything if the model actually contains
convolutions; a pure-Gemm autoencoder would have made static and dynamic
quantization look nearly identical, teaching the wrong lesson on stage.

`verify_fp32_vs_int8` also encodes the paper's own small-sample honesty
rule directly: a held-out set under 200 examples has its verdict forced to
`"no_loss"` even on a nominal INT8 improvement, rather than letting noise
get reported as a win. Both real runs so far (maize: n=80) landed in this
band, and the verdict text says so explicitly rather than showing a bare
number.

## Two bugs a live run caught that reading the code would not have

**`torch.onnx.export` under torch 2.13's default dynamo exporter crashes on
Windows.** Its verbose success message prints a checkmark emoji; Windows'
default console codepage (cp1252) can't encode it, raising
`UnicodeEncodeError` after tracing succeeds but before export completes.
Separately, the dynamo exporter warns that `dynamic_axes` (this module's
API) is unsupported under it and wants `dynamic_shapes` instead. Both
problems disappear with `dynamo=False`, which routes through the legacy
TorchScript-based exporter this module's API already assumes. Documented
inline in `export_onnx()` so it doesn't get "cleaned up" back into a crash.

**`Path.glob("*.jpg")` and `Path.glob("*.JPG")` match the identical file set
on Windows.** The filesystem is case-insensitive, so concatenating both
patterns' results (a reasonable-looking way to catch mixed-case
extensions) silently doubled every image in the maize corpus. The doubled
class counts happened to cancel out in the class-weight calculation (a
uniform 2x scale doesn't change inverse-frequency ratios), which is exactly
why it wasn't obvious from the printed numbers alone — total counts
matching `find`'s output was the tell. The real damage was upstream:
`rng.shuffle(files)` was shuffling a list where each real file appeared
twice, with a real chance both copies of one file landed in different
splits — a train/val leak, in the exact step (source-based splitting) this
entire build exists to enforce correctly. Fixed to a single
case-normalized suffix check (`p.suffix.lower() in {".jpg", ".jpeg",
".png"}`) over one directory listing.

## Verified live (2026-08-29)

`maize_train.py`: 480 train / 80 val / 80 test images (Kaggle's
`corn-or-maize-leaf-disease-dataset`, downloaded whole at 161MB), real
class-imbalance weighting (Gray_Leaf_Spot at roughly half Common_Rust's
example count), best-validation checkpoint restored (not last-epoch),
FP32 71.25% → INT8 70.00% test accuracy, correctly flagged `no_loss (n too
small to distinguish signal from noise)`, 3.01x ONNX size reduction
(94.8KB → 31.5KB). Real numbers now written into the shared passport via
`record_results.py` and confirmed rendering on the dashboard.

`mimii_anomaly.py`: the `ConvAutoencoder` shape round-trips exactly
((2,1,64,256) in, same out, 46,529 params); the full anomaly-only training
run against the fetched MIMII subset is pending the fetch completing.

## Decision: report the real MIMII AUC (0.42), don't tune until it looks better

The fetch finished; the full anomaly-only run against it produced FP32 AUC
0.27 at 25 epochs -- below 0.5, meaning held-out normal audio reconstructed
*worse* than the abnormal class, the opposite of the intended signal.

**Checked before assumed.** A number this surprising gets a diagnostic
before a writeup, not after. Reconstruction error was broken down by
source: held-out normal (id_00, 0.156 mean) scored *higher* than both
abnormal-id_00 (0.127, same physical unit as training) and abnormal-id_02
(0.141, a different unit). The inversion shows up within one physical unit
alone, which rules out the obvious alternative explanation (that the model
never saw id_02's normal characteristics in training, so anything from
id_02 looks unfamiliar regardless of label) -- this is not a domain-shift
artifact, it needed to be an actual property of the normal-vs-abnormal
signal difference at this training scale.

Re-run at 25 / 100 / 300 epochs to tell undertraining from a structural
ceiling: AUC 0.27 -> 0.40 -> 0.43, monotonic but visibly plateauing, loss
still dropping slowly at 300 but AUC gains shrinking each time. Settled on
150 epochs as the shipped default -- past that point, more session time
spent chasing a marginally better number stopped being the right trade
against everything else this build still needed.

**The number that shipped is real, not selected.** MIMII's own paper
documents valve as one of the harder machine types for autoencoder-based
anomaly detection even at full corpus scale (thousands of examples); this
build trains on 100. A modest-to-weak AUC (0.42 FP32, 0.42 INT8 -- the
`no_loss` verdict held correctly here too) is the honest result at this
scale, not a failure to paper over. This is also not a new argument invented
to excuse a bad number -- it is the exact distinction Full Spectrum's own
`diesel_generator` project card already draws about MIMII being a proxy
dataset, not the real corpus: a public baseline proves the *pipeline*
works (recipe steps 5-7, which it does, end to end, on real data), not
that the *product* is ready. Recorded in the passport via
`record_results.py` and rendering on the dashboard next to maize's own
(unrelated, unaffected) 71.25%/70.00% numbers.

## Next steps

1. Full end-to-end demo rehearsal: both mentor sessions via
   `--backend vertex`, both pipelines' real numbers on the dashboard side
   by side, the cross-project recall citation, from a cold environment.
2. If there's time before submission: a real local MIMII corpus, an actual
   larger bottleneck, or a different scoring function are the legitimate
   next levers for the AUC -- not further epochs past the plateau already
   found here.
