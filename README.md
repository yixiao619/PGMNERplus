# PGMNERplus

Source collection for “Learning to Retain and Align: Information Polarization and Partial Transport for Multimodal Named Entity Recognition”.

## Contents

- `core/`: MNER implementation, model modules, and journal experiment scripts under `kais_extension/`.
- `launchers/`: MNER training and evaluation launch scripts and configurations.
- `mechanisms/`: numerical controls and mechanism figure scripts.
- `prediction_interventions/`: local-gate model and prediction-level intervention scripts.
- `prediction_snapshot/`: preserved model source associated with the reported intervention archive.
- `figure_reproduction/`: scripts for the prediction-sensitive figures and cases.
- `grounded/`: grounded evaluation, adaptation, verification, calibration and development-search scripts. These directories include search variants as well as the selected configuration; do not interpret every variant as a reported final result.

## Status and execution

This is a source archive, not a newly trained model or a verified one-command reproduction. Existing scripts retain their original absolute paths and cross-project imports to preserve provenance. Configure dataset, pretrained model, checkpoint, cached prediction and output paths before execution. Review each component's README/PROTOCOL and SOURCE_MANIFEST.json. The launcher directory originally expects the MNER implementation in a sibling MINER-My directory; adapt that path to `core/` here.

Model weights, datasets, raw social-media records, cached generations and run outputs are not bundled. Numerical controls use constructed inputs; their outputs must not be reported as benchmark F1. Real-data inference and training require the corresponding data and checkpoints.

Dependencies are retained in component requirements/configuration files; different model stages may require different environments. No training jobs were started during packaging. VALIDATION.json records Python syntax parsing, not runtime or scientific reproduction checks.

## Manuscript mapping

- MNER architecture and original benchmarks: `core/`, `launchers/`.
- M1–M8 controlled mechanisms: `mechanisms/phenomena/`, `prediction_interventions/scripts/mechanism_audit.py`.
- Prediction-level interventions and Figures 8–9: `prediction_interventions/`, `prediction_snapshot/`, `figure_reproduction/`.
- Grounded routing and official scoring: `core/kais_extension/experiment14_gmner_integration/`, `core/kais_extension/experiment21_gmner_validation/`.
- Adapted entities and complementary verification: `grounded/PGMNER_gmner_adapt_20261110/`, `grounded/PGMNER_gmner_recombine_20261111/`, `grounded/PGMNER_gmner_threshold_20261111/`, plus their supporting grounded projects.

Public hosting has not been configured by this packaging step. A local directory is not a public repository URL. Preserve upstream licensing and attribution when publishing; no new blanket license is asserted for this mixed-provenance collection.
