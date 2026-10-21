"""AURA Monitor: patient-adaptive, interpretable EEG monitoring built from the paper's models.

Modules (docs/plans/2026-10-05-aura-monitor.md):
  signals    common 18-channel double banana, resampling, per-window signal quality, per-channel features
  baseline   the patient's rolling reference (35 to 5 min earlier) and a vigilance-state proxy
  ied        interictal epileptiform discharge detection, typing and channel maps (trained on VEPISET)
  patterns   seizure fingerprints, within-patient clustering, similarity to earlier seizures, lateralization
  risk       calibrated short-horizon seizure likelihood from seizure history, time of day and IED burden
  explain    per-alarm evidence cards
  report     a self-contained HTML report a clinician can read
  pipeline   one patient's recordings in, one report out

Research software: outputs support, never replace, expert review of the EEG.
"""
