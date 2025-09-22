# Stroke-on-CT AI (Advanced)

End-to-end system for intracranial hemorrhage (ICH) triage on non-contrast head CT:
- DICOM ingest → anonymize → resample → multi-window channel stacks
- 2.5D slice classifier + study-level attention aggregator
- (Optional) weakly-supervised segmentation placeholder
- FastAPI inference service + Streamlit triage UI
- MLOps notes (MLflow), Dockerfile, configs

> This is a **starter implementation** with production-minded structure You can extend each module.
