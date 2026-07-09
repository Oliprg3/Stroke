#CT scan based system

End-to-end system for intracranial hemorrhage triage on non contrast head CT:
- DICOM is ingest   and then anonymize after that resample multi-window channel stacks
- 2.5D slice classifier + study-level attention aggregator
- weakly-supervised segmentation placeholder
- FastAPI inference service + Streamlit triage UI
- MLOps notes (MLflow), Dockerfile, configs

      - This is starter implementation you can extend by adding features. 
