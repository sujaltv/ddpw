conda env export --no-build --from-history > environment.yaml
pip list --format=freeze > requirements.txt
