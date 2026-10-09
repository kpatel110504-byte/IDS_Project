@echo off
echo Running IDS Project Pipeline...
python src\01_eda.py
python src\02_preprocessing.py
python src\03_ml_models.py
python src\04_pytorch_nn.py
python src\05_compare_eval.py
echo Done! Results saved in results\ folder.
pause
