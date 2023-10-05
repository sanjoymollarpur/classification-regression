# classification-regression

python -m venv tfod
source tfod/bin/activate 
deactivate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python3 -m pip freeze > p.txt
cat p.txt


conda create --name tf python=3.9
conda activate tf
