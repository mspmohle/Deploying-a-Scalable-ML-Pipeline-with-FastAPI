End-to-end ML project that trains a binary income classifier on the Census (Adult) dataset and serves predictions via a FastAPI REST API. Includes unit tests, linting, CI, slice metrics, and a model card.

Project Structure
├─ data/
│  └─ census.csv
├─ ml/
│  ├─ data.py              # preprocessing (process_data)
│  └─ model.py             # train_model, inference, metrics, save/load, slice metrics
├─ model/                  # serialized artifacts (gitignored)
│  ├─ model.joblib
│  ├─ encoder.joblib
│  └─ lb.joblib
├─ screenshot/             # rubric proofs (singular)
│  ├─ metrics.json
│  ├─ slice_output.txt
│  ├─ local_api.png
│  ├─ health.json
│  ├─ predict_stdout.txt
│  ├─ cloud_health.json
│  └─ cloud_predict.json
├─ screenshots/            # some graders expect plural
│  ├─ continuous_integration.png
│  └─ unit_test.png
├─ tests/
│  ├─ conftest.py
│  ├─ test_data_preprocess.py
│  └─ test_model_core.py
├─ train_model.py          # full pipeline script
├─ main.py                 # FastAPI app (GET /, GET /health, POST /predict)
├─ local_api.py            # simple client: one GET and one POST
├─ requirements.txt
├─ environment.yml
├─ Procfile
├─ render.yaml
└─ model_card_template.md  # completed model card

Environment Setup
Conda (recommended)
conda env create -f environment.yml     # name: fastapi
conda activate fastapi
python -V                                # Python 3.10.x
pip install -r requirements.txt          # ensure pip deps (flake8/joblib/typing_extensions etc.)

Train the Model & Generate Proof Files
conda activate fastapi
python train_model.py --data-path data/census.csv --target salary

Outputs:

Artifacts (gitignored): model/model.joblib, model/encoder.joblib, model/lb.joblib

Overall metrics: screenshot/metrics.json

Slice metrics (per categorical value): screenshot/slice_output.txt

Example overall metrics (test):

Precision: 0.734

Recall: 0.636

F1: 0.682

Run the API Locally

Start the server:
conda activate fastapi
uvicorn main:app --reload
# runs at http://127.0.0.1:8000

Interact (in another terminal):
conda activate fastapi
python local_api.py
# Example output:
# Status Code: 200
# Result: Income classifier API
# Status Code: 200
# Result: <=50K

Endpoints:

GET / → welcome message

GET /health → {"status": "ok"}

POST /predict → JSON body (single record), returns {"prediction": "<=50K|>50K", "prob_gt_50k": float}

Sample payload:
{
  "age": 39,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital-gain": 2174,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}

Cloud Deployment (Render)

This repo includes a Procfile and render.yaml for one-click deploy.

Push this branch (p2-dev-clean) to GitHub.

In Render: New → Blueprint, pick this repo, and Deploy.

After “Live”, set:
BASE="https://YOUR-RENDER-URL.onrender.com"
curl -s $BASE/health

Notes

Artifacts not in Git: model/ is ignored to avoid large files in history; re-create via train_model.py.

Repro: All steps (train, serve, test) are automated locally and in CI.

Python 3.10: Match local env and CI (see badge above).

License

Educational use under course terms. Replace with your preferred license if needed.
