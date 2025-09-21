import os, json, traceback, requests

BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

def main():
    print(f"Using BASE_URL={BASE_URL}")

    # ---- GET /
    try:
        r = requests.get(f"{BASE_URL}/", timeout=10)
        print("GET / ->", r.status_code)
        try:
            msg = r.json().get("message", r.text)
        except Exception:
            msg = r.text
        print("Message:", msg)
    except Exception as e:
        print("GET error:", e)
        traceback.print_exc()

    # ---- POST /predict
    payload = {
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
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=20)
        print("POST /predict ->", r.status_code)
        try:
            body = r.json()
            print("Prediction:", body.get("prediction"))
            print("prob_gt_50k:", body.get("prob_gt_50k"))
        except json.JSONDecodeError:
            print("Raw body:", r.text)
    except Exception as e:
        print("POST error:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
