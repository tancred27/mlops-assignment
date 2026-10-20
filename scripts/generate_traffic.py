"""Send a mix of valid and invalid requests to the API to populate the dashboards."""
import argparse
from collections import Counter
import random
import time

import requests

SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    rng = random.Random(0)
    statuses = Counter()
    for i in range(args.n):
        record = {
            **SAMPLE,
            "age": rng.randint(30, 77),
            "chol": rng.randint(150, 400),
            "thalach": rng.randint(90, 200),
            "oldpeak": round(rng.uniform(0, 5), 1),
            "cp": rng.randint(1, 4),
        }
        if i % 15 == 0:
            record["cp"] = 9  # invalid category, returns 422
        response = requests.post(f"{args.url}/predict", json=record, timeout=5)
        statuses[response.status_code] += 1
        time.sleep(0.05)
    print("Status codes:", dict(statuses))


if __name__ == "__main__":
    main()