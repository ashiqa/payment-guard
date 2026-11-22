"""Sequential HTTP benchmark; not a concurrent load/capacity test."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--requests", type=int, default=200)
parser.add_argument("--output", default="artifacts/benchmark.json")
args = parser.parse_args()
if args.requests < 1:
    parser.error("--requests must be positive")
payload = Path("examples/transaction.json").read_bytes()


def request():
    with urlopen(Request(args.url + "/predict", data=payload,
                 headers={"Content-Type": "application/json"}), timeout=10) as response:
        return json.loads(response.read())


for _ in range(10):
    request()
elapsed = []
start = time.perf_counter()
for _ in range(args.requests):
    tick = time.perf_counter()
    result = request()
    elapsed.append((time.perf_counter() - tick) * 1000)
total = time.perf_counter() - start
report = {"mode": "sequential HTTP, 10 warmup requests, single repeated payload",
          "requests": args.requests, "model_version": result["model_version"],
          "data_kind": result["data_kind"], "p50_ms": float(np.percentile(elapsed, 50)),
          "p95_ms": float(np.percentile(elapsed, 95)), "requests_per_second": args.requests / total,
          "limitation": "Local measurements only; no concurrency or production capacity claim."}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
