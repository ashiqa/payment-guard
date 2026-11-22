import argparse
import json
from pathlib import Path

from paymentguard.data import generate
from paymentguard.train import train


def main():
    parser = argparse.ArgumentParser(description="PaymentGuard reproducible ML pipeline")
    subs = parser.add_subparsers(dest="command", required=True)
    gen = subs.add_parser("generate", help="Generate explicitly synthetic raw transactions")
    gen.add_argument("--rows", type=int, default=30000)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--output", default="data/synthetic.csv")
    fit = subs.add_parser("train")
    fit.add_argument("--data", default="data/synthetic.csv")
    fit.add_argument("--output", default="artifacts")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--max-fpr", type=float, default=.01)
    fit.add_argument("--data-kind", choices=["synthetic", "external"], required=True,
                     help="Explicit provenance label, included in reports and API responses")
    args = parser.parse_args()
    if args.command == "generate":
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        generate(args.rows, args.seed).to_csv(path, index=False)
        print(f"Wrote {args.rows} SYNTHETIC events to {path}")
    else:
        report = train(args.data, args.output, args.seed, args.max_fpr, args.data_kind)
        print(json.dumps({"selected_model": report["selected_model"], "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
