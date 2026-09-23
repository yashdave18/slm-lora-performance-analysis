"""Verify original export bytes in both the working tree and Git index."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "results/exports-submission-20260923-050201"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="Check the Git index that will be committed")
    args = parser.parse_args()
    def read(relative):
        if args.staged:
            return subprocess.run(["git", "show", ":" + relative], cwd=ROOT,
                                  check=True, capture_output=True).stdout
        return (ROOT / relative).read_bytes()
    inventory = json.loads(read(ARCHIVE + "/artifact_inventory.json"))
    failures = []
    for item in inventory:
        path = ARCHIVE + "/" + item["path"]
        if hashlib.sha256(read(path)).hexdigest() != item["sha256"]:
            failures.append(path)
    if failures:
        raise SystemExit("FAIL: mismatched artifact bytes:\n" + "\n".join(failures))
    where = "Git index" if args.staged else "working tree"
    print(f"PASS: {len(inventory)}/{len(inventory)} original artifact checksums verified ({where}).")

if __name__ == "__main__":
    main()
