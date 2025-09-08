import subprocess
import sys
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter

OUT = Path("data/raw/kindle_reviews.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

OWNER = "bharadwaj6/kindle-reviews"

FILE_CANDIDATES = [
    "kindle_reviews.csv",
    "kindle-reviews.csv",
    "Reviews.csv",
    "data/kindle_reviews.csv",
    "data/kaggle_reviews.csv",
]

ENCODINGS = ["utf-8-sig", "utf-8", "cp1251", "latin1", "iso-8859-1"]


def try_kagglehub(name, encoding):
    print(f"Trying {name} with encoding={encoding}")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        OWNER,
        name,
        pandas_kwargs={
            "encoding": encoding,
            "index_col": 0,
            "dtype": str,
            "low_memory": False,
        },
    )
    return df


def fallback_kaggle_cli():
    print(
        "Attempting fallback: kaggle CLI (requires kaggle CLI installed and configured)"
    )
    cmd = ["kaggle", "datasets", "download", "-d", OWNER, "--unzip"]
    try:
        subprocess.run(cmd, check=True)
        print("kaggle CLI succeeded: files downloaded to current working directory")
        return True
    except FileNotFoundError:
        print("kaggle CLI not found")
        return False
    except subprocess.CalledProcessError as e:
        print("kaggle CLI failed:", e)
        return False


def main():
    for name in FILE_CANDIDATES:
        for enc in ENCODINGS:
            try:
                df = try_kagglehub(name, enc)
                print("Loaded OK:", name, enc, "rows=", len(df))
                df.to_csv(OUT, index=False, encoding="utf-8-sig")
                print("Saved cleaned CSV to", OUT)
                print(df.head())
                return
            except Exception as e:
                msg = getattr(e, "message", str(e))
                print(f"Failed {name} @ {enc}: {type(e).__name__}: {msg}")

    ok = fallback_kaggle_cli()
    if not ok:
        print("All attempts failed. Inspect dataset page or try kaggle CLI manually.")
        sys.exit(2)

    for cand in FILE_CANDIDATES + ["kindle_reviews.csv", "Reviews.csv"]:
        p = Path(cand)
        if p.exists():
            import pandas as pd

            print("Found local file from kaggle CLI:", p)
            for enc in ENCODINGS:
                try:
                    df = pd.read_csv(p, encoding=enc, dtype=str, low_memory=False)
                    if df.columns[0] == "" or df.columns[0] == "_c0":
                        df = df.iloc[:, 1:]
                    df.to_csv(OUT, index=False, encoding="utf-8-sig")
                    print("Saved cleaned CSV to", OUT)
                    print(df.head())
                    return
                except Exception as e:
                    print("Reading local file failed with", enc, type(e).__name__)

    print("Fallback completed but no usable CSV found.")
    sys.exit(3)


if __name__ == "__main__":
    main()
