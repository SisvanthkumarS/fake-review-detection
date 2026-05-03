"""Download Amazon Customer Reviews from the ClickHouse public mirror.

Source: https://datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews/
Files are sharded by year. We grab the most recent two years (highest volume).
Total download: roughly 5-10 GB compressed.
"""
from pathlib import Path
import sys
import urllib.request
import urllib.error

YEARS = [2015, 2014]
BASE_URL = "https://datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews"
DEST_DIR = Path("data/raw")


def download_year(year):
    filename = f"amazon_reviews_{year}.snappy.parquet"
    dest = DEST_DIR / filename
    url = f"{BASE_URL}/{filename}"

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Already have {filename} ({size_mb:.0f} MB)")
        return dest

    print(f"  Downloading {filename}...")
    print(f"    URL: {url}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_done = downloaded // (1024 * 1024)
            mb_total = total_size // (1024 * 1024)
            sys.stdout.write(f"\r    {pct:3d}%  {mb_done:5d} / {mb_total} MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print(f"\n    Done: {dest}")
        return dest
    except urllib.error.HTTPError as e:
        print(f"\n    HTTP {e.code} - {filename} not available.")
        return None
    except Exception as e:
        if dest.exists():
            dest.unlink()
        print(f"\n    Failed: {e}")
        return None


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(YEARS)} year(s)...")
    print(f"Years: {YEARS}")
    print(f"Destination: {DEST_DIR.resolve()}\n")

    downloaded = []
    for y in YEARS:
        path = download_year(y)
        if path is not None:
            downloaded.append(path)

    if not downloaded:
        raise SystemExit("\nERROR: No files downloaded. Check internet and try again.")

    total_mb = sum(p.stat().st_size for p in downloaded) / (1024 * 1024)
    print(f"\nDownloaded {len(downloaded)} file(s), {total_mb:.0f} MB total.")
    print("\nNext: python batch/inspect_raw.py")


if __name__ == "__main__":
    main()
