import argparse
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sads import constants as c
from sads.data import load_returns, stress_releases

FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "49_Industry_Portfolios_daily_CSV.zip"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=STLFSI4"


def fetch(url: str) -> bytes:
    """Retrieve a public input with a bounded wait and an identifiable agent."""
    request = Request(url, headers={"User-Agent": "SADS-reproducibility/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def download_data(data_dir: Path = c.DATA_DIR) -> None:
    """Validate both downloads before replacing any existing dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=data_dir) as temporary:
        staging = Path(temporary)
        returns_path = staging / c.RETURNS_PATH.name
        stress_path = staging / c.STRESS_PATH.name
        for url, destination in ((FRENCH_URL, returns_path), (FRED_URL, stress_path)):
            try:
                payload = fetch(url)
                if destination == returns_path:
                    with ZipFile(BytesIO(payload)) as archive:
                        members = [
                            name for name in archive.namelist()
                            if Path(name).name.lower() == c.RETURNS_PATH.name.lower()
                        ]
                        if len(members) != 1:
                            raise ValueError("ZIP must contain the 49-industry daily CSV")
                        payload = archive.read(members[0])
                destination.write_bytes(payload)
                if destination == returns_path:
                    load_returns(destination)
                elif stress_releases(destination).empty:
                    raise ValueError("STLFSI4 contains no usable observations")
            except Exception as error:
                raise RuntimeError(
                    f"Download or validation failed for {url}: {type(error).__name__}: {error}. "
                    "Access may be blocked; existing datasets were preserved. "
                    f"Obtain the source manually and save it as {data_dir / destination.name}."
                ) from error
        for path in (returns_path, stress_path):
            path.replace(data_dir / path.name)
    print(f"Validated 49-industry value-weighted returns and STLFSI4 in {data_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=c.DATA_DIR)
    args = parser.parse_args()
    try:
        download_data(args.data_dir)
    except (RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
