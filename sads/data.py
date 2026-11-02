import hashlib
import re
from io import StringIO
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from . import constants as c
from .types import PathLike


class FileFingerprint(TypedDict):
    """Input file identity recorded with the experiment settings."""

    path: str
    sha256: str
    bytes: int


def load_returns(
    path: PathLike, start: str = c.SAMPLE_START, end: str | None = c.SAMPLE_END
) -> pd.DataFrame:
    """Read the 49-industry value-weighted block as daily decimal returns.

    The result has a trading-date index and one column per industry. Missing
    French return codes are converted to NaN; incomplete samples are rejected
    to preserve the portfolio weights and trading calendar.
    """
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    marker = next(
        i for i, line in enumerate(lines) if c.FRENCH_VALUE_WEIGHTED_HEADER in line
    )
    header = marker + 1
    while not lines[header].strip():
        header += 1
    rows = []
    for line in lines[header + 1 :]:
        if not re.match(c.FRENCH_DATE_PATTERN, line):
            break
        rows.append(line)
    frame = pd.read_csv(StringIO("\n".join([lines[header], *rows])), index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str), format=c.FRENCH_DATE_FORMAT)
    frame.index.name = "date"
    frame.columns = frame.columns.str.strip()
    if len(frame.columns) != c.N_INDUSTRIES or not frame.index.is_unique:
        raise ValueError("Expected 49 industries and unique dates")
    frame = (
        frame.astype(float).replace(list(c.FRENCH_MISSING_RETURNS), np.nan)
        / c.PERCENT_SCALE
    )
    frame = frame.sort_index().loc[slice(start, end)]
    if frame.empty or frame.isna().any().any():
        raise ValueError(
            "Selected return sample is empty or incomplete; "
            "choose a complete 49-industry sample"
        )
    return frame


def stress_releases(
    path: PathLike,
    release_calendar: PathLike | None = None,
    lag_days: int = c.PUBLICATION_LAG_DAYS,
) -> pd.DataFrame:
    """Attach release dates and an expanding weekly mean to STLFSI4.

    A release calendar supplies observation_date and release_date columns.
    Otherwise, publication is assumed to occur lag_days after observation.
    The returned rows are ordered by release date.
    """
    raw = pd.read_csv(path, na_values=[c.STRESS_MISSING_VALUE])
    raw = raw.rename(columns={raw.columns[0]: "observation_date"})
    raw["observation_date"] = pd.to_datetime(raw["observation_date"])
    raw = (
        raw[["observation_date", c.STRESS_SERIES]]
        .dropna()
        .sort_values("observation_date")
    )
    if raw.observation_date.duplicated().any():
        raise ValueError("Duplicate stress observations")
    if release_calendar:
        calendar = pd.read_csv(
            release_calendar, parse_dates=["observation_date", "release_date"]
        )
        raw = raw.merge(
            calendar, on="observation_date", validate="one_to_one", how="left"
        )
        if raw.release_date.isna().any():
            raise ValueError(
                "Release calendar must cover every nonmissing stress observation"
            )
    else:
        if lag_days < 1:
            raise ValueError("An assumed publication lag must be positive")
        raw["release_date"] = raw.observation_date + pd.Timedelta(days=lag_days)
    if (raw.release_date <= raw.observation_date).any():
        raise ValueError("Publication must follow the week-ending date")
    raw = raw.sort_values(["release_date", "observation_date"])
    if raw.release_date.duplicated().any():
        raise ValueError("Use one observation per release date")
    raw["stress_threshold"] = raw.STLFSI4.expanding().mean()
    return raw


def align_signals(
    returns: pd.DataFrame, releases: pd.DataFrame, vol_window: int = c.VOLATILITY_WINDOW
) -> pd.DataFrame:
    """Align released stress and trailing volatility to each forecast date.

    Release-day observations are excluded. Portfolio volatility ends on t-1;
    both expanding thresholds use values available before the return on t.
    """
    dates = pd.DataFrame({"date": returns.index})
    aligned = pd.merge_asof(
        dates,
        releases,
        left_on="date",
        right_on="release_date",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("date")
    aligned = aligned.rename(columns={c.STRESS_SERIES: "stress"})
    portfolio = returns.mean(axis=1)
    aligned["realized_vol"] = portfolio.rolling(vol_window).std(ddof=1).shift(1)
    aligned["vol_threshold"] = aligned.realized_vol.expanding().mean()
    return aligned


def fingerprint(path: PathLike) -> FileFingerprint:
    """Return the absolute path, content hash, and size of an input file."""
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
