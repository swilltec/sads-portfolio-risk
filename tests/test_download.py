from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZipFile

import pytest

from scripts import download_data as downloader
from sads import constants as c
from sads.data import load_returns


def french_zip() -> bytes:
    names = ",".join(f"industry{i}" for i in range(49))
    values = ",".join(["1.25"] * 49)
    content = f"Average Value Weighted Returns -- Daily\n,{names}\n20000103,{values}\n\n"
    result = BytesIO()
    with ZipFile(result, "w") as archive:
        archive.writestr(c.RETURNS_PATH.name, content)
    return result.getvalue()


def test_download_validates_and_saves_expected_inputs(tmp_path: Path, monkeypatch) -> None:
    responses = {
        downloader.FRENCH_URL: french_zip(),
        downloader.FRED_URL: b"observation_date,STLFSI4\n1999-12-31,1\n",
    }
    monkeypatch.setattr(downloader, "fetch", responses.__getitem__)
    downloader.download_data(tmp_path)
    assert load_returns(tmp_path / c.RETURNS_PATH.name).shape == (1, 49)
    assert (tmp_path / c.STRESS_PATH.name).read_bytes() == responses[downloader.FRED_URL]


@pytest.mark.parametrize("failure", ["blocked", "html", "empty"])
def test_failed_second_download_preserves_both_inputs(tmp_path: Path, monkeypatch, failure) -> None:
    for name in (c.RETURNS_PATH.name, c.STRESS_PATH.name):
        (tmp_path / name).write_text("existing dataset")

    def fetch(url):
        if url == downloader.FRENCH_URL:
            return french_zip()
        if failure == "blocked":
            raise HTTPError(url, 403, "Forbidden", {}, None)
        if failure == "empty":
            return b"observation_date,STLFSI4\n"
        return b"<html>Access denied</html>"

    monkeypatch.setattr(downloader, "fetch", fetch)
    with pytest.raises(RuntimeError, match="existing datasets were preserved"):
        downloader.download_data(tmp_path)
    for name in (c.RETURNS_PATH.name, c.STRESS_PATH.name):
        assert (tmp_path / name).read_text() == "existing dataset"
    assert len(list(tmp_path.iterdir())) == 2
