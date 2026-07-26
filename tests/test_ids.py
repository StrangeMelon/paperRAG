from __future__ import annotations

import hashlib

import pytest

from paper_rag.utils.ids import (
    make_paper_id,
    normalize_arxiv,
    normalize_doi,
    sha1_of_file,
    split_arxiv_version,
    to_safe_dirname,
)


def test_normalize_arxiv_removes_version_and_accepts_url():
    assert normalize_arxiv("2310.12345v2") == "2310.12345"
    assert normalize_arxiv("https://arxiv.org/abs/2310.12345") == "2310.12345"
    assert normalize_arxiv("not-an-arxiv-id") is None


def test_split_arxiv_version():
    assert split_arxiv_version("2310.12345v2") == ("2310.12345", "v2")
    assert split_arxiv_version("2310.12345") == ("2310.12345", None)
    assert split_arxiv_version("invalid") == (None, None)


def test_normalize_doi_removes_known_prefixes():
    assert normalize_doi(" DOI:10.1109/ABC.2024.000123 ") == "10.1109/abc.2024.000123"
    assert normalize_doi("https://doi.org/10.1000/example") == "10.1000/example"
    assert normalize_doi("not-a-doi") is None


def test_make_paper_id_uses_stable_priority(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"paper content")

    assert make_paper_id(arxiv_id="2310.12345v2") == "arxiv:2310.12345"
    assert make_paper_id(doi="10.1000/example") == "doi:10.1000/example"
    assert make_paper_id(pdf_path=pdf_path) == f"sha1:{hashlib.sha1(b'paper content').hexdigest()}"

    with pytest.raises(ValueError, match="Need at least one"):
        make_paper_id()


def test_sha1_of_file_and_safe_directory_name(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"paper content")

    assert sha1_of_file(pdf_path) == hashlib.sha1(b"paper content").hexdigest()
    assert to_safe_dirname("doi:10.1000/a/b") == "doi_10.1000_a_b"