import json

import pandas as pd
import pytest

from src.config import FIXED_CLAIM
from src.data_loading import LabelError, LoaderError, load_file, normalise_label


def _csv(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return p


def test_semeval_style_columns(tmp_path):
    p = _csv(tmp_path, "a.csv", "ID,Target,Tweet,Stance\n1,X,Hot planet #SemST,FAVOR\n2,X,Hoax!,AGAINST\n3,X,hi,NONE\n")
    df, info = load_file(p)
    assert info.mapping["comment"] == "Tweet" and info.mapping["label"] == "Stance"
    assert df["label"].tolist() == [0, 2, 1]
    assert df["comment"].tolist() == ["Hot planet", "Hoax!", "hi"]
    assert df["id"].tolist() == ["1", "2", "3"]


def test_claim_column_used_or_fixed(tmp_path):
    p = _csv(tmp_path, "a.csv", "text,label\nfoo,0\n")
    df, _ = load_file(p)
    assert df["claim"].iloc[0] == FIXED_CLAIM
    df2, _ = load_file(p, claim_col=None, fixed_claim="Other claim")
    assert df2["claim"].iloc[0] == "Other claim"


def test_tsv_semicolon_utf16(tmp_path):
    df, _ = load_file(_csv(tmp_path, "a.tsv", "comment\tstance\nएक\tagainst\n"))
    assert df["label"].tolist() == [2]
    df, info = load_file(_csv(tmp_path, "b.csv", "sno;yt_comment;class\n7;x;neutral\n"))
    assert info.delimiter == ";" and df["id"].tolist() == ["7"] and df["label"].tolist() == [1]
    df, info = load_file(_csv(tmp_path, "c.csv", "comment,label\nজলবায়ু,2\n", encoding="utf-16"))
    assert df["comment"].iloc[0] == "জলবায়ু" and df["label"].tolist() == [2]


def test_json_jsonl_txt_xlsx(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"data": [{"text": "a", "label": "favour"}]}), encoding="utf-8")
    assert load_file(tmp_path / "a.json")[0]["label"].tolist() == [0]
    (tmp_path / "a.jsonl").write_text('{"comment": "a"}\n{"comment": "b"}\n', encoding="utf-8")
    df, info = load_file(tmp_path / "a.jsonl")
    assert len(df) == 2 and not info.has_labels
    (tmp_path / "a.txt").write_text("one\n\ntwo\n", encoding="utf-8")
    assert load_file(tmp_path / "a.txt")[0]["comment"].tolist() == ["one", "two"]
    pytest.importorskip("openpyxl")
    pd.DataFrame({"Comment": ["x"], "Label": [2]}).to_excel(tmp_path / "a.xlsx", index=False)
    assert load_file(tmp_path / "a.xlsx")[0]["label"].tolist() == [2]


def test_numeric_label_spellings():
    assert normalise_label("1.0") == "neutral"
    assert normalise_label(" FAVOR ") == "favour"
    assert normalise_label("none") == "neutral"
    assert normalise_label("maybe") is None


def test_unknown_labels(tmp_path):
    p = _csv(tmp_path, "a.csv", "comment,label\na,favour\nb,maybe\n")
    with pytest.raises(LabelError):
        load_file(p)
    df, info = load_file(p, allow_unknown_labels=True)
    assert info.unknown_labels == {"maybe": 1} and df["label"].isna().tolist() == [False, True]


def test_custom_label_map_and_override(tmp_path):
    p = _csv(tmp_path, "a.csv", "remark,y\na,P\nb,N\n")
    df, _ = load_file(p, text_col="remark", label_map={"p": "favour", "n": "against"})
    assert df["label"].tolist() == [0, 2]


def test_missing_comment_column(tmp_path):
    with pytest.raises(LoaderError, match="No comment column"):
        load_file(_csv(tmp_path, "a.csv", "foo,bar\n1,2\n"))
    with pytest.raises(LoaderError):
        load_file(_csv(tmp_path, "b.csv", "text\nx\n"), text_col="nope")


def test_empty_comments_flagged(tmp_path):
    df, info = load_file(_csv(tmp_path, "a.csv", "comment\n#SemST\nok\n"))
    assert df["empty"].tolist() == [True, False] and info.n_empty == 1
