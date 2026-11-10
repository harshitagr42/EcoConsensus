from src.reporting import Report, write_latest
from src.utils import safe_write


def test_safe_write_never_overwrites(tmp_path):
    a = safe_write(tmp_path / "r.md", "one")
    b = safe_write(tmp_path / "r.md", "two")
    c = safe_write(tmp_path / "r.md", "three", add_timestamp=False)
    d = safe_write(tmp_path / "r.md", "four", add_timestamp=False)
    assert len({a, b, c, d}) == 4
    assert a.read_text() == "one" and b.read_text() == "two" and c.name == "r.md" and d.name == "r_2.md"


def test_second_result_archives_first(tmp_path):
    latest, archive = tmp_path / "result.txt", tmp_path / "archive"
    write_latest(latest, "first", archive)
    _, archived = write_latest(latest, "second", archive)
    assert latest.read_text() == "second"
    assert archived is not None and archived.read_text() == "first"
    write_latest(latest, "third", archive)
    assert len(list(archive.iterdir())) == 2


def test_report_sections_and_warnings(tmp_path):
    r = Report("T").add("Explainability", "w").add("System comparison", "c").warn("careful")
    text = r.render()
    assert text.index("System comparison") < text.index("Explainability") < text.index("careful")
    p = r.save(tmp_path / "rep.md")
    assert p.exists() and p.name.startswith("rep_")
