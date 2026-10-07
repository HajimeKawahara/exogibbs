"""Documentation builds must not change the audited source snapshot."""

from benchmarks.documented_examples import acceptance


def test_generated_sphinx_files_do_not_change_source_provenance_or_coverage(tmp_path):
    root = tmp_path / "checkout"
    documents = root / "documents"
    documents.mkdir(parents=True)
    source = documents / "example.rst"
    source.write_text("An authored example.\n")
    japanese = tmp_path / "japanese"
    japanese.mkdir()
    (japanese / "main_ja.tex").write_text("Japanese manual")
    resources = {"japanese_docs": str(japanese), "exoeos_checkout": str(tmp_path / "exoeos")}
    initial = acceptance.provenance(root, resources)["exogibbs"]

    for name in ("_build", "examples", "exogibbs", "backreferences"):
        output = documents / name
        output.mkdir()
        (output / "stale.rst").write_text("python examples/removed.py\n")
        (output / "copied.py").write_text("print('generated copy')\n")
    (documents / "sg_execution_times.rst").write_text("0.125 seconds\n")

    rebuilt = acceptance.provenance(root, resources)["exogibbs"]
    assert rebuilt["sha256"] == initial["sha256"]
    assert rebuilt["file_count"] == initial["file_count"] == 1
    assert acceptance.coverage_report(root, japanese, (), {})["accepted"]

    source.write_text("python examples/new.py\n")
    changed = acceptance.provenance(root, resources)["exogibbs"]
    assert changed["sha256"] != initial["sha256"]
    assert not acceptance.coverage_report(root, japanese, (), {})["accepted"]
