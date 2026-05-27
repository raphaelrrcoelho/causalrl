import tomllib
from pathlib import Path


def test_core_install_does_not_require_torch_and_supports_python_311():
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    project = config["project"]

    assert project["requires-python"] == ">=3.11"
    assert all(not dependency.startswith("torch") for dependency in project["dependencies"])
    assert "torch>=2.5" in project["optional-dependencies"]["torch"]
    assert "torch>=2.5" in project["optional-dependencies"]["dev"]


def test_public_library_metadata_and_documentation_surface_exist():
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert "Documentation" in project["urls"]
    assert "Source" in project["urls"]
    assert "docs" in project["optional-dependencies"]
    for path in (
        "mkdocs.yml",
        "docs/index.md",
        "docs/guarantees.md",
        "docs/benchmarks.md",
        "docs/api.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        ".github/workflows/publish.yml",
    ):
        assert (root / path).exists(), f"missing public library surface: {path}"
