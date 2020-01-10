from docs.autogen import generate
import pytest
import pathlib


def test_docs_in_custom_destination_dir(tmpdir):
    tmpdir = pathlib.Path(tmpdir)
    generate(tmpdir)
    assert (tmpdir / 'examples').is_dir()
    assert (tmpdir / 'tutorials').is_dir()
    assert (tmpdir / 'documentation').is_dir()
    assert 'hyperparameter tuning' in (tmpdir / 'index.md').read_text()
    tuners_file = tmpdir / 'documentation' / 'tuners.md'
    assert 'Random search tuner' in tuners_file.read_text()


if __name__ == '__main__':
    pytest.main([__file__])
