# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Pull Request Guide
Before you submit a pull request, check that it meets these guidelines:

1. Is this the first pull request that you're making with GitHub? If so, read the guide [Making a pull request to an open-source project](https://github.com/gabrieldemarmiesse/getting_started_open_source).

2. Include "resolves #issue_number" in the description of the pull request if applicable and briefly describe your contribution.

3. For the case of bug fixes, add new test cases which would fail before your bug fix.

## Code Style Guide
This project tries to closely follow the official Python Style Guide detailed in [PEP8](https://www.python.org/dev/peps/pep-0008/). We use [Flake8](http://flake8.pycqa.org/en/latest/) to enforce it.
The docstrings follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#381-docstrings).

## Testing Guide
[Pytest](https://docs.pytest.org/en/latest/) is used to write the unit tests.
You should test your code by writing unit testing code in `tests` directory.
The testing file name should be the `.py` file with a prefix of `test_` in the corresponding directory,
e.g., the name should be `test_layers.py` if the code of which is to test `layer.py`.

## Pre-commit hook

You can make git run `flake8` before every commit automatically. It will make you go faster by
avoiding pushing commit which are not passing the flake8 tests. To do this, 
open `.git/hooks/pre-commit` with a text editor and write `flake8` inside. If the `flake8` test doesn't
pass, the commit will be aborted.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).


# Rebuilding Protos
If you make changes to any `.proto` file, you'll have to rebuild the generated
`*_pb2.py` files. To do this, run these commands from the root directory of this
project:

```
pip install grpcio-tools
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. kerastuner/protos/kerastuner.proto
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. kerastuner/protos/service.proto
```
