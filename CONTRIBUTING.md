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


## Setup Environment
We introduce 2 different options: **GitHub Codespaces**, **VS Code & Remote-Containers**.
You may also use any other environment as long as you install the dependencies in `setup.py`.
Be sure that you have the same environment as us, we recommend you to install like this:

```shell
pip install --upgrade pip
pip install -e ".[tensorflow-cpu,tests]"
echo "sh shell/lint.sh" > .git/hooks/pre-commit
chmod a+x .git/hooks/pre-commit
```

### Option 1: GitHub Codespaces
You can simply open the repository in GitHub Codespaces.
The environment is already setup there.

### Option 2: VS Code & Remote-Containers
Open VS Code.
Install the `Remote-Containers` extension.
Press `F1` key. Enter `Remote-Containers: Open Folder in Container` to open the repository root folder.
The environment is already setup there.

## Run Tests
You can simply open any `*_test.py` file under the `tests` directory,
and wait a few seconds, you will see the test tab on the left of the window.
We use PyTest for the tests, you may also use the `pytest` command to run the tests.

## Code Style
We use `flake8`, `black` and `isort` for linting.
You can run the following manually every time you want to format your code.
1. Run `shell/format.sh` to format your code.
2. Run `shell/lint.sh` to check.

## Rebuilding Protos
If you make changes to any `.proto` file, you'll have to rebuild the generated
`*_pb2.py` files. To do this, run these commands from the root directory of this
project:

```
pip install grpcio-tools
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. keras_tuner/protos/keras_tuner.proto
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. keras_tuner/protos/service.proto
```

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google.com/conduct/).