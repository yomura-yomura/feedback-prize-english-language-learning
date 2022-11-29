#!/bin/sh

python3 setup.py bdist_wheel
python3 -m pip download --only-binary :all: . -d dist/wheels
kaggle datasets version -p dist/ --dir-mode skip -m ""
kaggle datasets version -p dist/wheels --dir-mode skip -m ""
