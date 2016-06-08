#!/bin/bash
source venv/bin/activate
python analysis.py test
python analysis.py somethingelse
deactivate