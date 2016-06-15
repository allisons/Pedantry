#!/bin/bash
source venv/bin/activate
python analysis.py bootstrap point_five_smoothed_tokens outputfiles/ERPA-stats_point_five_smoothed_tokens.csv
python analysis.py bootstrap unsmoothed_tokens ERPA-stats_unsmoothed_tokens.csv
python analysis.py bootstrap point_five_smoothed_types ERPA-stats_point_five_smoothed_types.csv
python analysis.py bootstrap unsmoothed_types ERPA-stats_unsmoothed_types.csv
deactivate