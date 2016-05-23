#!/bin/bash
source pedantry_env/bin/activate
python analysis.py probabilities_smoothed_with_+0.5.csv processed_corpora/ERPA.csv 1 point_five_smoothed_tokens
python analysis.py unsmoothed_probabilities.csv processed_corpora/ERPA.csv 1 unsmoothed_tokens
python analysis.py probabilities_smoothed_with_+0.5.csv processed_corpora/ERPA.csv 0 point_five_smoothed_types
python analysis.py unsmoothed_probabilities.csv processed_corpora/ERPA.csv 0 unsmoothed_types
deactivate