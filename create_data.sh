#!/bin/bash
source venv/bin/activate
python analysis.py data point_five_smoothed_tokens probabilities_smoothed_with_+0.5.csv processed_corpora/ERPA.csv 1 
python analysis.py data unsmoothed_tokens unsmoothed_probabilities.csv processed_corpora/ERPA.csv 1 
python analysis.py data point_five_smoothed_types probabilities_smoothed_with_+0.5.csv processed_corpora/ERPA.csv 0 
python analysis.py data unsmoothed_types unsmoothed_probabilities.csv processed_corpora/ERPA.csv 0 
deactivate