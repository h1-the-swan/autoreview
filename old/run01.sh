#!/usr/bin/zsh
nohup python pipeline_experiments.py data/collect_haystack_2127048411_seed-3 --seed 999 --debug >& data/collect_haystack_2127048411_seed-3/pipeline_experiments_different_classifiers_featuresAvgDistanceAndEFAndAvgCosSim_20180517.log &
nohup python pipeline_experiments.py data/collect_haystack_2127048411_seed-4 --seed 999 --debug >& data/collect_haystack_2127048411_seed-4/pipeline_experiments_different_classifiers_featuresAvgDistanceAndEFAndAvgCosSim_20180517.log &
nohup python pipeline_experiments.py data/collect_haystack_2127048411_seed-1 --seed 999 --debug >& data/collect_haystack_2127048411_seed-1/pipeline_experiments_different_classifiers_featuresAvgDistanceAndEFAndAvgCosSim_20180517.log &
nohup python pipeline_experiments.py data/collect_haystack_2127048411_seed-2 --seed 999 --debug >& data/collect_haystack_2127048411_seed-2/pipeline_experiments_different_classifiers_featuresAvgDistanceAndEFAndAvgCosSim_20180517.log &
nohup python pipeline_experiments.py data/collect_haystack_2127048411_seed-5 --seed 999 --debug >& data/collect_haystack_2127048411_seed-5/pipeline_experiments_different_classifiers_featuresAvgDistanceAndEFAndAvgCosSim_20180517.log &
