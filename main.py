import argparse
import logging
from experiments.score_cc_set import score_cc_set_experiment
from scipy.stats import pearsonr, kendalltau, spearmanr
from config.settings import LOGGING_CONFIG

def main():
    parser = argparse.ArgumentParser("Probability Prediction Baseline")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--results_dir", type=str, default="")
    parser.add_argument("--corr_type", type=list, choices=["pearsonr", "kendalltau", "spearmanr"], default=["pearsonr", "kendalltau", "spearmanr"])
    parser.add_argument("--max_samples", type=int, default=6000)
    parser.add_argument("--datasets_to_run", nargs="+", default=["arc_challenge", "hellaswag", "gsm8k", "winogrande", "POPE"])
    parser.add_argument("--max_dataset_size", type=int, default=6000)
    parser.add_argument("--num_source_models", type=int, default=69)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--point_counts", type=int, nargs='+', default=[10,15,20,25,30])    # Number of N-set size **excluding** G-set size **(!!! Very Very Important !!!)** point_counts + num_cc_set(G-set) = N-set 
    parser.add_argument("--exp", type=str, default="score_cc_set")
    parser.add_argument("--num_cc_set", type=int, default=10)
    parser.add_argument("--num_processes", type=int, default=24)
    parser.add_argument("--distance", type=str, default="manhattan")
    
    args = parser.parse_args()
    args.corr_fn = {"pearsonr": pearsonr, "kendalltau": kendalltau, "spearmanr": spearmanr}
    
    logging.basicConfig(level=logging.INFO)
    if args.exp == 'score_cc_set':
        score_cc_set_experiment(args)

if __name__ == "__main__":
    main()