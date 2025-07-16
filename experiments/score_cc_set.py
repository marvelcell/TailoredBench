import datetime
import numpy as np
import os
from tqdm import tqdm
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error,pairwise_distances
import copy
from sklearn.preprocessing import StandardScaler
from config.settings import get_dataset_config
from data.data_loader import ModelDataLoader
from data.data_processor import DataProcessor
from models.clustering import score_cc_set_selection,get_target_model_corrs_models
from utils.file_utils import create_excel_with_sheets, append_to_excel
from utils.parallel_utils import split_dict, worker
from utils.metrics import get_sorted_indices_desc,remove_data_from_dict,compute_correlations

def score_cc_set_experiment(args):
    """
    Execute main experiment pipeline for cc_set selection and evaluation.
    
    Args:
        args: Configuration parameters including:
            datasets_to_run: List of dataset names
            max_dataset_size: Maximum number of examples to use
            point_counts: List of anchor point counts to evaluate
            num_cc_set: Size of initial cc_set
            distance: Distance metric for clustering
            num_runs: Number of experimental repetitions
            num_processes: Parallel processing degree
            corr_fn: Correlation functions dictionary
            
    Returns:
        None (results saved to output files)
    """
    timestamp = datetime.datetime.now().timestamp()
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    formatted_datetime = dt_object.strftime('%Y-%m-%d_%H-%M-%S')
    
    data_loader = ModelDataLoader()
    data_processor = DataProcessor()
    
    for ds in args.datasets_to_run:
        print(f"Loading dataset: {ds}")
        config = get_dataset_config(ds)
        
        # Load dataset
        source_all_data, target_all_data = data_loader.load_dataset(ds)
        source_all_data = source_all_data[:, :args.max_dataset_size]
        target_all_data = target_all_data[:, :args.max_dataset_size]
        
        gt_label = None
        if config['requires_gt']:
            gt_path = config['gt_path']
            gt_label = np.load(gt_path, allow_pickle=True)[:args.max_dataset_size]
            source_models_golddata = data_processor.compute_acc_data(source_all_data, gt_label)
            target_models_golddata = data_processor.compute_acc_data(target_all_data, gt_label)
        else:
            source_models_golddata = source_all_data
            target_models_golddata = target_all_data
        
        true_scores = data_processor.get_dataset_specific_scores(
            ds, target_models_golddata, target_all_data, gt_label
        )
        
        # Prepare output file
        point_counts_str = str(args.point_counts).replace('[', '').replace(']', '').replace(' ', '')
        excel_name = f'./AnchorPoints/{ds}_result/clusterunionset_quantile/excel/dynamic_cpu_ccset_{args.num_cc_set}_cset_{point_counts_str}_time_{formatted_datetime}_distance_{args.distance}.xlsx'
        create_excel_with_sheets(excel_name)
        
        # Run experiment for each point count
        for num_medoids in tqdm(args.point_counts):
            print(f"Running experiment with num_medoids: {num_medoids}")
            all_val_correlations, all_estimated_scores = score_dynamic_source_models_selection(
                source_models_golddata, target_models_golddata,source_all_data,target_all_data,args.num_dynamic_source_models, num_medoids, true_scores, args, ds
            )
            # Compute and save results
            estimate_score_savepath = f'./AnchorPoints/{args.datasets_to_run[0]}_result/clusterunionset_quantile'
            result_save_path = f'./AnchorPoints/{args.datasets_to_run[0]}_result/clusterunionset_quantile/dynamic_cpu_ccset:{args.num_cc_set}_cset:{args.point_counts}_time:{formatted_datetime}_distance:{args.distance}_withllama.txt'
            if not os.path.exists(estimate_score_savepath):
                os.makedirs(estimate_score_savepath)
            
            result = "num_subset: " + str(num_medoids) + "; num_cc_set: " + str(args.num_cc_set) + "; num_runs: " + str(args.num_runs) + "\n"

            values_val_excel = []
            values_err_excel = []
            for method in all_val_correlations:
                val_correlations = all_val_correlations[method]
                val_mean = np.mean(val_correlations)
                val_err = np.std(val_correlations) / np.sqrt(args.num_runs)
                result += "method: " + method + "\n" + "val_mean: " + str(val_mean) + "; val_err: " + str(val_err) + "\n" + str(val_correlations) + "\n"
                values_val_excel.append(round(val_mean, 4))
                values_err_excel.append(round(val_err, 6))
            append_to_excel(excel_name, values_val_excel, values_err_excel)
            print(f"Appended results to Excel for num_medoids: {num_medoids}")

            with open(result_save_path, 'a') as file:
                file.write(result + '\n')

def score_dynamic_source_models_selection(source_models_golddata, target_models_golddata,source_all_data, target_all_data,
                                        num_dynamic_source_models, num_dynamic_anchor_points,
                                        true_scores, args, dataset_name):
    """
    Perform dynamic source model selection and target model score prediction.
    
    Args:
        source_models_golddata: Source model accuracy data (n_models x n_examples)
        target_models_golddata: Target model accuracy data (n_models x n_examples)
        source_all_data: Full source model prediction data (n_models x n_examples x n_classes)
        target_all_data: Full target model prediction data (n_models x n_examples x n_classes)
        num_dynamic_source_models: Number of source models per target
        num_dynamic_anchor_points: Number of clustering anchor points
        true_scores: Ground truth performance scores for target models
        args: Configuration parameters object
        dataset_name: Name of dataset for parameter tuning
        
    Returns:
        val_correlations: Dictionary of correlation metrics across runs
        all_estimated_scores: List of predicted scores per experimental run
    """
    val_correlations = {"pearsonr": [], "kendalltau": [], "spearmanr": [], "MAE": [], "MSE": []}
    all_estimated_scores = []
    all_data = np.concatenate((source_all_data, target_all_data), axis=0)
    gold_data = np.concatenate((source_models_golddata, target_models_golddata), axis=0)
    
    for run in range(args.num_runs):
        print(f"Starting run {run + 1}/{args.num_runs}")
        # Select cc_set using clustering
        cc_set = score_cc_set_selection(source_models_golddata, target_models_golddata,
                                      args.num_cc_set, args.distance, dataset_name)
        cc_set_gold_data = gold_data[ : , cc_set]

        # Identify source and target models
        source_models = list(range(source_all_data.shape[0]))
        target_models = list(set(list(range(all_data.shape[0]))) - set(source_models))
        target_models.sort()
        # Standardize the cc_set gold data
        scaler = StandardScaler()
        cc_set_gold_data = scaler.fit_transform(cc_set_gold_data)   
        
        # Compute distance/correlation matrix
        if args.distance == 'correlation':
            corrs = np.corrcoef(cc_set_gold_data, rowvar=True)
            corrs = 1 - corrs
        else:                                                       # args.distance == 'manhattan' or args.distance == 'euclidean' or args.distance == 'cosine'
            corrs = pairwise_distances(cc_set_gold_data, metric=args.distance)
        # print("finish corrs")
        target_model_corrs_model = corrs[target_models]     

        target_model_corrs_models = {}  
        # Sort model indices based on distance
        for target_model_idx, each_corr in zip(target_models, target_model_corrs_model):
            target_model_corrs_models[target_model_idx] = get_sorted_indices_desc(each_corr)
        target_model_corrs_models = remove_data_from_dict(target_model_corrs_models, target_models)    

        new_num_dynamic_source_models = get_target_model_corrs_models(corrs, source_models, target_models)
        print("new_num_dynamic_source_models: "+str(new_num_dynamic_source_models))
        # Select top source models for each target
        for key in target_model_corrs_models:
            target_model_corrs_models[key] = target_model_corrs_models[key][: new_num_dynamic_source_models]    
            target_model_corrs_models[key].sort()
        
         ##### End of dynamic source model selection via distance metrics #####
        
        # Begin distributed k-medoids clustering for each target model
        target_model_corrs_golddata = {}
        estimated_scores = []                                               # Stores estimated scores for target models

        # Begin distributed k-medoids clustering for each target model
        num_processes = args.num_processes 
        # Split target models into chunks for parallel processing
        target_model_corrs_models_list = split_dict(target_model_corrs_models, num_processes) 

        # Prepare arguments for parallel processing
        pool_args = [( cc_set, chunk_models, process_id, gold_data, num_dynamic_anchor_points, args.distance,dataset_name)
                    for process_id, chunk_models in enumerate(target_model_corrs_models_list)]
        # Execute parallel processing using joblib
        with joblib.Parallel(n_jobs=num_processes) as parallel:
            results = parallel(joblib.delayed(worker)(args) for args in pool_args)

        # Collect and consolidate results from parallel processes

        all_selected_idxs = [None] * num_processes
        for process_id, selected_idxs in results:
            all_selected_idxs[process_id] = copy.deepcopy(selected_idxs)

        estimated_scores = [item for sublist in all_selected_idxs for item in sublist]
        ##### End of distributed k-medoids algorithm #####
        true_target_model_scores = np.array(true_scores)     
        for method in args.corr_fn:   
            corr = args.corr_fn[method](true_target_model_scores, estimated_scores).correlation
            if corr != corr:
                val_correlations[method].append(0)
            else:
                val_correlations[method].append(
                    args.corr_fn[method](true_target_model_scores, estimated_scores).correlation    
                )
        val_correlations['MAE'].append(mean_absolute_error(true_target_model_scores, estimated_scores))
        val_correlations['MSE'].append(mean_squared_error(true_target_model_scores, estimated_scores))

        print('kendalltau: ' + str(val_correlations['kendalltau']))
        print('MAE: ' + str(val_correlations['MAE']))

        all_estimated_scores.append(copy.deepcopy(estimated_scores))
    
    return val_correlations, all_estimated_scores



