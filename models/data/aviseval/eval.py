import os
import tqdm
import traceback
import numpy as np

from . import utils
from . import _timing
from .metrics import Count
from .utils import TrackEvalException
from .metrics import compute_av_loc, combine_av_loc_sequences


class Evaluator:
    """Evaluator class for evaluating different metrics for different datasets"""

    @staticmethod
    def get_default_eval_config():
        """Returns the default config values for evaluation"""
        code_path = utils.get_code_path()
        default_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.

            'PRINT_RESULTS': False,
            'PRINT_ONLY_COMBINED': False,
            'PRINT_CONFIG': False,
            'TIME_PROGRESS': False,
            'DISPLAY_LESS_PROGRESS': True,

            'OUTPUT_SUMMARY': False,
            'OUTPUT_EMPTY_CLASSES': False,
            'OUTPUT_DETAILED': False,
            'PLOT_CURVES': False,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_eval_config(), 'Eval')
        # Only run timing analysis if not run in parallel.
        if self.config['TIME_PROGRESS'] and not self.config['USE_PARALLEL']:
            _timing.DO_TIMING = True
            if self.config['DISPLAY_LESS_PROGRESS']:
                _timing.DISPLAY_LESS_PROGRESS = True

    @_timing.time
    def evaluate(self, dataset_list, metrics_list):
        """Evaluate a set of metrics on a set of datasets"""
        config = self.config
        metrics_list = metrics_list + [Count()]  # Count metrics are always run
        metric_names = utils.validate_metrics_list(metrics_list)
        dataset_names = [dataset.get_name() for dataset in dataset_list]
        output_res = {}
        output_msg = {}

        for dataset, dataset_name in zip(dataset_list, dataset_names):
            # Get dataset info about what to evaluate
            output_res[dataset_name] = {}
            output_msg[dataset_name] = {}
            tracker_list, seq_list, class_list = dataset.get_eval_info()

            # Evaluate each tracker
            for tracker in tracker_list:
                # if not config['BREAK_ON_ERROR'] then go to next tracker without breaking
                try:
                    print('\nEvaluating model ...... \n')
                    res = {}
                    res_av_loc = {}

                    seq_list_sorted = sorted(seq_list)
                    for curr_seq in tqdm.tqdm(seq_list_sorted):
                        res[curr_seq] = eval_sequence(curr_seq, dataset, tracker, class_list, metrics_list, metric_names)
                        res_av_loc[curr_seq] = eval_av_loc_sequence(curr_seq, dataset, tracker)

                    # Combine results over all sequences and then over all classes
                    res_av_loc_all = combine_av_loc_sequences(res_av_loc)

                    # collecting combined cls keys (cls averaged, det averaged, super classes)
                    combined_cls_keys = []
                    res['COMBINED_SEQ'] = {}
                    # combine sequences for each class
                    for c_cls in class_list:
                        res['COMBINED_SEQ'][c_cls] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            curr_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value in res.items() if
                                        seq_key != 'COMBINED_SEQ'}
                            res['COMBINED_SEQ'][c_cls][metric_name] = metric.combine_sequences(curr_res)
                    # combine classes
                    if dataset.should_classes_combine:
                        combined_cls_keys += ['cls_comb_cls_av', 'cls_comb_det_av', 'all']
                        res['COMBINED_SEQ']['cls_comb_cls_av'] = {}
                        res['COMBINED_SEQ']['cls_comb_det_av'] = {}
                        for metric, metric_name in zip(metrics_list, metric_names):
                            cls_res = {cls_key: cls_value[metric_name] for cls_key, cls_value in
                                       res['COMBINED_SEQ'].items() if cls_key not in combined_cls_keys}
                            res['COMBINED_SEQ']['cls_comb_cls_av'][metric_name] = \
                                metric.combine_classes_class_averaged(cls_res)
                            res['COMBINED_SEQ']['cls_comb_det_av'][metric_name] = \
                                metric.combine_classes_det_averaged(cls_res)
                    # combine classes to super classes
                    if dataset.use_super_categories:
                        for cat, sub_cats in dataset.super_categories.items():
                            combined_cls_keys.append(cat)
                            res['COMBINED_SEQ'][cat] = {}
                            for metric, metric_name in zip(metrics_list, metric_names):
                                cat_res = {cls_key: cls_value[metric_name] for cls_key, cls_value in
                                           res['COMBINED_SEQ'].items() if cls_key in sub_cats}
                                res['COMBINED_SEQ'][cat][metric_name] = metric.combine_classes_det_averaged(cat_res)

                    # Print and output results in various formats
                    output_fol = dataset.get_output_fol(tracker)
                    tracker_display_name = dataset.get_display_name(tracker)
                    for c_cls in res['COMBINED_SEQ'].keys():  # class_list + combined classes if calculated
                        summaries = []
                        details = []
                        num_dets = res['COMBINED_SEQ'][c_cls]['Count']['Dets']
                        if config['OUTPUT_EMPTY_CLASSES'] or num_dets > 0:
                            for metric, metric_name in zip(metrics_list, metric_names):
                                # for combined classes there is no per sequence evaluation
                                if c_cls in combined_cls_keys:
                                    table_res = {'COMBINED_SEQ': res['COMBINED_SEQ'][c_cls][metric_name]}
                                else:
                                    table_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value in res.items()}
                                if config['PLOT_CURVES']:
                                    metric.plot_single_tracker_results(table_res, tracker_display_name, c_cls, output_fol)
                            if config['OUTPUT_SUMMARY']:
                                utils.write_summary_results(summaries, c_cls, output_fol)
                            if config['OUTPUT_DETAILED']:
                                utils.write_detailed_results(details, c_cls, output_fol)

                    # Output for returning from function
                    res_output = {}

                    res_output["AP_all"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['TrackMAP']['AP_all']), 2)
                    res_output["AP_s"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['TrackMAP']['AP_area_s']), 2)
                    res_output["AP_m"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['TrackMAP']['AP_area_m']), 2)
                    res_output["AP_l"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['TrackMAP']['AP_area_l']), 2)
                    res_output["AR_all"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['TrackMAP']['AR_all']), 2)

                    res_output["HOTA"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['HOTA']), 2)
                    res_output["DetA"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['DetA']), 2)
                    res_output["DetRe"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['DetRe']), 2)
                    res_output["DetPr"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['DetPr']), 2)
                    res_output["AssA"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['AssA']), 2)
                    res_output["AssRe"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['AssRe']), 2)
                    res_output["AssPr"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['AssPr']), 2)
                    res_output["LocA"] = round(100 * np.mean(res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['LocA']), 2)

                    res_output["FA"] = round(100 * np.mean(res_av_loc_all['FA']), 2)
                    res_output["FAn"] = round(100 * np.mean(res_av_loc_all['FAn']), 2)
                    res_output['FAn_count'] = int(np.mean(res_av_loc_all['FAn_count']))
                    res_output['FAn_all'] = int(np.mean(res_av_loc_all['FAn_all']))
                    res_output["FAs"] = round(100 * np.mean(res_av_loc_all['FAs']), 2)
                    res_output['FAs_count'] = int(np.mean(res_av_loc_all['FAs_count']))
                    res_output['FAs_all'] = int(np.mean(res_av_loc_all['FAs_all']))
                    res_output["FAm"] = round(100 * np.mean(res_av_loc_all['FAm']), 2)
                    res_output['FAm_count'] = int(np.mean(res_av_loc_all['FAm_count']))
                    res_output['FAm_all'] = int(np.mean(res_av_loc_all['FAm_all']))

                    output_res[dataset_name][tracker] = res_output
                    output_msg[dataset_name][tracker] = 'Success'

                except Exception as err:
                    output_res[dataset_name][tracker] = None
                    if type(err) == TrackEvalException:
                        output_msg[dataset_name][tracker] = str(err)
                    else:
                        output_msg[dataset_name][tracker] = 'Unknown error occurred.'
                    print('Tracker %s was unable to be evaluated.' % tracker)
                    print(err)
                    traceback.print_exc()
                    if config['LOG_ON_ERROR'] is not None:
                        with open(config['LOG_ON_ERROR'], 'a') as f:
                            print(dataset_name, file=f)
                            print(tracker, file=f)
                            print(traceback.format_exc(), file=f)
                            print('\n\n\n', file=f)
                    if config['BREAK_ON_ERROR']:
                        raise err
                    elif config['RETURN_ON_ERROR']:
                        return output_res, output_msg

        return output_res, output_msg


@_timing.time
def eval_sequence(seq, dataset, tracker, class_list, metrics_list, metric_names):
    """Function for evaluating a single sequence"""

    raw_data = dataset.get_raw_seq_data(tracker, seq)
    seq_res = {}
    for cls in class_list:
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls)
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[cls][met_name] = metric.eval_sequence(data)
    return seq_res


def eval_av_loc_sequence(seq, dataset, tracker):
    """Function for evaluating a single sequence"""

    raw_data = dataset.get_raw_seq_data(tracker, seq)
    av_loc_res = compute_av_loc(raw_data)

    return av_loc_res