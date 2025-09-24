try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass


from contextlib import contextmanager
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import copy
import itertools
import logging

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import random
import numpy as np

from datetime import timedelta
import torch.distributed as dist

orig_init_pg = dist.init_process_group
def init_pg_with_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", timedelta(hours=2))
    return orig_init_pg(*args, **kwargs)

dist.init_process_group = init_pg_with_timeout

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config
from models import (
    AVISDatasetMapper,
    AVISEvaluator,
    build_detection_train_loader,
    build_detection_test_loader,
    add_avism_config,
)


from detectron2.utils.events import get_event_storage
import detectron2.utils.comm as comm

def _flatten(d, parent=""):
    """
    {'segm': {'AP': 0.41, 'AR': 0.55}} -->
    {'segm/AP': 0.41, 'segm/AR': 0.55}
    """
    out = {}
    for k, v in d.items():
        nk = f"{parent}{k}" if parent == "" else f"{parent}/{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, nk))
        else:
            out[nk] = v
    return out

from detectron2.engine.hooks import EvalHook


from detectron2.engine import hooks as d2_hooks
from detectron2.engine.hooks import BestCheckpointer

import wandb
from detectron2.utils.events import EventWriter, get_event_storage

class SaveBestHook(d2_hooks.HookBase):
    """
    After *another* evaluation hook has written val metrics into EventStorage,
    save the checkpoint iff the tracked metric improves.
    """
    def __init__(self, metric="val/segm/AP_all", mode="max", fname="model_best"):
        self.metric   = metric
        self.mode     = mode
        self.fname    = fname
        self.best_val = None

    def after_step(self):
        storage = get_event_storage()
        if self.metric not in storage.latest():     
            return

        cur_val, _ = storage.latest()[self.metric]
        better = (
            cur_val >  self.best_val if self.mode == "max"
            else cur_val < self.best_val
        ) if self.best_val is not None else True   

        if better:
            self.best_val = cur_val
            self.trainer.checkpointer.save(self.fname)

@contextmanager
def force_single_gpu():
    """
    Temporarily make detectron2 think we're in single-GPU mode.
    Only patches comm.get_world_size / comm.get_rank,
    so training afterwards is untouched.
    """
    orig_ws   = comm.get_world_size
    orig_rank = comm.get_rank
    comm.get_world_size = lambda: 1
    comm.get_rank       = lambda: 0
    try:
        yield
    finally:                 # ← 꼭 복원해야 학습이 정상 진행
        comm.get_world_size = orig_ws
        comm.get_rank       = orig_rank

class WandbEvalHook(EvalHook):
    def _do_eval(self):
        if not comm.is_main_process():
            comm.synchronize()
            return {}

        model = self.trainer.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        was_training = model.training
        model.eval()

        with force_single_gpu():
            results = Trainer.test(self.trainer.cfg, model)

        flat = _flatten(results)
        storage = get_event_storage()
        storage.put_scalars(
            **{f"val/{k}": v for k, v in flat.items() if not isinstance(v, list)},
            smoothing_hint=False
        )

        comm.synchronize()
        if was_training:
            model.train()
        return results



class WandbWriter(EventWriter):
    """
    Minimal drop-in replacement of detectron2>=0.7 WandbWriter.
    Logs every scalar in EventStorage to W&B.
    """

    def __init__(self,
                 window_size: int = 20,
                 prefix: str = "",
                 project: str = "avis",
                 run_name: str = None):
        """
        Args:
            window_size: median window for noisy scalars
            prefix:   logged key will be '<prefix><scalar_name>'
            project:  wandb project
            run_name: wandb run name (None → let wandb choose)
        """
        self.window_size = window_size
        self.prefix = prefix
        self._last_write = -1     
        self._inited = False
        self.project = project
        self.run_name = run_name
        self._seen = set()       

    def _maybe_init(self):
        if self._inited:
            return
        from detectron2.utils import comm
        if comm.is_main_process():
            wandb.init(project=self.project,
                       name=self.run_name,
                       config={}, 
                       resume="allow")
        else:
            wandb.init(mode="disabled")
        self._inited = True

    def write(self):
        self._maybe_init()
        storage = get_event_storage()

        metrics = {}
        newest_iter = -1

        for k, (v, itr) in storage.latest_with_smoothing_hint(self.window_size).items():
            if (itr, k) in self._seen:  
                continue
            self._seen.add((itr, k))

            metrics[self.prefix + k] = v
            newest_iter = max(newest_iter, itr)

        if metrics:
            wandb.log(metrics, step=newest_iter)  
        if self._inited:
            wandb.log({}, commit=True)

    def close(self):
        if self._inited:
            wandb.finish()




class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super().__init__(cfg)  
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            bare_model = self.model.module
        else:                        
            bare_model = self.model

        if comm.get_world_size() > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                bare_model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        return AVISEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = AVISDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN[0])
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        mapper = AVISDatasetMapper(cfg, is_train=False, need_gt = True)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """

        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i

            print("AP: {} || AP_s: {} || AP_m: {} || AP_l: {} || AR: {}".format(results_i['segm']['AP_all'],
                                                                                results_i['segm']['AP_s'],
                                                                                results_i['segm']['AP_m'],
                                                                                results_i['segm']['AP_l'],
                                                                                results_i['segm']['AR_all']))

            print("DetA: {} || DetRe: {} || DetPr: {}".format(results_i['segm']['DetA'],
                                                                results_i['segm']['DetRe'],
                                                                results_i['segm']['DetPr']))

            print("AssA: {} || AssRe: {} || AssPr: {}".format(results_i['segm']['AssA'],
                                                                results_i['segm']['AssRe'],
                                                                results_i['segm']['AssPr']))

            print("HOTA: {} || LocA: {} || DetA: {} || AssA: {}".format(results_i['segm']['HOTA'],
                                                                        results_i['segm']['LocA'],
                                                                        results_i['segm']['DetA'],
                                                                        results_i['segm']['AssA']))

            print("FSLAn_count: {} || FSLAn_all: {} || FSLAs_count: {} || FSLAs_all: {} || FSLAm_count: {} || FSLAm_all: {}".format(
                results_i['segm']['FAn_count'],
                results_i['segm']['FAn_all'],
                results_i['segm']['FAs_count'],
                results_i['segm']['FAs_all'],
                results_i['segm']['FAm_count'],
                results_i['segm']['FAm_all']))

            print("FSLA: {} || FSLAn: {} || FSLAs: {} || FSLAm: {}".format(results_i['segm']['FA'],
                                                                            results_i['segm']['FAn'],
                                                                            results_i['segm']['FAs'],
                                                                            results_i['segm']['FAm']))
            
            print("sound accuracy: {}%".format(results_i["sound_acc"]["accurate"] / (results_i["sound_acc"]["total"] + 1e-6)* 100))
            print("sound accuracy per class:", results_i["sound_acc_per_cls"])

        if len(results) == 1:
            results = list(results.values())[0]
        return results


    def build_writers(self):
        if self.cfg.eval_only:
            return super().build_writers()
    
        writers = super().build_writers()
        if comm.is_main_process():
            writers.append(
                WandbWriter(
                    window_size=20,
                    prefix="",
                    project="avis",
                        run_name=self.cfg.OUTPUT_DIR[10:]
                )
            )
        return writers
    
    def build_hooks(self):
        hooks = super().build_hooks()

        hooks = [h for h in hooks if not isinstance(h, d2_hooks.EvalHook)]

        eval_period = self.cfg.TEST.EVAL_PERIOD
        hooks.append(
            WandbEvalHook(eval_period,
                        lambda: Trainer.test(self.cfg, self.model))
        )

        hooks.append(
            SaveBestHook(metric="val/segm/AP_all",  # 원하는 지표
                        mode="max",
                        fname="model_best")
        )
        
        hooks.append(
            d2_hooks.PeriodicCheckpointer(
                self.checkpointer,
                period=self.cfg.SOLVER.CHECKPOINT_PERIOD,
                max_to_keep=self.cfg.SOLVER.MAX_ITER
            )
        )
        return hooks

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_avism_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg["eval_only"] = args.eval_only
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="avism")
    return cfg

import wandb

def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

    if comm.is_main_process():
        wandb.init(
            project="avis",               
            name=cfg.OUTPUT_DIR[10:]
        )
    else:  
        wandb.init(mode="disabled")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Trainable params: {num_params/1e6:.2f} M")
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)


    num_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {num_params/1e6:.2f} M")
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)


    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
