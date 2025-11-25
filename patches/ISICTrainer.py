import inspect
import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import *
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from typing import List
from torch import amp

# needed for the type hints in build_network_architecture
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


# ---- HARD DISABLE torch.compile GLOBALLY ----
if hasattr(torch, "compile"):
    _orig_torch_compile = torch.compile  # keep a handle, just in case

    def _no_compile(model, *args, **kwargs):
        print("[ISICTrainer] torch.compile() disabled – running in eager mode.")
        return model

    torch.compile = _no_compile
# ---------------------------------------------


# if not already imported elsewhere in the file:
try:
    from nnunetv2.training.lr_scheduler.poly_lr import PolyLRScheduler
except ModuleNotFoundError:
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import dummy_context

# --- AMP autocast helper (new & old API) ---
try:
    _AMP_AUTOCAST = torch.amp.autocast  # new API: torch.amp.autocast('cuda')
    def _amp_ctx(device_type: str):
        return _AMP_AUTOCAST(device_type)
except Exception:
    from torch import amp as _OLD_AUTOCAST  # old API: autocast(enabled=True, dtype=...)
    def _amp_ctx(device_type: str):
        return _OLD_AUTOCAST(enabled=True)
# -------------------------------------------



class ISICTrainer(nnUNetTrainer):
    base_ch = 32                      # c = 32 (paper)
    block = "FusedMBConv"
    use_my_unet = True
    network_name = "my_unet"
    project_prefix = "isic"
    setting = 2

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=None, debug=False, job_id=None, *args, **kwargs):
        # Call parent __init__ with only the args it actually supports
        base_init = super(ISICTrainer, self).__init__
        sig = inspect.signature(base_init)
        accepted = set(sig.parameters.keys())

        payload = {
            'plans': plans,
            'configuration': configuration,
            'fold': fold,
            'dataset_json': dataset_json,
            'unpack_dataset': unpack_dataset,
            'device': device,
            'debug': debug,
            'job_id': job_id,
        }
        payload = {k: v for k, v in payload.items() if k in accepted}
        base_init(**payload)

        self.debug = bool(debug)
        # Paper settings
        self.initial_lr = 1e-3
        self.backbone_lr = 1e-3
        self.num_epochs = 80

        if debug:
            self.batch_size = 2
            self.num_iterations_per_epoch = 2
            self.num_val_iterations_per_epoch = 2
        else:
            pass            # self.num_iterations_per_epoch = 2
            # self.batch_size = 12

    def _get_deep_supervision_scales(self):
        pool_op_kernel_sizes = self.configuration_manager.pool_op_kernel_sizes
        # pool_op_kernel_sizes = pool_op_kernel_sizes[:-1]
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            pool_op_kernel_sizes), axis=0))
        # deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
        #     self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-2]

        deep_supervision_scales = deep_supervision_scales[:4]

        return deep_supervision_scales


    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            ignore_lbl = self.label_manager.ignore_label if getattr(self.label_manager, 'ignore_label', None) is not None else 255
            loss = DC_and_CE_loss(
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'smooth': 1e-5,
                    'do_bg': False,
                    'ddp': self.is_ddp
                },
                {},
                weight_ce=1,
                weight_dice=1.5,
                ignore_label=ignore_lbl,
                dice_class=MemoryEfficientSoftDiceLoss
            )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        print(f"ds wegihts: {weights}")

# ===== forced override of build_network_architecture (instance method) =====
def _isic_build_network_architecture(self, plans_manager, dataset_json, configuration_manager,
                                    num_input_channels, enable_deep_supervision=True, *args, **kwargs):
    import json, os
    # Accept dict or path-like for dataset_json
    ds = dataset_json
    try:
        if isinstance(dataset_json, (str, bytes, os.PathLike)):
            with open(dataset_json, 'r') as f:
                ds = json.load(f)
    except Exception:
        ds = dataset_json

    # Derive output_channels
    out_ch = None
    try:
        lm = plans_manager.get_label_manager(ds)
        out_ch = getattr(lm, 'num_labels', None) or getattr(lm, 'num_classes', None)
    except Exception:
        pass
    if out_ch is None:
        try:
            out_ch = len(ds.get('labels', {})) or 1
        except Exception:
            out_ch = 1

    # Ensure at least 2 channels for CE (binary segmentation)
    try:
        oc = int(out_ch) if out_ch is not None else 0
    except Exception:
        oc = 0
    if oc < 2:
        print('[ISICTrainer] forcing out_ch=2 (binary softmax)')
        out_ch = 2

    # Build network
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans as _get_net
    return _get_net(
        plans_manager,
        ds,
        configuration_manager,
        num_input_channels,
        out_ch,
        deep_supervision=enable_deep_supervision
    )

ISICTrainer.build_network_architecture = _isic_build_network_architecture
# ========================================================================

# === ISICTrainer train_step wrapper (maps 255->1) ===
def _isic_map255_and_train_step(self, batch: dict) -> dict:
    """Wrapper that maps 255->1 for binary masks, then calls base train_step."""
    target = batch.get('target', None)
    try:
        ncls = getattr(self.label_manager, 'num_labels', None)
    except Exception:
        ncls = None
    if ncls == 2 and target is not None:
        import torch
        if isinstance(target, list):
            target = [torch.where(t == 255, torch.tensor(1, device=t.device, dtype=t.dtype), t) for t in target]
        else:
            target = torch.where(target == 255, torch.tensor(1, device=target.device, dtype=target.dtype), target)
        batch['target'] = target
    # delegate to base implementation
    return super(ISICTrainer, self).train_step(batch)

# attach the override
ISICTrainer.train_step = _isic_map255_and_train_step

# === override _build_loss to ensure non-None DeepSupervision loss ===
def _isic_build_loss(self):
    import numpy as np
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

    # choose base loss
    if getattr(self.label_manager, 'has_regions', False):
        loss = DC_and_BCE_loss(
            {},
            {
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': True,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            use_ignore_label=getattr(self.label_manager, 'ignore_label', None) is not None,
            dice_class=MemoryEfficientSoftDiceLoss
        )
    else:
        ignore_lbl = getattr(self.label_manager, 'ignore_label', None)
        if ignore_lbl is None:
            ignore_lbl = 255  # common "ignore" in natural-image masks
        loss = DC_and_CE_loss(
            {
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp
            },
            {},
            weight_ce=1,
            weight_dice=1.5,
            ignore_label=ignore_lbl,
            dice_class=MemoryEfficientSoftDiceLoss
        )

    # deep supervision weights
    ds_scales = self._get_deep_supervision_scales()
    weights = np.array([1/(2**i) for i in range(len(ds_scales))], dtype=float)
    weights = weights / weights.sum()
    print(f"ds wegihts: {weights}")  # just informational

    # WRAP & RETURN (critical!)
    return DeepSupervisionWrapper(loss, weights)

# attach override
ISICTrainer._build_loss = _isic_build_loss

# === override train_step with robust label mapping/clamp ===
def _isic_map255_and_train_step(self, batch):
    import torch
    data = batch['data']
    target = batch['target']

    # derive ignore label and n_classes from label_manager
    ignore_lbl = getattr(self.label_manager, 'ignore_label', 255)
    n_classes = None
    for attr in ('num_labels','num_classes'):
        if hasattr(self.label_manager, attr):
            try:
                n_classes = int(getattr(self.label_manager, attr))
                break
            except Exception:
                pass

    def _fix(t):
        # move & cast
        t = t.to(self.device, non_blocking=True).long()
        if n_classes == 2:
            # common ISIC case: (0, 255) or sometimes (0, 2)
            t = torch.where(t == 255, torch.tensor(1, device=t.device, dtype=t.dtype), t)
            t = torch.clamp(t, 0, 1)  # ensure in (0, 1)
        else:
            if n_classes is not None:
                bad = (t < 0) | (t >= n_classes)
                if bad.any():
                    t = torch.where(bad, torch.tensor(ignore_lbl, device=t.device, dtype=t.dtype), t)
        return t

    if isinstance(target, list):
        target = [_fix(x) for x in target]
    else:
        target = _fix(target)

    # one-time visibility
    if not hasattr(self, '_printed_label_stats'):
        first = target[0] if isinstance(target, list) else target
        try:
            u = torch.unique(first).detach().cpu().tolist()
        except Exception:
            u = ['<unavailable>']
        print(f"[ISICTrainer] labels after mapping/clamp: {u} (ignore={ignore_lbl}, n_classes={n_classes})")
        self._printed_label_stats = True

    # forward to base trainer using the adjusted target
    return super(ISICTrainer, self).train_step({'data': data, 'target': target})

# bind override
ISICTrainer.train_step = _isic_map255_and_train_step
