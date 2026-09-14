# Importing these modules populates TASK_REGISTRY, PHYSICS_DECODER_REGISTRY
# and LOSS_REGISTRY via their decorators.
import gridfm_app_dist.tasks.utils  # noqa: F401
# physics decoders (OPF/PF/SE/VLD)
# import gridfm_app_dist.training.loss  # loss functions incl. VLDTopologyLoss

from gridfm_app_dist.tasks.pf_task import PowerFlowTask
from gridfm_app_dist.tasks.opf_task import OptimalPowerFlowTask
from gridfm_app_dist.tasks.se_task import StateEstimationTask
from gridfm_app_dist.tasks.vld_task import VoltageLossDetectionTask

__all__ = [
    "PowerFlowTask",
    "OptimalPowerFlowTask",
    "StateEstimationTask",
    "VoltageLossDetectionTask",
]
