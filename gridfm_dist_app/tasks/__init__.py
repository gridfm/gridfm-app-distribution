# Importing these modules populates TASK_REGISTRY, PHYSICS_DECODER_REGISTRY
# and LOSS_REGISTRY via their decorators.
import gridfm_dist_app.tasks.utils  # physics decoders (OPF/PF/SE/VLD)
import gridfm_dist_app.training.loss  # loss functions incl. VLDTopologyLoss

from gridfm_dist_app.tasks.pf_task import PowerFlowTask
from gridfm_dist_app.tasks.opf_task import OptimalPowerFlowTask
from gridfm_dist_app.tasks.se_task import StateEstimationTask
from gridfm_dist_app.tasks.vld_task import VoltageLossDetectionTask

__all__ = [
    "PowerFlowTask",
    "OptimalPowerFlowTask",
    "StateEstimationTask",
    "VoltageLossDetectionTask",
]
