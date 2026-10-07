"""Model registry for the OpenADMET CYP challenge.

To add a new model: create a module in this directory, subclass CYPModel,
set a unique `name` class variable, and add it here.
"""

from models.base import CYPModel
from models.decision_tree import DecisionTree
from models.ridge import Ridge
from models.random_forest import RandomForest
from models.gbm import LightGBM
from models.XGBoost import XGBoost

REGISTRY: dict[str, type[CYPModel]] = {
    DecisionTree.name: DecisionTree,
    Ridge.name: Ridge,
    RandomForest.name: RandomForest,
    LightGBM.name: LightGBM,
    XGBoost.name: XGBoost,
}

# tabpfn is an optional dependency: it pulls a large checkpoint and is not
# needed by the other models, so a missing install drops `tabpfn` from the
# registry rather than breaking every CLI that imports this module.
try:
    from models.TabPFN import TabPFN
except ImportError:  # pragma: no cover - depends on the environment
    pass
else:
    REGISTRY[TabPFN.name] = TabPFN

__all__ = ["CYPModel", "REGISTRY"]
