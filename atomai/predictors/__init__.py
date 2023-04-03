from .epredictor import EnsemblePredictor, ensemble_locate
from .predictor import (BasePredictor, ImSpecPredictor, Locator, RegPredictor,
                        SegPredictor, clsPredictor)

__all__ = ["BasePredictor", "SegPredictor", "ImSpecPredictor", "RegPredictor",
           "clsPredictor", "EnsemblePredictor", "ensemble_locate", "Locator"]
