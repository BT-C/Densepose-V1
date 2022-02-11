# Copyright (c) Facebook, Inc. and its affiliates.

from . import DensePoseChartConfidencePredictorMixin, DensePoseChartPredictor
from densepose.modeling.predictors.chart_confidence import MyDensePoseChartConfidencePredictorMixin
from densepose.modeling.predictors.chart import MyDensePoseChartPredictor
from .registry import DENSEPOSE_PREDICTOR_REGISTRY


@DENSEPOSE_PREDICTOR_REGISTRY.register()
class DensePoseChartWithConfidencePredictor(
    DensePoseChartConfidencePredictorMixin, DensePoseChartPredictor
):
    """
    Predictor that combines chart and chart confidence estimation
    """

    pass



#================================================================================================
@DENSEPOSE_PREDICTOR_REGISTRY.register()
class MyDensePoseChartWithConfidencePredictor(
    MyDensePoseChartConfidencePredictorMixin, MyDensePoseChartPredictor
):
    """
    Predictor that combines chart and chart confidence estimation
    """

    pass
#================================================================================================