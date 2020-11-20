from core.SpeechBubblesExperiment import SpeechBubblesExperiment
from core.svgparser import get_all_configs
from methods.speechbubbles.ml import ml_method

experiment = SpeechBubblesExperiment()
tmp = experiment.add_method("ML", ml_method)

experiment.execute_method_for_all(tmp)