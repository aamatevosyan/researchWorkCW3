from core.SpeechBubblesExperiment import SpeechBubblesExperiment
from methods.speechbubbles.ml import ml_method
from speechbubbles.opencv_advanced import opencv_advanced_method
from speechbubbles.opencv_basic import opencv_basic_method
from utils import prepare_dataset_for_speech_bubbles_experiment

if __name__ == '__main__':
    prepare_dataset_for_speech_bubbles_experiment()

    experiment = SpeechBubblesExperiment()
    # experiment.add_method("ML", ml_method)
    # experiment.add_method("opencv_basic_method", opencv_basic_method)
    experiment.add_method("opencv_advanced_method", opencv_advanced_method)

    experiment.execute_all_methods()
    experiment.get_bounding_box_for_all_methods()
    # experiment.get_scores_for_all_methods()


