import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pytorch import log_model

from nflvision.cfg.constants import TrackingConstants
from nflvision.ml.model import ImgClassifier


class ExperimentTracker:

    def start_tracking(self):

        experiment_name = "nfl-vision"

        mlflow.set_tracking_uri(TrackingConstants.TRACKING_URI)

        experiment_list = {exp.name: exp.experiment_id for exp in MlflowClient().list_experiments()}
        if experiment_name not in experiment_list.keys():
            mlf_experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=TrackingConstants.ARTIFACT_LOCATION)
        else:
            mlf_experiment_id = experiment_list[experiment_name]

        mlflow.start_run(experiment_id=mlf_experiment_id)

        return self


    @staticmethod
    def log_metric(metric_name, value):
        mlflow.log_metric(metric_name, value)

    @staticmethod
    def log_param(param_name, value):
        mlflow.log_param(param_name, value)

    @staticmethod
    def log_net(model: ImgClassifier):
        log_model(model.estimator, TrackingConstants.NET_PATH)






