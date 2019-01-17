from nflvision import ImgPlotter, ImgLoader, ImgClassifier, nets, losses, optim
from nflvision.ml.model import ImgClassifierEvaluator
from nflvision.components.tracker import ExperimentTracker


def run_mlapp():

    tracker = ExperimentTracker().start_tracking()

    try:
        loader = ImgLoader().build(batch_size=64)

        n_classes = len(loader.train_loader.dataset.class_to_idx)
        experiment_net = nets.CNN(loader.img_size, n_conv_blocks=1, n_classes=n_classes, n_channels_in=3, channel_increase_rate=2)

        model = ImgClassifier().build(net=experiment_net, loss_fn=losses.CrossEntropyLoss, optimizer=optim.Adam)

        model.fit(loader.train_loader, num_epochs=1, init_lr=0.01)

        evaluator = ImgClassifierEvaluator()

        evaluation_result = evaluator.evaluate(model, loader.valid_loader)

        tracker.log_metric("evaluationResult", evaluation_result)
        tracker.log_net(model)

        tracker.log_param("executionResult", "SUCCESS")

    except Exception as e:
        tracker.log_param("executionResult", "FAILURE")


if __name__ == '__main__':

    run_mlapp()

    print("DONE")
