from nflvision import ImgPlotter, ImgLoader, ImgModelBuilder, nets, losses, optim


def run_mlapp():

    loader = ImgLoader().build(batch_size=64)

    experiment_net = nets.CNN(loader.img_size, n_conv_blocks=1, n_classes=2, n_channels_in=3, channel_increase_rate=2)

    model = ImgModelBuilder().build(net=experiment_net, loss_fn=losses.CrossEntropyLoss, optimizer=optim.Adam)

    model.fit(loader.train_loader, num_epochs=1, init_lr=0.01)


if __name__ == '__main__':

    run_mlapp()

    print("DONE")
