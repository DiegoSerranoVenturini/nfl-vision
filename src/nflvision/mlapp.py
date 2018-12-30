from nflvision import ImgPlotter, ImgLoader


def run_mlapp():

    img_loader = ImgLoader.build()

    for img in img_loader:

        print(img)


if __name__ == '__main__':

    run_mlapp()

    print("DONE")
