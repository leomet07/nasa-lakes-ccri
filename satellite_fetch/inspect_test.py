from matplotlib import pyplot as plt
import rasterio
import math
import numpy as np

out_file = input("Enter the path of a tif to inspect: ") or "out/prospect.tif"

def inspect(filepath: str):
    with rasterio.open(filepath) as src:
        # Read the number of bands and the dimensions
        num_bands = src.count
        height = src.height
        width = src.width
        print(src.profile)
        print(src.tags())


        print(f"Number of bands: {num_bands}")
        print(f"Dimensions: {width} x {height}")

        # Read the entire image into a numpy array (bands, height, width)
        img = src.read()

        number_of_bands = img.shape[0]

        if number_of_bands == 1:
            print("band data: ")
            with np.printoptions(threshold=np.inf):
                print(img[0])
            print(filepath)
        # Display each band separately
        fig, axes = plt.subplots(nrows=3, ncols=math.ceil(float(number_of_bands) / 3), figsize=(15, 10))

        for i, ax in enumerate(axes.flatten()):
            if i < num_bands:
                ax.imshow(img[i, :, :], cmap="gray")  # Display each band separately
                ax.set_title(f"Band {i+1}")
                ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    inspect(out_file)                
