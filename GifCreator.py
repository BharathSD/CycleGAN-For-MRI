import imageio
import glob
import os

def createGif(path2imgFiles, anim_file = 'GAN.gif'):
    with imageio.get_writer(anim_file, mode='I') as writer:
        imagesFilter = os.path.join(path2imgFiles, 'image*.png')
        filenames = glob.glob(imagesFilter)
        filenames = sorted(filenames)
        for i in range(0,len(filenames),3):
            image = imageio.imread(filenames[i])
            writer.append_data(image)
        image = imageio.imread(filenames[-1])
        writer.append_data(image)

if __name__ == '__main__':
    imagePath = r'D:\CycleGAN_Assignment\ForGIF'
    createGif(imagePath)