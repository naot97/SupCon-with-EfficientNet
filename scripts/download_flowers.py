import urllib.request
import tarfile
import os


FLOWERS_DIR = './data'


def download_images():
    """If the images aren't already downloaded, save them to FLOWERS_DIR."""
    if not os.path.exists(FLOWERS_DIR):
        DOWNLOAD_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
        print('Downloading flower images from %s...' % DOWNLOAD_URL)
        urllib.request.urlretrieve(DOWNLOAD_URL, 'flower_photos.tgz')
        tar = tarfile.open('flower_photos.tgz')
        tar.extractall(path=FLOWERS_DIR)
        tar.close()
    print('Flower photos are located in %s' % FLOWERS_DIR)
