from os import walk
from random import sample
from shutil import copyfile

def CopyRandom(source, destination, number, prefix):
    """
    :param source: The path where to choose random file from
    :param destination: The path to save the chosen files at.
    :param number: The amount of files getting chosen randomly
    :param prefix: The Prefix for the new filename followed up by the number.
    """
    if number % 5 != 0:
        return -1

    split = int(number * 0.8)
    _, _, filenames = walk(source).__next__()
    selected = sample(filenames, number)

    for n in range(split):
        copyfile(source + selected[n], destination + "Training/" + prefix + "_" + str(n) + ".png")

    for n in range(split, number):
        copyfile(source + selected[n], destination + "Test/" + prefix + "_" + str(n) + ".png")

    return 0


_, folders, _ = walk("./DataSetUnsorted").__next__()

for folder in folders:
    source = "./DataSetUnsorted/" + folder + "/"
    destination = "./DataSet/" + folder + "/"
    CopyRandom(source, destination,  1000, folder)
