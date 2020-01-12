from os import walk
from shutil import copyfile

_, folders, _ = walk("./DataSet").__next__()

for folder in folders:
    sourceTr = "./DataSet/" + folder + "/Training/"
    sourceTest = "./DataSet/" + folder + "/Test/"

    destinationTr = "./DataSetNew/Training/" + folder + "/"
    destinationTest = "./DataSetNew/Test/" + folder + "/"
    destinationVal = "./DataSetNew/Validation/" + folder + "/"

    _, _, trFilenames = walk(sourceTr).__next__()
    _, _, testFilenames = walk(sourceTest).__next__()

    for file in trFilenames:
        copyfile(sourceTr + file, destinationTr + file)

    for i in range(150):
        copyfile(sourceTest + testFilenames[i], destinationVal + testFilenames[i])

    for i in range(150, 200):
        copyfile(sourceTest + testFilenames[i], destinationTest + testFilenames[i])
