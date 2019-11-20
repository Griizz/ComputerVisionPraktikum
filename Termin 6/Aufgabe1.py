
import numpy as np
import matplotlib.pyplot as plt
import bilderGenerator as bg

TrainingsBilder = bg.zieheBilder(250)
ValidierungsBilder = bg.zieheBilder(25)

w1, w2 = np.random.normal(0, 0.001, 2)
b = 0
alpha = 0.0000005

plt.close("all")
fig, ax = plt.subplots(1, 1)

cat1 = np.where(TrainingsBilder[2] == -1)
cat2 = np.where(TrainingsBilder[2] == 1)

#Aufgabe 1.2
for x in cat1:
    plt.plot(TrainingsBilder[0][x], TrainingsBilder[1][x], "go")

for x in cat2:
    plt.plot(TrainingsBilder[0][x], TrainingsBilder[1][x], "ro")

#Aufgabe 1.3
yTrain = np.sign(TrainingsBilder[0] * 0.0001 + TrainingsBilder[1] * -0.0002 + 0.001)
yValidation = np.sign(ValidierungsBilder[0] * 0.0001 + ValidierungsBilder[1] * -0.0002 + 0.001)

rTraining = yTrain == TrainingsBilder[2]
rValidation = yValidation == ValidierungsBilder[2]
print(np.sum(rValidation) / 50)

#Aufgabe 1.4
result = [0] * 100
for i in range (100):
    for x in range(500):
        p = TrainingsBilder[0][x] * w1 + TrainingsBilder[1][x] * w2 + b
        if np.sign(p) != TrainingsBilder[2][x]:
           w1 = w1 - alpha * 2 * (p - TrainingsBilder[2][x]) * TrainingsBilder[0][x]
           w2 = w2 - alpha * 2 * (p - TrainingsBilder[2][x]) * TrainingsBilder[1][x]
           b = b - alpha * 2 * (p - TrainingsBilder[2][x])

    yValidation = np.sign(ValidierungsBilder[0] * w1 + ValidierungsBilder[1] * w2 + b)
    rValidation = yValidation == ValidierungsBilder[2]
    result[i] = np.sum(rValidation) / 50

for x in np.arange(0.0, 255.1, 0.1):
    for y in np.arange(0.0, 128.1, 0.1):
        p = x * w1 + y * w2 + b
        if np.abs(p) < 0.0001:
            plt.plot(x, y, "bx")

print(result)
print(np.max(result))
