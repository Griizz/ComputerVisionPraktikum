#Allgemeine Notizen für den Bericht

##Batchsize
Finde die best batch-size für 4 GB VRAM.
Eine Grafikkarte kann alle Bilder simultan überprüfen und nicht wie eine CPU nur seriell.

Eine größere Batchsize erhöht das Trainingstempo.

Learningrate sollte linear zur Batchsize skaliert werden. 




##Learningrate
Zu große Learningrate führt dazu, dass die Gewichte zu schnell verstellt werden
und man bei einer Trefferquote von etwa 1/n (n = Anzahl der Kategorien) bleibt.
"Der Computer ratet."

Eine zu kleine Learning rate wiederum, führt entweder dazu, dass man in einem lokalen Minimum zu schnell stecken bleibt
und das Training schnell beendet wird. Oder es verschwendet Rechenzeit, weil dass selbe Ergebnis mit eine höheren
Learningrate schneller erreicht worden wäre.

## Plateau-optimierung
Keras hat einen eingbauten Callback *ReduceLROnPlateau()*, welcher der LR langsam redeuziert, sobald ein "Becken" erreciht
wurde, aus dem es nicht mehr "herraus kommt". So wird das lokale Minimum gefunden. Das optimiert die **Acc** am Ende.