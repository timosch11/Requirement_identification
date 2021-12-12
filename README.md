# Requirement_identification

Ich habe hier ein Codebeispiel meiner aktuellen Bachelorarbeit angefügt:

-> Laden der Daten

-> Bereinigen der Daten

-> Formatieren der Daten in ein Tensorflow kompatibles Datenformat (Tensor)

-> Erstellung eines Modells, welches auf einem vortrainierten Encoder basiert (Googles BERT), 
dessen Output in einem neuronalen Netz verarbeitet und für die Klassifikation verwendet

-> Training des Modells mit einer K-Fold-Kreuzvalidierung, welche verwendet wird um besser
abschätzen zu können, wie das Modell auf nie gesehene Daten reagiert

-> Visualisierung der Ergebnisse (Vergleich der Labels (Predicted True))

Die Visualisierungen wurden von mir erstellt und zeigen:

- Boxplot_for_accuracy.png: Den Vergleich der Genauigkeit einiger Modelle auf unbekannten Validierungsdaten

- Confussion_Matrix.png: auf der vertikalen Achse die True Labels und auf der horizontalen die Ausgaben des Modells

- Explainable_AI_evaluation.png: Eine Evaluation der Auswertung eines bestimmten Satzes. Die Balken zeigen den
Einfluss der einzelnen Wörter auf das Ergebnis des Modells. Errechnet werden die Werte durch Methoden der
Spieltheorie
