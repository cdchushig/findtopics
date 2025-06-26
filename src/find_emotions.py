from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

tweet = "Hoy me siento con muchas ganas de llorar y no salir de casa."

resultados = emotion_classifier(tweet)

for emocion in resultados[0]:
    print(f"{emocion['label']}: {emocion['score']:.4f}")
