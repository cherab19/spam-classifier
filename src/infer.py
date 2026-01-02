import joblib
import os

MODEL_PATH = os.path.join("models", "spam_nb_pipeline.joblib")

def predict(messages):
    pipeline = joblib.load(MODEL_PATH)
    return pipeline.predict(messages)

if __name__ == "__main__":
    samples = [
        "Congratulations! You have won a free lottery ticket. Click here to claim.",
        "Hey, can we reschedule our meeting to 10am tomorrow?"
    ]
    preds = predict(samples)
    for msg, label in zip(samples, preds):
        print(f"[{label}] {msg}")
