import pandas as pd
from dataset import ITTechDataset
import joblib

# create a new dataset object
dataset = ITTechDataset()

# train the model
dataset.train()

# save the trained model to disk
joblib.dump(dataset.clf, "model.joblib")

# add new data to the dataset
new_data = pd.DataFrame({
    "text": ["This is a new example text", "Another example text"],
    "labels": ["label1,label2", "label2"]
})
dataset.add_data(new_data)

# retrain the model with the new data
dataset.train()

# save the retrained model to disk
joblib.dump(dataset.clf, "model_retrained.joblib")

# load the saved model from disk
loaded_model = joblib.load("model_retrained.joblib")

# use the model to make predictions on new data
text_to_predict = "This is a new text to predict labels for"
predicted_labels = loaded_model.predict(text_to_predict)

print(predicted_labels)
