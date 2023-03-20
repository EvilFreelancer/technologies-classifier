from dataset import ITTechDataset
from model import ITTechModel

# Create a new dataset object and load data
dataset = ITTechDataset("datasets/output_2022-11-11_03:24.csv")

# Create and train model
model = ITTechModel(dataset=dataset)
model.fit_transform()
model.train()

# use the model to make predictions on new data
text_to_predict = "Text about ReactJS, VueJS and other JavaScript related things"
predicted_labels = model.predict(text_to_predict)

print(predicted_labels)
