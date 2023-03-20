# Technologies classifier

Simple neural network model based on Sklean framework for detecting technologies in provided text,
on input it will receive any text, in output you will receive array of mentioned technologies.

## How to use

```python
from dataset import ITTechDataset
from model import ITTechModel

# Create a new dataset object and load data
dataset = ITTechDataset(file_path="datasets/texts_with_labels.csv")

# Create and train model
model = ITTechModel(dataset=dataset)
model.fit_transform()
model.train()

# use the model to make predictions on new data
text_to_predict = "Text about ReactJS, VueJS and other JavaScript related things"
predicted_labels = model.predict(text_to_predict)

print(predicted_labels) # >>> ['javascript']
```

## How to save trained model

```python
from dataset import ITTechDataset
from model import ITTechModel

# Create a new dataset object and load data
dataset = ITTechDataset(file_path="datasets/texts_with_labels.csv")

# Train and save model to disk
model = ITTechModel(dataset=dataset, model_path="./model/tech_model.joblib")
model.fit_transform()
model.train()
model.save_model()
```

## How to load saved model

```python
from model import ITTechModel

# Load pretrained model
model = ITTechModel(model_path="./model/tech_model.joblib")
model.load_model()
```

## Dataset example

```csv
text,keys
"Text about JavaScript",javascript
"Text about Django, JavaScript and PHP","javascript,python,php"
...
...
```
