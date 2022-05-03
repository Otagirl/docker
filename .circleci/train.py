from json import loads, dumps
from sklearn.linear_model import LinearRegression

with open("input.data.json", "w") as f:
    content = f.read()
    TRAIN_INPUT = loads(content)


with open("output.data.json", "w") as f:
    content = f.read()
    TRAIN_OUTPUT = loads(content)

predictor = LinearRegression(n_jobs=-1)

predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

with open('model.json', ) as f: 
    f.write(dumps(predictor.coef_)) 