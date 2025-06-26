from app.model import predict
from app.log import log

def test_positive_predict():

  text = "I love this project!!!"
  target = "positive"

  label, scores = predict(text)

  assert label == target
  assert scores[0] < 0.5
  assert scores[1] < 0.5
  assert scores[2] > 0.5

def test_negative_predict():

  text = "I hate this project!!!"
  target = "negative"

  label, scores = predict(text)

  assert label == target
  assert scores[0] > 0.5
  assert scores[1] < 0.5
  assert scores[2] < 0.5

def test_neutral_predict():

  text = "I'm working on this project"
  target = "neutral"

  label, scores = predict(text)

  assert label == target
  assert scores[0] < 0.5
  assert scores[1] > 0.5
  assert scores[2] < 0.5