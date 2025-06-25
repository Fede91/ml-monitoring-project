import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

labels = ['negative', 'neutral', 'positive']

tokenizer = AutoTokenizer.from_pretrained("0xfedev/corporate-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("0xfedev/corporate-sentiment")

model.eval()

def predict(text, target):
  encoded_input = tokenizer(text, return_tensors='pt')

  with torch.no_grad():
    output = model(**encoded_input)
    scores = F.softmax(output.logits,
                       dim=1)
    predicted_class = torch.argmax(scores,
                                   dim=1).item()
    
    predicted_label = labels[predicted_class]

    return labels[predicted_class], scores[0].tolist()