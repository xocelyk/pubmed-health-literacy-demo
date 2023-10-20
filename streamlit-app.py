import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

test_text = st.text_input("Enter your text here", "Patient has dementia")


checkpoint = '.'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

inputs = tokenizer(test_text, return_tensors='pt', truncation=True, padding=True, max_length=512)


# 2. Pass the processed input through the model
with torch.no_grad():
    logits = model(**inputs).logits

# 3. Get the predicted label
# turn logits into probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)
st.write(f"Probability Poor Historian: {probs[0][1].item()}")
