import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Fake News Detector", page_icon="📰")


@st.cache_resource
def load_model():
        tokenizer = AutoTokenizer.from_pretrained("anamcodes/fake-news-bert")
    model = AutoModelForSequenceClassification.from_pretrained("anamcodes/fake-news-bert")
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

st.title("📰 Fake News Detector")
st.write("Type or paste a news statement. The BERT model will guess if it looks real or fake.")

text = st.text_area("News statement", height=150)

if st.button("Check"):
    if not text.strip():
        st.warning("Please type a statement first.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)[0]
        real = float(probs[0]) * 100
        fake = float(probs[1]) * 100

        if fake > real:
            st.error(f"Looks FAKE ({fake:.1f}% sure)")
        else:
            st.success(f"Looks REAL ({real:.1f}% sure)")

        st.progress(int(fake), text=f"Fake: {fake:.1f}%")
        st.progress(int(real), text=f"Real: {real:.1f}%")

st.caption("Note: this model learns the style of fake claims. It does not check facts, so it can be wrong.")