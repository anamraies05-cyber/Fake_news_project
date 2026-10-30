# Fake News Detection using BERT

A text classifier that reads a news statement and predicts whether it looks **real** or **fake**. It is the text-analysis part of my final year project on fake news and deepfake detection.

## How it works
1. Start with BERT (`bert-base-uncased`), a model that already understands English.
2. Fine-tune it on the LIAR dataset (about 4,300 labeled political statements).
3. Group the 6 original labels into two: fake (false, pants-fire, barely-true) and real (half-true, mostly-true, true).
4. Use the trained model in a small Streamlit web app.

## Results (test set, 1,241 statements)
| Metric | Score |
|---|---|
| Accuracy | 61.6% |
| F1 score | 0.50 |

This is a first version. LIAR is a hard dataset with short statements, and the model still tends to say "real" too often.

## Limitations
- The model learns the *style* of fake claims. It does not check facts, so it can be wrong.
- It was trained on English political statements only.

## Run it yourself
1. Install Python 3.12 and Git.
2. Download this project:
   git clone https://github.com/anamraies05-cyber/Fake_news_project
   cd Fake_news_project
3. Install the tools:
   pip install -r requirements.txt
4. Download the trained model from the link below, and place the folder `fake_news_bert` inside the project folder.
   Model link: ([(https://drive.google.com/drive/folders/1nb93coY0uTpGpThcfMk3mjCp0KjnDN10?usp=drive_link))
5. Start the app:
   python -m streamlit run app.py

## Files
- `bert_fake_news.ipynb`: training notebook (run in Google Colab)
- `app.py`: the Streamlit web app
- `requirements.txt`: tools needed

## Next steps
- Improve accuracy with a larger dataset and better training settings
- Add web fact-checking
- Add deepfake image and video detection
