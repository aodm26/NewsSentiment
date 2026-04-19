# NewsSentiment

A lightweight Streamlit app for live RSS headline sentiment monitoring.

It fetches headlines from public RSS feeds, scores them using a fast rule-based economic/news sentiment classifier, and shows live dashboard metrics, confidence labels, and downloadable CSV results.

## Features

- Live RSS feed monitoring
- Fast headline sentiment classification
- Confidence scoring
- Positive / Negative / Neutral counts
- CSV export of current snapshot
- Optional auto refresh
- Streamlit-friendly deployment

## Project structure

```text
NewsSentiment/
├── app/
│   └── rss_sentiment_dashboard.py
├── tests/
│   └── test_classifier.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/rss_sentiment_dashboard.py
```

## Deploy from GitHub

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select your GitHub repo and branch.
5. Set the main file path to:

```text
app/rss_sentiment_dashboard.py
```

6. Click **Deploy**.

## Notes

This project uses a fast heuristic classifier instead of a large language model for live refresh performance. That makes it much faster and cheaper to run, but it can still misclassify ambiguous headlines.

## Suggested next improvements

- Add manual review mode for low-confidence headlines
- Add benchmark dataset and evaluation metrics
- Add hybrid LLM fallback for uncertain cases
- Add screenshot/GIF to this README
