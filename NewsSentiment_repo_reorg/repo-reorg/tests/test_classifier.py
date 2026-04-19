import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rss_sentiment_dashboard",
    Path(__file__).resolve().parents[1] / "app" / "rss_sentiment_dashboard.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_positive_headline():
    label, reason, confidence, *_ = module.classify_headline("Company profit rises as revenue beats estimates")
    assert label == "Positive"


def test_negative_headline():
    label, reason, confidence, *_ = module.classify_headline("Firm announces layoffs after profit warning")
    assert label == "Negative"


def test_neutral_headline():
    label, reason, confidence, *_ = module.classify_headline("Government statement after meeting on transport policy")
    assert label == "Neutral"
