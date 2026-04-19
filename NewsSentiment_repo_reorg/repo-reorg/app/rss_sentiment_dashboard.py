import re
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import feedparser
import pandas as pd
import streamlit as st

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(page_title="Live RSS Sentiment Dashboard", page_icon="📰", layout="wide")

RSS_FEEDS = {
    "Google News Business (IE)": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IE&gl=IE&ceid=IE:en",
    "Google News World (IE)": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IE&gl=IE&ceid=IE:en",
    "Google News Technology (IE)": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IE&gl=IE&ceid=IE:en",
    "Google News Top Stories (IE)": "https://news.google.com/rss?hl=en-IE&gl=IE&ceid=IE:en",
    "BBC Top Stories": "http://feeds.bbci.co.uk/news/rss.xml",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
}

SOURCE_CLEAN = [
    r"\s*[-|–|—]\s*RTE\.ie\s*$",
    r"\s*[-|–|—]\s*The Irish Times\s*$",
    r"\s*[-|–|—]\s*The Irish Independent\s*$",
    r"\s*[-|–|—]\s*BBC News\s*$",
    r"\s*[-|–|—]\s*Reuters\s*$",
    r"\s*[-|–|—]\s*Bloomberg\s*$",
    r"\s*[-|–|—]\s*CNBC\s*$",
]

POSITIVE_PATTERNS = [
    ("beats estimates", 5, "beats estimates"),
    ("tops estimates", 5, "tops estimates"),
    ("profit rises", 4, "profit growth"),
    ("profits rise", 4, "profit growth"),
    ("profit jumps", 5, "profit jump"),
    ("higher revenue", 4, "revenue growth"),
    ("revenue rises", 4, "revenue growth"),
    ("sales rise", 3, "sales growth"),
    ("raises forecast", 4, "better outlook"),
    ("upgrades forecast", 4, "better outlook"),
    ("strong demand", 3, "strong demand"),
    ("growth", 2, "growth signal"),
    ("expansion", 3, "business expansion"),
    ("expand", 2, "business expansion"),
    ("investment", 3, "investment support"),
    ("invests", 3, "investment support"),
    ("new jobs", 4, "job creation"),
    ("job growth", 4, "job creation"),
    ("hiring", 3, "hiring growth"),
    ("record high", 3, "strong performance"),
    ("rate cut", 2, "easier financial conditions"),
    ("cuts rates", 2, "easier financial conditions"),
    ("discount", 2, "lower consumer cost"),
    ("affordable", 2, "better affordability"),
    ("support scheme", 2, "policy support"),
    ("raises savings rate", 4, "higher saver returns"),
    ("higher savings rate", 4, "higher saver returns"),
    ("savings account", 2, "saver benefit"),
    ("inflation eases", 4, "inflation easing"),
    ("inflation cools", 4, "inflation easing"),
]

NEGATIVE_PATTERNS = [
    ("misses estimates", 5, "misses estimates"),
    ("below estimates", 4, "below expectations"),
    ("profit falls", 4, "profit decline"),
    ("profits fall", 4, "profit decline"),
    ("profit warning", 5, "profit warning"),
    ("revenue falls", 4, "revenue decline"),
    ("sales fall", 3, "sales decline"),
    ("warning", 3, "warning signal"),
    ("warns", 3, "warning signal"),
    ("loss", 3, "loss reported"),
    ("losses", 3, "loss reported"),
    ("decline", 2, "decline signal"),
    ("slump", 4, "slump signal"),
    ("drops", 2, "drop signal"),
    ("cuts jobs", 5, "job cuts"),
    ("job cuts", 5, "job cuts"),
    ("layoffs", 5, "layoffs"),
    ("fine", 3, "regulatory penalty"),
    ("lawsuit", 3, "legal risk"),
    ("probe", 3, "regulatory probe"),
    ("downgrade", 3, "downgrade"),
    ("crisis", 4, "crisis risk"),
    ("disruption", 4, "economic disruption"),
    ("supply shock", 4, "supply shock"),
    ("shortage", 4, "shortage risk"),
    ("war", 4, "geopolitical risk"),
    ("risk", 2, "risk signal"),
    ("inflation rises", 4, "higher inflation"),
    ("prices rose", 3, "higher prices"),
    ("house prices rose", 4, "housing less affordable"),
    ("home prices rose", 4, "housing less affordable"),
    ("rent rises", 4, "higher rents"),
    ("fuel left", 4, "fuel shortage"),
    ("jet fuel left", 5, "fuel shortage"),
    ("bankruptcy", 5, "bankruptcy risk"),
    ("default", 4, "default risk"),
]

NEUTRAL_PATTERNS = [
    ("on the market", 1, "asset sale update"),
    ("for sale", 1, "sale listing"),
    ("live updates", 2, "rolling update"),
    ("latest on", 2, "rolling update"),
    ("meeting", 1, "scheduled event"),
    ("statement", 1, "official statement"),
    ("report", 1, "report mention"),
    ("appoints", 1, "executive appointment"),
    ("names", 1, "naming update"),
    ("talks", 1, "talks ongoing"),
    ("said", 0, "reported statement"),
]

INTENSIFIERS = {
    "sharply": 1,
    "surges": 2,
    "soars": 2,
    "jumps": 2,
    "plunges": 2,
    "tumbles": 2,
    "slumps": 2,
    "strong": 1,
    "weak": 1,
}
NEGATORS = {"not", "no", "without"}
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_LABEL_MAP = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}


@st.cache_resource(show_spinner=False)
def load_hf_model():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    return tokenizer, model


@st.cache_data(ttl=90, show_spinner=False)
def fetch_rss(feed_name: str, max_stories: int) -> pd.DataFrame:
    feed = feedparser.parse(RSS_FEEDS[feed_name])
    rows = []
    for entry in feed.entries[:max_stories]:
        title = str(getattr(entry, "title", "") or "").strip()
        if not title:
            continue
        link = str(getattr(entry, "link", "") or "").strip()
        published = str(getattr(entry, "published", "") or "").strip()
        source_name = feed_name
        source_obj = getattr(entry, "source", None)
        if source_obj:
            try:
                source_name = source_obj.get("title", feed_name) or feed_name
            except Exception:
                pass
        rows.append({"source": source_name, "headline": title, "published": published, "url": link})
    return pd.DataFrame(rows)


def normalize_headline(text: str) -> str:
    out = str(text or "").strip()
    for pat in SOURCE_CLEAN:
        out = re.sub(pat, "", out, flags=re.I)
    out = out.replace("’", "'")
    out = out.replace("–", "-").replace("—", "-")
    out = re.sub(r"\s+", " ", out)
    return out.strip(" .:-\\n\\t\"'")


def token_list(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())


def token_set(text: str):
    return set(token_list(text))


def apply_patterns(text: str, patterns, label_name: str):
    score = 0
    reasons = []
    for phrase, weight, reason in patterns:
        if phrase in text:
            score += weight
            reasons.append((label_name, reason, weight))
    return score, reasons


def entity_rules(text: str, tokens: set):
    pos = neg = neu = 0
    reasons = []

    if ("profit" in tokens or "revenue" in tokens or "sales" in tokens or "earnings" in tokens) and any(
        w in tokens for w in {"up", "rise", "rises", "boost", "higher", "jump", "jumps"}
    ):
        pos += 4
        reasons.append(("Positive", "improving business results", 4))

    if ("profit" in tokens or "revenue" in tokens or "sales" in tokens or "earnings" in tokens) and any(
        w in tokens for w in {"down", "fall", "falls", "lower", "drop", "drops"}
    ):
        neg += 4
        reasons.append(("Negative", "weakening business results", 4))

    if any(w in text for w in ["interest rates rise", "rate hike", "rates higher", "higher rates"]):
        neg += 2
        reasons.append(("Negative", "tighter financial conditions", 2))

    if any(w in text for w in ["rate cut", "cuts rates", "rates cut"]):
        pos += 2
        reasons.append(("Positive", "easier financial conditions", 2))

    if ("inflation" in tokens or "prices" in tokens or "rents" in tokens) and any(
        w in tokens for w in {"up", "rise", "rises", "higher", "jump", "jumps"}
    ):
        neg += 3
        reasons.append(("Negative", "cost pressure rising", 3))

    if ("inflation" in tokens or "prices" in tokens) and any(
        w in tokens for w in {"down", "falls", "eases", "cools", "lower"}
    ):
        pos += 3
        reasons.append(("Positive", "cost pressure easing", 3))

    if ("jobs" in tokens or "hiring" in tokens or "employment" in tokens) and any(
        w in tokens for w in {"up", "rise", "strong", "growth"}
    ):
        pos += 3
        reasons.append(("Positive", "labour market strength", 3))

    if ("jobs" in tokens or "employment" in tokens or "hiring" in tokens) and any(
        w in tokens for w in {"cuts", "losses", "weak", "fall", "slump"}
    ):
        neg += 4
        reasons.append(("Negative", "labour market weakness", 4))

    if any(w in text for w in ["acquires", "merger", "deal"]):
        neu += 1
        reasons.append(("Neutral", "transaction headline", 1))

    return pos, neg, neu, reasons


def resolve_negation(text: str, pos: int, neg: int):
    words = token_list(text)
    for i, word in enumerate(words[:-1]):
        if word in NEGATORS:
            nxt = words[i + 1]
            if nxt in {"growth", "profit", "gain", "recovery", "improvement"}:
                neg += 2
            if nxt in {"loss", "layoffs", "decline", "inflation", "shortage"}:
                pos += 1
    return pos, neg


def calibrate_neutral(pos: int, neg: int, neu: int):
    directional_peak = max(pos, neg)
    if directional_peak >= 3:
        neu = max(0, neu - 1)
    if directional_peak >= 5:
        neu = max(0, neu - 2)
    return neu


def choose_label(pos: int, neg: int, neu: int):
    directional_peak = max(pos, neg)
    directional_gap = abs(pos - neg)

    if directional_peak == 0 and neu <= 1:
        return "Neutral", "Low"

    if pos >= neg + 2 and pos >= max(2, neu + 1):
        margin = pos - max(neg, neu)
        return "Positive", ("High" if margin >= 4 else "Medium")

    if neg >= pos + 2 and neg >= max(2, neu + 1):
        margin = neg - max(pos, neu)
        return "Negative", ("High" if margin >= 4 else "Medium")

    if directional_peak >= 4 and directional_gap >= 1:
        if pos > neg:
            return "Positive", "Low"
        if neg > pos:
            return "Negative", "Low"

    if neu >= directional_peak + 2:
        return "Neutral", ("Medium" if neu >= 3 else "Low")

    if pos > neg:
        return "Positive", "Low"
    if neg > pos:
        return "Negative", "Low"
    return "Neutral", "Low"


def classify_rules(headline: str):
    h = normalize_headline(headline)
    low = h.lower()
    tokens = token_set(h)

    pos, pos_reasons = apply_patterns(low, POSITIVE_PATTERNS, "Positive")
    neg, neg_reasons = apply_patterns(low, NEGATIVE_PATTERNS, "Negative")
    neu, neu_reasons = apply_patterns(low, NEUTRAL_PATTERNS, "Neutral")

    ep, en, eu, extra = entity_rules(low, tokens)
    pos += ep
    neg += en
    neu += eu
    pos, neg = resolve_negation(low, pos, neg)

    for word, boost in INTENSIFIERS.items():
        if word in tokens:
            if neg > pos:
                neg += boost
            elif pos > neg:
                pos += boost

    if "discount" in low and "tenant" in low:
        pos += 4
        extra.append(("Positive", "lower housing costs", 4))
    if ("home prices rose" in low or "house prices rose" in low) and "%" in low:
        neg += 3
        extra.append(("Negative", "affordability worsening", 3))
    if "savings account" in low and "raises rate" in low:
        pos += 4
        extra.append(("Positive", "better saver returns", 4))
    if "jet fuel left" in low or ("six weeks" in low and "fuel" in low):
        neg += 5
        extra.append(("Negative", "fuel shortage risk", 5))
    if "on the market" in low and any(w in low for w in ["resort", "hotel", "property"]):
        neu += 1
        extra.append(("Neutral", "asset sale with unclear impact", 1))

    neu = calibrate_neutral(pos, neg, neu)
    label, confidence = choose_label(pos, neg, neu)

    reasons = pos_reasons + neg_reasons + neu_reasons + extra
    label_reasons = [r for lab, r, w in sorted(reasons, key=lambda x: x[2], reverse=True) if lab == label]

    if label_reasons:
        reason = label_reasons[0]
    elif label == "Positive":
        reason = "headline points to improvement"
    elif label == "Negative":
        reason = "headline points to deterioration"
    else:
        reason = "headline remains mainly informational"

    return {
        "predicted": label,
        "reasoning": reason,
        "confidence": confidence,
        "score_positive": pos,
        "score_negative": neg,
        "score_neutral": neu,
        "method": "Rules",
    }


def classify_hf(headline: str):
    tokenizer, model = load_hf_model()
    if tokenizer is None or model is None:
        return None
    text = normalize_headline(headline)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    idx = int(torch.argmax(probs).item())
    raw_label = model.config.id2label[idx]
    label = HF_LABEL_MAP.get(raw_label, raw_label.title())
    confidence_value = float(probs[idx].item())
    confidence = "High" if confidence_value >= 0.75 else "Medium" if confidence_value >= 0.55 else "Low"
    return {
        "predicted": label,
        "reasoning": f"transformer confidence {confidence_value:.2f}",
        "confidence": confidence,
        "method": "LLM",
    }


def classify_headline(headline: str, use_llm=False, llm_on_low_conf=True):
    rule_result = classify_rules(headline)
    if use_llm and TRANSFORMERS_AVAILABLE:
        should_call = llm_on_low_conf and rule_result["confidence"] == "Low"
        if should_call:
            hf_result = classify_hf(headline)
            if hf_result is not None:
                merged = rule_result.copy()
                merged.update(hf_result)
                return merged
    return rule_result


def format_time(value: str) -> str:
    try:
        return parsedate_to_datetime(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return value


def analyze(df: pd.DataFrame, use_llm=False, llm_on_low_conf=True) -> pd.DataFrame:
    if df.empty:
        return df
    outputs = [classify_headline(h, use_llm=use_llm, llm_on_low_conf=llm_on_low_conf) for h in df["headline"].tolist()]
    out = df.copy()
    out["predicted"] = [x["predicted"] for x in outputs]
    out["reasoning"] = [x["reasoning"] for x in outputs]
    out["confidence"] = [x["confidence"] for x in outputs]
    out["method"] = [x.get("method", "Rules") for x in outputs]
    out["score_positive"] = [x.get("score_positive", None) for x in outputs]
    out["score_negative"] = [x.get("score_negative", None) for x in outputs]
    out["score_neutral"] = [x.get("score_neutral", None) for x in outputs]
    out["published"] = out["published"].map(format_time)
    out["domain"] = out["url"].map(lambda u: urlparse(u).netloc.replace("www.", "") if u else "")
    return out


def sentiment_emoji(label: str) -> str:
    return {"Positive": "🟢 Positive", "Negative": "🔴 Negative", "Neutral": "⚪ Neutral"}.get(label, "⚪ Neutral")


st.title("📰 Live RSS headline sentiment")
st.caption("Hybrid version: fast rules by default, lightweight transformer fallback for low-confidence headlines")

with st.sidebar:
    st.header("Controls")
    feed_name = st.selectbox("Feed", list(RSS_FEEDS.keys()), index=0)
    max_stories = st.slider("Stories", 10, 100, 30, 5)
    auto_refresh = st.checkbox("Auto refresh every 2 min", value=False)
    show_scores = st.checkbox("Show scoring columns", value=False)
    use_llm = st.checkbox("Enable lightweight LLM fallback", value=False, disabled=not TRANSFORMERS_AVAILABLE)
    llm_on_low_conf = st.checkbox("Only use LLM on low-confidence rules", value=True, disabled=not TRANSFORMERS_AVAILABLE)
    refresh = st.button("Refresh now", use_container_width=True)

if auto_refresh:
    st.markdown('<meta http-equiv="refresh" content="120">', unsafe_allow_html=True)

start = time.time()
if refresh:
    fetch_rss.clear()

df = analyze(fetch_rss(feed_name, max_stories), use_llm=use_llm, llm_on_low_conf=llm_on_low_conf)
elapsed = time.time() - start

if df.empty:
    st.warning("No stories returned from this feed.")
    st.stop()

counts = df["predicted"].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stories", int(len(df)))
col2.metric("Positive", int(counts["Positive"]))
col3.metric("Negative", int(counts["Negative"]))
col4.metric("Refresh time", f"{elapsed:.2f}s")

chart_df = pd.DataFrame({"sentiment": counts.index, "count": counts.values})
st.bar_chart(chart_df.set_index("sentiment"))

view_cols = ["source", "headline", "predicted", "confidence", "method", "reasoning", "published", "domain", "url"]
if show_scores:
    view_cols += ["score_positive", "score_negative", "score_neutral"]
show = df[view_cols].copy()
show["predicted"] = show["predicted"].map(sentiment_emoji)
st.dataframe(show, use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download current results CSV", csv, file_name="rss_sentiment_snapshot.csv", mime="text/csv")
