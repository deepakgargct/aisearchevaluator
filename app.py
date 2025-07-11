import streamlit as st
from transformers import pipeline
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---- CONFIG ----
st.set_page_config(page_title="AI Search Optimization Evaluator", layout="wide")

# ---- Load Local Model ----
st.cache_resource
def load_model():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")

generator = load_model()

# ---- SCORING METRICS ----
principles = [
    {
        "title": "Chunk-Level Retrieval",
        "description": "Is each section self-contained and focused on one topic?",
        "prompt": "Analyze if the content contains semantically complete chunks that are self-contained and independently understandable. Does each section focus on a single idea? Rate 0-10 and explain."
    },
    {
        "title": "Answer Synthesis Optimization",
        "description": "Is the answer easy to extract and fit into multi-source AI answers?",
        "prompt": "Evaluate if the content is structured with a summary followed by elaboration, plain tone, and Q&A format. Rate 0-10 and explain."
    },
    {
        "title": "Citation-Worthiness",
        "description": "Does the content look trustworthy and fact-based?",
        "prompt": "Check if content includes citations, author credentials, accurate claims, and timestamps. Rate 0-10 and explain."
    },
    {
        "title": "Topical Breadth & Depth",
        "description": "Does the site use hub-cluster model to fully cover a topic?",
        "prompt": "Assess if the content follows a pillar-and-cluster approach, linking to detailed subtopics. Rate 0-10 and explain."
    },
    {
        "title": "Multi-Modal Support",
        "description": "Are visuals and tables machine-readable and helpful?",
        "prompt": "Evaluate if the content uses HTML tables, descriptive alt text, <figure> markup, and captions for images. Rate 0-10 and explain."
    },
    {
        "title": "Authoritativeness Signals",
        "description": "Is there clear EEAT (expertise, authority, trust) in the content?",
        "prompt": "Check for expert bylines, original data, external mentions, structured metadata, and citations. Rate 0-10 and explain."
    },
    {
        "title": "Personalization Resilience",
        "description": "Does the content serve diverse intents, personas, or regions?",
        "prompt": "Analyze if content targets multiple personas, intents, and locales with segmented sections. Rate 0-10 and explain."
    },
    {
        "title": "AI Crawlability & Indexability",
        "description": "Is the content accessible and indexable by AI bots?",
        "prompt": "Evaluate if content is server-rendered, allows AI bots via robots.txt, and avoids noindex/nosnippet. Rate 0-10 and explain."
    }
]

# ---- Scrape URL ----
def scrape_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract visible text only
        for script in soup(["script", "style"]):
            script.decompose()
        return "\n".join([t.strip() for t in soup.stripped_strings])
    except Exception as e:
        return f"Error scraping URL: {e}"

# ---- UI ----
st.title("🤖 AI Search Optimization Evaluator")
st.markdown("Evaluate your content's readiness for AI Overviews, ChatGPT Answers, Perplexity, and more — without any API key.")

content_mode = st.radio("Choose input type:", ["Paste Text", "Scrape from URL"])

if content_mode == "Paste Text":
    input_content = st.text_area("Paste your page content or HTML here:", height=300)
else:
    url = st.text_input("Enter a URL to scrape:")
    input_content = scrape_url(url) if url else ""
    if url:
        st.text_area("Scraped Content:", value=input_content, height=300)

evaluate = st.button("Run Local Evaluation")

# ---- PROCESSING ----
def evaluate_content(content):
    results = []
    for p in principles:
        full_prompt = f"Content:\n{content}\n\nInstruction:\n{p['prompt']}"
        reply = generator(full_prompt, max_new_tokens=256)[0]['generated_text']
        score_line = next((line for line in reply.splitlines() if any(str(n) in line for n in range(0, 11))), "Score: 0")
        score = int(''.join(filter(str.isdigit, score_line))) if any(char.isdigit() for char in score_line) else 0
        results.append({
            "Principle": p['title'],
            "Score": score,
            "Explanation": reply.strip()
        })
    return results

# ---- OUTPUT ----
if evaluate and input_content:
    with st.spinner("Evaluating content using local model..."):
        scores = evaluate_content(input_content)
        df = pd.DataFrame(scores)

        st.subheader("📊 Evaluation Summary Table")
        st.dataframe(df[['Principle', 'Score']].set_index('Principle'), use_container_width=True)

        st.subheader("📝 Detailed Feedback")
        for row in scores:
            st.markdown(f"### {row['Principle']}")
            st.markdown(f"**Score:** {row['Score']}/10")
            st.markdown(row['Explanation'])
            st.markdown("---")

        avg_score = sum(r['Score'] for r in scores) / len(scores)
        st.markdown(f"### ✅ Overall Content Score: **{avg_score:.1f}/10**")
