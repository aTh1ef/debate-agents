# ⚖️ AI Claim Verification System

A Streamlit-based application that uses local LLMs and LangGraph orchestration to verify claims by scraping evidence from the web, simulating a multi-round debate, and producing a final structured judgment.

## 📋 Features

* Structured AI debate with Verifier, Counter-Explainer, and Judge agents
* Web scraping with error handling and content extraction
* Final JSON-based verdict with confidence and evidence analysis
* LM Studio for local, private LLM inference
* Fully state-driven flow using LangGraph

## 🔧 Technology Stack

* **Frontend & Application**: [Streamlit](https://streamlit.io/) – For interactive UI
* **Language Models**: [LM Studio](https://lmstudio.ai/) – Runs local models via OpenAI-compatible API
* **Debate & Graph Framework**: [LangGraph](https://github.com/langchain-ai/langgraph) – For stateful, multi-agent orchestration
* **Web Scraping**: `requests`, `BeautifulSoup` – For extracting content from URLs
* **Message Handling**: `langchain_core`, `openai`, `dataclasses`, `pydantic` – Structured message and state management

## 🔄 How It Works

1. **User Input**

   * Enter a claim and a list of URLs.
   * Choose how many rounds the AI agents will debate.

2. **Web Scraping**

   * Each URL is scraped using `requests` and parsed with `BeautifulSoup`.
   * Content and metadata (title, content, full text) are extracted.
   * Failed or invalid sources are logged with retry support.

3. **LangGraph Workflow Begins**

   * The system constructs a LangGraph with these nodes:

     * `scrape_evidence`
     * `verifier_turn`
     * `counter_explainer_turn`
     * `check_rounds`
     * `judge_decision`

4. **Verifier Turn**

   * LM Studio generates an argument supporting the claim using the Qwen model.
   * It references extracted evidence.

5. **Counter-Explainer Turn**

   * Another response is generated (same or different model) to present counterpoints.

6. **Rounds Loop**

   * The debate continues for the defined number of rounds.

7. **Judge Decision**

   * Judge agent (using a reasoning-focused model like Phi-4-Mini) evaluates the full debate.
   * Produces a 5-part analysis and structured JSON verdict.

8. **Scoring & Display**

   * The judge analysis is transformed into JSON verdict.
   * Output includes:

     * Final verdict (TRUE / FALSE / INSUFFICIENT\_EVIDENCE)
     * Confidence score
     * Evidence quality
     * Winning side
     * Key reasoning

## 🧪 Example Output

```json
{
  "verdict": "TRUE",
  "confidence": 0.85,
  "evidence_quality": "STRONG",
  "winning_side": "verifier",
  "reasoning": "The verifier presented stronger evidence and rebutted counterpoints."
}
```

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* [LM Studio](https://lmstudio.ai/) installed and running
* Models downloaded:

  * `Qwen/Qwen1.5-1.7B-Chat`
  * `microsoft/phi-2` or `phi-4-mini`

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/claim-verifier.git
cd claim-verifier
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit langgraph langchain-core openai beautifulsoup4 requests pandas
```

### 4. Run LM Studio

1. Launch LM Studio from [https://lmstudio.ai](https://lmstudio.ai)
2. Download and start the models listed above
3. Confirm it’s listening at: `http://localhost:1234/v1`

### 5. Run the Streamlit App

```bash
streamlit run main.py
```

Access the app at `http://localhost:8501`

## 📖 Usage Guide

1. Enter your claim (e.g. "Video games improve cognitive function")
2. Add relevant URLs (studies, articles, etc.)
3. Set number of debate rounds
4. Click "Start Verification"
5. View:

   * Arguments from each side
   * Final structured verdict
   * Judge analysis and scoring

## 🧰 Troubleshooting

* **LM Studio not running**:

  * Ensure it's open and models are loaded
  * Check port `1234` is active

* **Scraping issues**:

  * Some sites block bots — try using more accessible URLs

* **Model issues**:

  * Restart LM Studio and re-load the models


