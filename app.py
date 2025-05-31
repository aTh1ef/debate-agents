import streamlit as st
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import time
from datetime import datetime
import logging
from urllib.parse import urljoin, urlparse
import re
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import openai
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Optimized for lower-end hardware
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
# Using smaller, more efficient models
PHI_MODEL = "microsoft/phi-4-mini-reasoning"  # Correct model ID
GEMMA_MODEL = "google/gemma-3-4b"  # Correct model ID
# Alternative lightweight models
FALLBACK_MODEL = "microsoft/DialoGPT-small"  # Backup option

# Memory optimization settings
MAX_CONTENT_LENGTH = 2000  # Reduced from 5000
MAX_FULL_TEXT_LENGTH = 3000  # Reduced from 10000
MAX_RESPONSE_TOKENS = 500  # Reduced from 800-1000

class AgentRole(Enum):
    VERIFIER = "verifier"
    OPPOSER = "opposer"
    JUDGE = "judge"

@dataclass
class DebateState:
    claim: str
    urls: List[str]
    scraped_content: List[Dict[str, Any]]
    verifier_arguments: List[str]
    opposer_arguments: List[str]
    debate_rounds: int
    current_round: int
    final_judgment: Optional[Dict[str, Any]]
    debate_history: List[Dict[str, Any]]

class WebScraper:
    """Optimized web scraper with error handling and rate limiting"""
    
    def __init__(self):
        self.session = requests.Session()
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a single URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text() if title else "No title found"
            
            # Extract main content (try to find article or main content areas)
            main_content = ""
            for selector in ['article', 'main', '.content', '#content', '.post-content']:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem.get_text(strip=True)
                    break
            
            if not main_content:
                main_content = text
            
            return {
                'url': url,
                'title': title_text,
                'content': main_content[:MAX_CONTENT_LENGTH],  # Reduced memory usage
                'full_text': text[:MAX_FULL_TEXT_LENGTH],  # Reduced memory usage
                'status': 'success',
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'full_text': '',
                'status': 'error',
                'error': str(e),
                'scraped_at': datetime.now().isoformat()
            }
    
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs"""
        results = []
        for url in urls:
            st.write(f"🔍 Scraping: {url}")
            result = self.scrape_url(url)
            results.append(result)
            time.sleep(1)  # Rate limiting
        return results

class LMStudioClient:
    """Client for interacting with LM Studio API"""
    
    def __init__(self, base_url: str = LM_STUDIO_BASE_URL):
        self.base_url = base_url
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="lm-studio"  # LM Studio doesn't require a real API key
        )
    
    def generate_response(self, model: str, messages: List[Dict[str, str]], 
                         temperature: float = 0.7, max_tokens: int = MAX_RESPONSE_TOKENS) -> str:
        """Generate response using specified model with memory optimization"""
        try:
            # Truncate messages if too long to save memory
            truncated_messages = []
            for msg in messages:
                content = msg['content']
                if len(content) > 3000:  # Truncate very long prompts
                    content = content[:3000] + "...[truncated for memory efficiency]"
                truncated_messages.append({
                    'role': msg['role'],
                    'content': content
                })
            
            response = self.client.chat.completions.create(
                model=model,
                messages=truncated_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response with {model}: {str(e)}")
            return f"Error: Could not generate response. Please ensure LM Studio is running and the model {model} is loaded."

class DebateAgent:
    """Base class for debate agents"""
    
    def __init__(self, role: AgentRole, model: str, client: LMStudioClient):
        self.role = role
        self.model = model
        self.client = client
        self.conversation_history = []
    
    def generate_argument(self, claim: str, evidence: List[Dict[str, Any]], 
                         opponent_arguments: List[str] = None, round_num: int = 1) -> str:
        """Generate argument based on role and evidence"""
        raise NotImplementedError

class VerifierAgent(DebateAgent):
    """Agent that argues in favor of the claim"""
    
    def generate_argument(self, claim: str, evidence: List[Dict[str, Any]], 
                         opponent_arguments: List[str] = None, round_num: int = 1) -> str:
        
        evidence_text = "\n\n".join([
            f"Source: {item['title']}\nURL: {item['url']}\nContent: {item['content']}"
            for item in evidence if item['status'] == 'success'
        ])
        
        opponent_context = ""
        if opponent_arguments:
            opponent_context = f"\n\nOpponent's previous arguments:\n{chr(10).join(opponent_arguments)}"
        
        prompt = f"""You are a skilled debater arguing that the following claim is TRUE:
CLAIM: {claim}

EVIDENCE FROM WEB SOURCES:
{evidence_text}

{opponent_context}

Round {round_num}: Present a strong, evidence-based argument supporting the claim. Use specific facts, quotes, and logical reasoning. Be persuasive but factual. Focus on:
1. Direct evidence supporting the claim
2. Credible sources and expert opinions
3. Counter-arguments to opposition points (if any)
4. Logical reasoning chain

Provide a structured argument in 1-2 paragraphs (keep concise for efficiency)."""

        messages = [
            {"role": "system", "content": "You are an expert debater and fact-checker who argues for the truth of claims using evidence and logical reasoning."},
            {"role": "user", "content": prompt}
        ]
        
        return self.client.generate_response(self.model, messages, temperature=0.7, max_tokens=800)

class OpposerAgent(DebateAgent):
    """Agent that argues against the claim"""
    
    def generate_argument(self, claim: str, evidence: List[Dict[str, Any]], 
                         opponent_arguments: List[str] = None, round_num: int = 1) -> str:
        
        evidence_text = "\n\n".join([
            f"Source: {item['title']}\nURL: {item['url']}\nContent: {item['content']}"
            for item in evidence if item['status'] == 'success'
        ])
        
        opponent_context = ""
        if opponent_arguments:
            opponent_context = f"\n\nVerifier's previous arguments:\n{chr(10).join(opponent_arguments)}"
        
        prompt = f"""You are a skilled debater arguing that the following claim is FALSE or QUESTIONABLE:
CLAIM: {claim}

EVIDENCE FROM WEB SOURCES:
{evidence_text}

{opponent_context}

Round {round_num}: Present a strong argument challenging the claim. Look for:
1. Contradictory evidence in the sources
2. Lack of sufficient evidence
3. Biased or unreliable sources
4. Logical fallacies or weak reasoning
5. Alternative explanations

Be critical but fair. Use evidence-based reasoning. Provide a structured counter-argument in 1-2 paragraphs (keep concise)."""

        messages = [
            {"role": "system", "content": "You are a critical thinker and skeptical debater who challenges claims by finding weaknesses in evidence and reasoning."},
            {"role": "user", "content": prompt}
        ]
        
        return self.client.generate_response(self.model, messages, temperature=0.7, max_tokens=800)

class JudgeAgent(DebateAgent):
    """Agent that evaluates the debate and makes final judgment"""
    
    def make_judgment(self, claim: str, evidence: List[Dict[str, Any]], 
                     verifier_arguments: List[str], opposer_arguments: List[str]) -> Dict[str, Any]:
        
        evidence_text = "\n\n".join([
            f"Source: {item['title']}\nURL: {item['url']}\nContent: {item['content'][:1000]}"
            for item in evidence if item['status'] == 'success'
        ])
        
        verifier_text = "\n\n".join([f"Verifier Argument {i+1}: {arg}" for i, arg in enumerate(verifier_arguments)])
        opposer_text = "\n\n".join([f"Opposer Argument {i+1}: {arg}" for i, arg in enumerate(opposer_arguments)])
        
        # Simplified prompt for better JSON generation
        prompt = f"""You are an impartial judge evaluating a debate about this claim: {claim}

EVIDENCE SUMMARY:
{evidence_text[:1500]}

VERIFIER'S ARGUMENTS (Supporting the claim):
{verifier_text[:1000]}

OPPOSER'S ARGUMENTS (Challenging the claim):
{opposer_text[:1000]}

Judge the claim and respond ONLY with valid JSON in this format:
{{
    "verdict": "TRUE",
    "confidence": 0.8,
    "reasoning": "Brief explanation here",
    "key_evidence": ["evidence point 1", "evidence point 2"],
    "verifier_score": 7,
    "opposer_score": 6,
    "evidence_quality": "STRONG"
}}

Choose verdict from: TRUE, FALSE, or INSUFFICIENT_EVIDENCE
Choose evidence_quality from: STRONG, MODERATE, or WEAK
Use confidence from 0.0 to 1.0
Use scores from 0 to 10"""

        messages = [
            {"role": "system", "content": "You are an impartial judge. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.generate_response(self.model, messages, temperature=0.3, max_tokens=1000)
            
            # Enhanced JSON extraction
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                judgment = json.loads(json_match.group())
                
                # Validate required fields
                required_fields = ['verdict', 'confidence', 'reasoning', 'key_evidence', 'verifier_score', 'opposer_score', 'evidence_quality']
                for field in required_fields:
                    if field not in judgment:
                        raise KeyError(f"Missing field: {field}")
                        
                # Validate verdict values
                if judgment['verdict'] not in ['TRUE', 'FALSE', 'INSUFFICIENT_EVIDENCE']:
                    judgment['verdict'] = 'INSUFFICIENT_EVIDENCE'
                    
                # Validate numeric ranges
                judgment['confidence'] = max(0.0, min(1.0, float(judgment['confidence'])))
                judgment['verifier_score'] = max(0, min(10, int(judgment['verifier_score'])))
                judgment['opposer_score'] = max(0, min(10, int(judgment['opposer_score'])))
                
                return judgment
                
        except Exception as e:
            logger.error(f"Error parsing judgment: {str(e)}")
            logger.error(f"Raw response: {response}")
        
        # Fallback judgment with detailed reasoning
        fallback_reasoning = f"Unable to generate proper judgment due to model limitations. Based on available evidence, the claim '{claim}' requires further analysis."
        
        if len(verifier_arguments) > len(opposer_arguments):
            verdict = "TRUE"
            confidence = 0.6
        elif len(opposer_arguments) > len(verifier_arguments):
            verdict = "FALSE" 
            confidence = 0.6
        else:
            verdict = "INSUFFICIENT_EVIDENCE"
            confidence = 0.5
            
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": fallback_reasoning,
            "key_evidence": ["Evidence analysis incomplete due to technical limitations"],
            "verifier_score": 5,
            "opposer_score": 5,
            "evidence_quality": "MODERATE"
        }

class ClaimVerificationSystem:
    """Main system orchestrating the debate"""
    
    def __init__(self):
        self.client = LMStudioClient()
        self.scraper = WebScraper()
        self.verifier = VerifierAgent(AgentRole.VERIFIER, PHI_MODEL, self.client)
        self.opposer = OpposerAgent(AgentRole.OPPOSER, PHI_MODEL, self.client)
        self.judge = JudgeAgent(AgentRole.JUDGE, GEMMA_MODEL, self.client)
    
    def run_verification(self, claim: str, urls: List[str], num_rounds: int = 2) -> Dict[str, Any]:
        """Run the complete verification process"""
        
        # Initialize state
        state = DebateState(
            claim=claim,
            urls=urls,
            scraped_content=[],
            verifier_arguments=[],
            opposer_arguments=[],
            debate_rounds=num_rounds,
            current_round=0,
            final_judgment=None,
            debate_history=[]
        )
        
        try:
            # Step 1: Scrape URLs
            st.write("## 🔍 Scraping Evidence")
            state.scraped_content = self.scraper.scrape_urls(urls)
            
            successful_scrapes = [item for item in state.scraped_content if item['status'] == 'success']
            st.write(f"✅ Successfully scraped {len(successful_scrapes)} out of {len(urls)} URLs")
            
            # Step 2: Run debate rounds
            st.write("## 🎭 AI Agents Debate")
            
            for round_num in range(1, num_rounds + 1):
                st.write(f"### Round {round_num}")
                state.current_round = round_num
                
                # Verifier's turn
                try:
                    with st.spinner("🟢 Verifier Agent thinking..."):
                        verifier_arg = self.verifier.generate_argument(
                            claim, state.scraped_content, state.opposer_arguments, round_num
                        )
                        state.verifier_arguments.append(verifier_arg)
                    
                    st.write("**🟢 Verifier (Supporting the claim):**")
                    st.write(verifier_arg)
                except Exception as e:
                    st.error(f"Verifier error: {str(e)}")
                    state.verifier_arguments.append("Unable to generate argument due to technical issues.")
                
                # Opposer's turn
                try:
                    with st.spinner("🔴 Opposer Agent thinking..."):
                        opposer_arg = self.opposer.generate_argument(
                            claim, state.scraped_content, state.verifier_arguments, round_num
                        )
                        state.opposer_arguments.append(opposer_arg)
                    
                    st.write("**🔴 Opposer (Challenging the claim):**")
                    st.write(opposer_arg)
                except Exception as e:
                    st.error(f"Opposer error: {str(e)}")
                    state.opposer_arguments.append("Unable to generate argument due to technical issues.")
                
                # Record round
                state.debate_history.append({
                    'round': round_num,
                    'verifier_argument': state.verifier_arguments[-1] if state.verifier_arguments else "No argument generated",
                    'opposer_argument': state.opposer_arguments[-1] if state.opposer_arguments else "No argument generated"
                })
                
                st.write("---")
            
            # Step 3: Judge's final decision
            st.write("## ⚖️ Final Judgment")
            try:
                with st.spinner("🧑‍⚖️ Judge Agent deliberating..."):
                    judgment = self.judge.make_judgment(
                        claim, state.scraped_content, 
                        state.verifier_arguments, state.opposer_arguments
                    )
                    state.final_judgment = judgment
            except Exception as e:
                st.error(f"Judge error: {str(e)}")
                # Provide a fallback judgment
                judgment = {
                    "verdict": "INSUFFICIENT_EVIDENCE",
                    "confidence": 0.5,
                    "reasoning": f"Technical error prevented proper judgment: {str(e)}",
                    "key_evidence": ["Analysis incomplete due to technical issues"],
                    "verifier_score": 5,
                    "opposer_score": 5,
                    "evidence_quality": "MODERATE"
                }
                state.final_judgment = judgment
            
            return {
                'state': state,
                'judgment': judgment,
                'scraped_content': state.scraped_content,
                'debate_history': state.debate_history
            }
            
        except Exception as e:
            st.error(f"System error: {str(e)}")
            # Return minimal fallback results
            return {
                'state': state,
                'judgment': {
                    "verdict": "INSUFFICIENT_EVIDENCE",
                    "confidence": 0.0,
                    "reasoning": f"System encountered critical error: {str(e)}",
                    "key_evidence": ["Analysis failed due to system error"],
                    "verifier_score": 0,
                    "opposer_score": 0,
                    "evidence_quality": "WEAK"
                },
                'scraped_content': state.scraped_content,
                'debate_history': state.debate_history
            }

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Claim Verification System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 AI Claim Verification System")
    st.markdown("""
    **An AI-powered system that verifies claims through intelligent debate**
    
    This system uses multiple AI agents to analyze claims:
    - 🟢 **Verifier Agent**: Argues for the claim's truth
    - 🔴 **Opposer Agent**: Challenges the claim  
    - ⚖️ **Judge Agent**: Makes the final verdict
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check LM Studio connection
        st.subheader("LM Studio Connection")
        try:
            client = LMStudioClient()
            test_response = client.client.models.list()
            st.success("✅ Connected to LM Studio")
            
            models = [model.id for model in test_response.data]
            st.write("Available models:", models)
            
            if PHI_MODEL not in models:
                st.warning(f"⚠️ {PHI_MODEL} not found. Please load it in LM Studio.")
            if GEMMA_MODEL not in models:
                st.warning(f"⚠️ {GEMMA_MODEL} not found. Please load it in LM Studio.")
                
        except Exception as e:
            st.error(f"❌ Cannot connect to LM Studio: {str(e)}")
            st.info("Please ensure LM Studio is running on http://localhost:1234")
        
        # Debate configuration
        st.subheader("Debate Settings")
        num_rounds = st.slider("Number of debate rounds", 1, 5, 2)
        
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("1. Enter Your Claim")
        claim = st.text_area(
            "What claim would you like to verify?",
            placeholder="e.g., Climate change is primarily caused by human activities",
            height=100
        )
        
        st.header("2. Add Evidence URLs")
        urls = []
        
        # URL input system
        if 'url_count' not in st.session_state:
            st.session_state.url_count = 1
        
        for i in range(st.session_state.url_count):
            url = st.text_input(f"URL {i+1}:", key=f"url_{i}")
            if url:
                urls.append(url)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("➕ Add Another URL"):
                st.session_state.url_count += 1
                st.rerun()
        
        with col_b:
            if st.button("➖ Remove Last URL") and st.session_state.url_count > 1:
                st.session_state.url_count -= 1
                st.rerun()
        
        # Start verification
        if st.button("🚀 Start Verification", type="primary", disabled=not (claim and urls)):
            if claim and urls:
                with st.spinner("Initializing AI agents..."):
                    system = ClaimVerificationSystem()
                    
                start_time = time.time()
                results = system.run_verification(claim, urls, num_rounds)
                end_time = time.time()
                
                # Display results - with safe access to judgment
                st.success(f"✅ Verification completed in {end_time - start_time:.1f} seconds")
                
                judgment = results.get('judgment', {})
                
                # Safe access to judgment fields with defaults
                verdict = judgment.get('verdict', 'INSUFFICIENT_EVIDENCE')
                confidence = judgment.get('confidence', 0.0)
                reasoning = judgment.get('reasoning', 'No reasoning provided')
                verifier_score = judgment.get('verifier_score', 0)
                opposer_score = judgment.get('opposer_score', 0)
                key_evidence = judgment.get('key_evidence', [])
                
                # Verdict display
                verdict_color = {
                    'TRUE': '🟢',
                    'FALSE': '🔴', 
                    'INSUFFICIENT_EVIDENCE': '🟡'
                }
                
                st.header(f"## Final Verdict: {verdict_color.get(verdict, '❓')} {verdict}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col2:
                    st.metric("Verifier Score", f"{verifier_score}/10")
                with col3:
                    st.metric("Opposer Score", f"{opposer_score}/10")
                
                st.write("**Reasoning:**")
                st.write(reasoning)
                
                if key_evidence:
                    st.write("**Key Evidence:**")
                    for evidence in key_evidence:
                        st.write(f"• {evidence}")
                
                # Detailed results in expander
                with st.expander("📊 Detailed Results"):
                    tab1, tab2, tab3 = st.tabs(["🔍 Scraped Content", "🎭 Debate History", "📈 Analysis"])
                    
                    with tab1:
                        scraped_content = results.get('scraped_content', [])
                        for i, content in enumerate(scraped_content):
                            st.subheader(f"Source {i+1}: {content.get('title', 'No title')}")
                            st.write(f"**URL:** {content.get('url', 'No URL')}")
                            st.write(f"**Status:** {content.get('status', 'Unknown')}")
                            if content.get('status') == 'success':
                                st.write(f"**Content Preview:**")
                                content_text = content.get('content', '')
                                st.write(content_text[:500] + "..." if len(content_text) > 500 else content_text)
                            else:
                                st.error(f"Error: {content.get('error', 'Unknown error')}")
                            st.write("---")
                    
                    with tab2:
                        debate_history = results.get('debate_history', [])
                        for round_data in debate_history:
                            st.subheader(f"Round {round_data.get('round', '?')}")
                            st.write("**🟢 Verifier:**")
                            st.write(round_data.get('verifier_argument', 'No argument generated'))
                            st.write("**🔴 Opposer:**")
                            st.write(round_data.get('opposer_argument', 'No argument generated'))
                            st.write("---")
                    
                    with tab3:
                        # Create summary dataframe
                        evidence_quality = judgment.get('evidence_quality', 'N/A')
                        summary_data = {
                            'Metric': ['Evidence Quality', 'Confidence', 'Verifier Performance', 'Opposer Performance'],
                            'Value': [
                                evidence_quality,
                                f"{confidence:.2f}",
                                f"{verifier_score}/10",
                                f"{opposer_score}/10"
                            ]
                        }
                        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                        
                        # URLs success rate
                        scraped_content = results.get('scraped_content', [])
                        successful_scrapes = len([item for item in scraped_content if item.get('status') == 'success'])
                        total_urls = len(scraped_content)
                        if total_urls > 0:
                            success_rate = successful_scrapes/total_urls*100
                            st.metric("Scraping Success Rate", f"{successful_scrapes}/{total_urls} ({success_rate:.1f}%)")
                        else:
                            st.metric("Scraping Success Rate", "0/0 (0%)")
    
    with col2:
        st.header("📋 Instructions")
        st.markdown("""
        ### How it works:
        
        1. **Enter a claim** you want to verify
        
        2. **Add URLs** containing relevant evidence
        
        3. **AI agents debate**:
           - Verifier argues the claim is true
           - Opposer challenges the claim
           - Judge makes final verdict
        
        4. **Get results** with confidence scores and reasoning
        
        ### Tips:
        - Use reputable news sources
        - Include diverse perspectives  
        - More evidence = better analysis
        - Be specific with claims
        """)
        
        st.header("🔧 System Status")
        
        # System health checks
        try:
            client = LMStudioClient()
            st.success("✅ LM Studio Connected")
        except Exception as e:
            st.error("❌ LM Studio Disconnected")
        
        st.info("💡 Make sure LM Studio is running with the required models loaded")

if __name__ == "__main__":
    main()
