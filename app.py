import streamlit as st
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Literal
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
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import pandas as pd
import operator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import langgraph
        import langchain_core
        return True
    except ImportError as e:
        st.error(f"Missing dependencies: {str(e)}")
        st.error("Please install: pip install langgraph langchain-core")
        return False

class GraphState(TypedDict):
    """LangGraph state that gets passed between nodes"""
    claim: str
    urls: List[str]
    scraped_content: List[Dict[str, Any]]
    verifier_arguments: Annotated[List[str], operator.add]  # Automatically accumulates
    opposer_arguments: Annotated[List[str], operator.add]  # Automatically accumulates
    current_round: int
    max_rounds: int
    final_judgment: Optional[Dict[str, Any]]
    messages: Annotated[List[BaseMessage], operator.add]  # Message history
    next_action: str
    round_complete: bool
    error_message: Optional[str]
    debate_complete: bool
    retry_count: int  # Track retry attempts
    last_error_node: Optional[str]  # Track which node failed

# Configuration - Optimized for better responses
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
# Using specific LM Studio model names
QWEN_MODEL = "qwen/qwen3-1.7b"  # For Verifier and Counter-Explainer
PHI_MODEL = "microsoft/phi-4-mini-reasoning"  # For Judge
# Alternative lightweight models
FALLBACK_MODEL = "microsoft/DialoGPT-small"  # Backup option

# Updated configuration with much higher token limits
MAX_RESPONSE_TOKENS = 4096  # Significantly increased for fuller responses
MAX_CONTENT_LENGTH = 8000  # Increased to allow more content per source
MAX_FULL_TEXT_LENGTH = 12000  # Increased for fuller text processing

class AgentRole(Enum):
    VERIFIER = "verifier"
    COUNTER_EXPLAINER = "counter_explainer"
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
        """Generate response using specified model with improved context management"""
        try:
            # Better context management - keep essential context while truncating excess
            processed_messages = []
            for msg in messages:
                content = msg['content']
                # More intelligent truncation that preserves structure
                if len(content) > 8000:  # Increased from 4000
                    # Try to preserve key sections
                    lines = content.split('\n')
                    preserved_lines = []
                    char_count = 0
                    
                    for line in lines:
                        if char_count + len(line) > 7500:  # Increased from 3500
                            preserved_lines.append("...[content truncated for context management]...")
                            break
                        preserved_lines.append(line)
                        char_count += len(line)
                    
                    content = '\n'.join(preserved_lines)
                
                processed_messages.append({
                    'role': msg['role'],
                    'content': content
                })
            
            response = self.client.chat.completions.create(
                model=model,
                messages=processed_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                # Adjusted parameters for better output
                top_p=0.95,  # Slightly increased for more diverse responses
                frequency_penalty=0.2,  # Increased to reduce repetition
                presence_penalty=0.2  # Increased to encourage more original content
            )
            
            result = response.choices[0].message.content
            
            # Validate response quality
            if not self._validate_response(result):
                logger.warning("Generated response failed validation, attempting regeneration...")
                # Try once more with lower temperature for more focused response
                response = self.client.chat.completions.create(
                    model=model,
                    messages=processed_messages,
                    temperature=0.5,
                    max_tokens=max_tokens,
                    stream=False,
                    top_p=0.9
                )
                result = response.choices[0].message.content
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with {model}: {str(e)}")
            return f"Error: Could not generate response. Please ensure LM Studio is running and the model {model} is loaded."
    
    def _validate_response(self, response: str) -> bool:
        """Validate response quality and coherence"""
        if not response or len(response.strip()) < 50:
            return False
        
        # Check for repetitive patterns
        words = response.lower().split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Flag if any word appears more than 20% of the time
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.2:
                return False
        
        # Check for incomplete sentences
        sentences = response.split('.')
        incomplete_sentences = sum(1 for s in sentences if len(s.strip()) < 10 and s.strip())
        if incomplete_sentences > len(sentences) * 0.3:
            return False
        
        return True

class DebateAgent:
    """Base class for debate agents"""
    
    def __init__(self, role: AgentRole, model: str, client: LMStudioClient):
        self.role = role
        self.model = model
        self.client = client  # Add this line that was missing
        self.conversation_history = []
    
    def generate_argument(self, claim: str, evidence: List[Dict[str, Any]], 
                         opponent_arguments: List[str] = None, round_num: int = 1) -> str:
        """Generate argument based on role and evidence"""
        raise NotImplementedError

class VerifierAgent(DebateAgent):
    """Agent that argues in favor of the claim with improved prompting"""
    
    def generate_argument(self, claim: str, evidence: List[Dict[str, Any]], 
                         opponent_arguments: List[str] = None, round_num: int = 1) -> str:
        
        # Better evidence processing
        evidence_text = self._process_evidence(evidence)
        opponent_context = self._process_opponent_arguments(opponent_arguments, "Counter-Explainer")
        
        # More structured and specific prompt
        prompt = f"""You are a skilled fact-checker and debater. Your role is to present a strong, evidence-based argument supporting this claim:

CLAIM TO SUPPORT: "{claim}"

AVAILABLE EVIDENCE:
{evidence_text}

{opponent_context}

TASK FOR ROUND {round_num}:
Write a focused argument that SUPPORTS the claim. Your response must:

1. START with a clear thesis statement about why the claim is true
2. PRESENT 2-3 specific pieces of evidence from the sources
3. USE direct quotes or facts from the evidence (cite sources)
4. ADDRESS any counter-arguments raised by the opponent (if any)
5. END with a strong concluding statement

REQUIREMENTS:
- Keep your argument between 400-800 words
- Use logical reasoning and evidence-based arguments
- Stay focused on supporting the claim
- Be persuasive but factual
- Do not repeat previous arguments unless building upon them

Write your argument now:"""

        messages = [
            {
                "role": "system", 
                "content": """You are an expert fact-checker and debater specializing in evidence-based argumentation. You present clear, logical arguments supporting claims using credible sources. Always stay on topic and provide structured, coherent responses. Focus on quality over quantity."""
            },
            {"role": "user", "content": prompt}
        ]
        
        return self.client.generate_response(self.model, messages, temperature=0.6, max_tokens=2048)
    
    def _process_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        """Process evidence into a clean, structured format"""
        if not evidence:
            return "No evidence available."
        
        processed = []
        for i, item in enumerate(evidence[:5]):  # Limit to top 5 sources
            if item.get('status') == 'success':
                title = item.get('title', 'Unknown Source')[:200]  # Increased from 100
                url = item.get('url', '')
                content = item.get('content', '')[:2000]  # Increased from 800
                
                processed.append(f"""SOURCE {i+1}: {title}
URL: {url}
CONTENT: {content}
---""")
        
        return '\n\n'.join(processed) if processed else "No valid evidence found."
    
    def _process_opponent_arguments(self, arguments: List[str], opponent_name: str) -> str:
        """Process opponent arguments for context"""
        if not arguments:
            return ""
        
        formatted_args = []
        for i, arg in enumerate(arguments[-2:]):  # Only use last 2 arguments to avoid token overflow
            formatted_args.append(f"{opponent_name} Argument {i+1}: {arg[:1000]}...")  # Increased from 500
        
        return f"\n\nPREVIOUS {opponent_name.upper()} ARGUMENTS:\n" + '\n\n'.join(formatted_args)

class CounterExplainerAgent(DebateAgent):
    """Agent that provides alternative explanations with improved prompting"""
    
    def generate_argument(self, claim: str, evidence: List[Dict[str, Any]], 
                         opponent_arguments: List[str] = None, round_num: int = 1) -> str:
        
        evidence_text = self._process_evidence(evidence)
        verifier_context = self._process_opponent_arguments(opponent_arguments, "Verifier")
        
        # More structured prompt with clear role definition
        prompt = f"""You are a thoughtful analyst providing balanced perspective on complex topics. Your role is to offer nuanced analysis and alternative viewpoints.

CLAIM BEING DISCUSSED: "{claim}"

AVAILABLE EVIDENCE:
{evidence_text}

{verifier_context}

TASK FOR ROUND {round_num}:
Provide a balanced analysis that adds depth and nuance to the discussion. Your response should:

1. START with acknowledgment of any valid points from the Verifier
2. IDENTIFY alternative explanations or interpretations of the evidence
3. HIGHLIGHT important context or background information that adds nuance
4. POINT OUT any limitations, uncertainties, or missing perspectives
5. SUGGEST additional factors that should be considered
6. END with a balanced summary of the complexity involved

APPROACH:
- Be constructive and analytical, not simply oppositional
- Focus on adding depth rather than just disagreeing
- Use evidence to support alternative interpretations
- Acknowledge uncertainties and complexities
- Maintain intellectual honesty

REQUIREMENTS:
- Keep response between 500-1000 words
- Stay focused and avoid repetition
- Provide substantive analysis
- Be respectful of different viewpoints

Write your analysis now:"""

        messages = [
            {
                "role": "system", 
                "content": """You are a thoughtful analyst who provides nuanced perspectives on complex topics. You excel at identifying alternative explanations, adding important context, and highlighting the complexity of issues. You are constructive and balanced, seeking to enrich understanding rather than simply oppose. Always provide substantive, well-reasoned analysis."""
            },
            {"role": "user", "content": prompt}
        ]
        
        return self.client.generate_response(self.model, messages, temperature=0.7, max_tokens=2048)
    
    def _process_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        """Process evidence into a clean, structured format"""
        if not evidence:
            return "No evidence available."
        
        processed = []
        for i, item in enumerate(evidence[:5]):
            if item.get('status') == 'success':
                title = item.get('title', 'Unknown Source')[:100]
                url = item.get('url', '')
                content = item.get('content', '')[:800]
                
                processed.append(f"""SOURCE {i+1}: {title}
URL: {url}
CONTENT: {content}
---""")
        
        return '\n\n'.join(processed) if processed else "No valid evidence found."
    
    def _process_opponent_arguments(self, arguments: List[str], opponent_name: str) -> str:
        """Process opponent arguments for context"""
        if not arguments:
            return ""
        
        formatted_args = []
        for i, arg in enumerate(arguments[-2:]):
            formatted_args.append(f"{opponent_name} Argument {i+1}: {arg[:500]}...")
        
        return f"\n\nPREVIOUS {opponent_name.upper()} ARGUMENTS TO ANALYZE:\n" + '\n\n'.join(formatted_args)

class JudgeAgent(DebateAgent):
    """Agent that provides comprehensive debate analysis with structured, detailed prompting"""
    
    def make_judgment(self, claim: str, evidence: List[Dict[str, Any]], 
                     verifier_arguments: List[str], counter_explainer_arguments: List[str]) -> str:
        
        # Validate inputs first
        if not verifier_arguments or not counter_explainer_arguments:
            return "Insufficient debate content to analyze. At least one argument from each side is required."
        
        evidence_summary = self._summarize_evidence_safely(evidence)
        verifier_summary = self._format_arguments_for_analysis(verifier_arguments, "Verifier")
        counter_explainer_summary = self._format_arguments_for_analysis(counter_explainer_arguments, "Counter-Explainer")
        
        # Detailed, structured prompt similar to other agents
        prompt = f"""You are an expert debate judge and critical analyst. Your role is to provide a thorough, objective analysis of the debate between two AI agents regarding a specific claim.

CLAIM BEING ANALYZED: "{claim}"

EVIDENCE PROVIDED TO BOTH SIDES:
{evidence_summary}

VERIFIER ARGUMENTS (supporting the claim):
{verifier_summary}

COUNTER-EXPLAINER ARGUMENTS (providing alternative perspectives):
{counter_explainer_summary}

YOUR TASK AS JUDGE:
Provide a comprehensive analysis following this EXACT structure:

1. ARGUMENT QUALITY ASSESSMENT
   - Evaluate the logical structure of each side's arguments
   - Assess how well each side used the provided evidence
   - Identify the strongest points made by each side
   - Note any logical fallacies or weak reasoning

2. EVIDENCE UTILIZATION ANALYSIS
   - How effectively did each side reference the source material?
   - Were quotes and citations used appropriately?
   - Did either side misrepresent or overstate the evidence?
   - What evidence was overlooked or underutilized?

3. DEBATE DYNAMICS EVALUATION
   - How well did each side respond to their opponent's points?
   - Were counter-arguments addressed effectively?
   - Did the debate evolve constructively across rounds?
   - What were the key turning points in the debate?

4. CRITICAL GAPS AND LIMITATIONS
   - What important information or perspectives were missing?
   - What assumptions did each side make?
   - What questions remain unanswered?
   - How might additional evidence change the analysis?

5. OVERALL ASSESSMENT
   - Which side presented the more compelling case overall?
   - What are the key factors that influenced your assessment?
   - How confident can we be in any conclusions drawn?
   - What are the main takeaways from this debate?

ANALYSIS REQUIREMENTS:
- Base your analysis ONLY on the provided arguments and evidence
- Quote directly from the arguments when making specific points
- Be objective and identify strengths/weaknesses in both sides
- Acknowledge uncertainties and limitations explicitly
- Do NOT introduce external information not present in the materials
- Keep each section focused and substantive (100-200 words per section)
- Use clear, professional language throughout

Begin your structured analysis now:"""

        messages = [
            {
                "role": "system", 
                "content": """You are a professional debate judge and analytical expert. You provide structured, objective analysis of debates using only the information provided to you. You excel at:
- Evaluating argument quality and logical reasoning
- Assessing evidence utilization and source credibility
- Identifying strengths and weaknesses objectively
- Recognizing gaps and limitations in reasoning
- Providing balanced, nuanced assessments

You NEVER introduce information not present in the source materials. You quote directly from arguments when making specific points. You maintain strict objectivity and acknowledge when evidence is insufficient for strong conclusions."""
            },
            {"role": "user", "content": prompt}
        ]
        
        # Use lower temperature for more focused, structured responses
        response = self.client.generate_response(self.model, messages, temperature=0.3, max_tokens=2048)
        
        # Validate the response structure
        validated_response = self._validate_structured_response(response, verifier_arguments, counter_explainer_arguments, evidence)
        
        return validated_response
    
    def _format_arguments_for_analysis(self, arguments: List[str], role_name: str) -> str:
        """Format arguments in a clear, structured way for judge analysis"""
        if not arguments:
            return f"{role_name}: No arguments were presented during the debate."
        
        formatted = f"{role_name.upper()} PRESENTED {len(arguments)} ARGUMENT(S):\n\n"
        
        for i, arg in enumerate(arguments):
            # Add clear argument markers and preserve full content
            formatted += f"--- {role_name.upper()} ROUND {i+1} ARGUMENT ---\n"
            formatted += f"{arg}\n\n"
        
        return formatted
    
    def _validate_structured_response(self, response: str, verifier_args: List[str], 
                                counter_args: List[str], evidence: List[Dict[str, Any]]) -> str:
        """Validate that the judge response follows the required structure"""
        
        # Comment out the section validation warnings
        # required_sections = [
        #     "ARGUMENT QUALITY ASSESSMENT",
        #     "EVIDENCE UTILIZATION ANALYSIS", 
        #     "DEBATE DYNAMICS EVALUATION",
        #     "CRITICAL GAPS AND LIMITATIONS",
        #     "OVERALL ASSESSMENT"
        # ]
        
        # missing_sections = []
        # for section in required_sections:
        #     if section.lower() not in response.lower():
        #         missing_sections.append(section)
        
        # # If response is missing key sections, flag it
        # if missing_sections:
        #     warning = f"\n\n[JUDGE ANALYSIS NOTE: This response appears to be missing the following required sections: {', '.join(missing_sections)}. The analysis may be incomplete.]"
        #     response += warning
        
        # Comment out the hallucination check warnings
        # response = self._check_for_hallucinations(response, verifier_args, counter_args, evidence)
        
        # Comment out the minimum length warning
        # if len(response.split()) < 400:
        #     response += "\n\n[JUDGE ANALYSIS NOTE: This analysis appears shorter than expected for a comprehensive debate evaluation. Additional detail may be needed.]"
        
        return response
    
    def _check_for_hallucinations(self, response: str, verifier_args: List[str], 
                                counter_args: List[str], evidence: List[Dict[str, Any]]) -> str:
        """Enhanced hallucination detection for structured judge responses - DISABLED"""
        
        # All validation checks are commented out to remove warnings
        # The method now just returns the response unchanged
        return response
    
    def _summarize_evidence_safely(self, evidence: List[Dict[str, Any]]) -> str:
        """Enhanced evidence summary with better structure"""
        if not evidence:
            return "No evidence sources were provided for analysis."
        
        successful_sources = [e for e in evidence if e.get('status') == 'success']
        failed_sources = len(evidence) - len(successful_sources)
        
        if not successful_sources:
            return f"EVIDENCE STATUS: All {len(evidence)} evidence sources failed to load successfully.\nNo source content available for analysis."
        
        summary = f"EVIDENCE STATUS: Successfully loaded {len(successful_sources)} out of {len(evidence)} sources.\n"
        if failed_sources > 0:
            summary += f"Note: {failed_sources} source(s) failed to load and are not included in this analysis.\n\n"
        
        summary += "AVAILABLE EVIDENCE SOURCES:\n\n"
        
        for i, source in enumerate(successful_sources[:4]):  # Limit to top 4 to prevent overflow
            title = source.get('title', 'No title available')[:100]
            url = source.get('url', 'No URL')
            content_preview = source.get('content', 'No content')[:400]  # Increased preview
            
            summary += f"SOURCE {i+1}:\n"
            summary += f"Title: {title}\n"
            summary += f"URL: {url}\n"
            summary += f"Content Preview: {content_preview}\n"
            summary += f"[Content length: ~{len(source.get('content', ''))} characters]\n\n"
        
        if len(successful_sources) > 4:
            summary += f"[{len(successful_sources) - 4} additional sources available but not shown in detail to manage context length]\n\n"
        
        return summary

class ScoringAgent:
    """Enhanced scoring agent with strict anti-hallucination measures"""
    
    def __init__(self, client: LMStudioClient, model: str):
        self.client = client
        self.model = model
    
    def score_debate(self, judge_summary: str, claim: str) -> Dict[str, Any]:
        """Convert judge summary into structured verdict with strict validation"""
        
        prompt = f"""Convert the judge's analysis into a structured verdict. Base your scoring ONLY on the judge's actual analysis.

ORIGINAL CLAIM: "{claim}"

JUDGE'S ANALYSIS (USE ONLY THIS INFORMATION):
{judge_summary}

TASK: Create a JSON verdict based SOLELY on the judge's analysis above.

STRICT RULES:
- Use ONLY information present in the judge's analysis
- Do NOT add external knowledge or assumptions
- If the judge's analysis is unclear, use "INSUFFICIENT_EVIDENCE"
- Base confidence on how clearly the judge reached conclusions
- Quote directly from the judge's analysis for reasoning

OUTPUT ONLY THIS JSON FORMAT:
{{
    "verdict": "TRUE/FALSE/INSUFFICIENT_EVIDENCE",
    "confidence": 0.0-1.0,
    "reasoning": "Brief summary based only on judge's analysis",
    "evidence_quality": "STRONG/MODERATE/WEAK",
    "winning_side": "verifier/counter_explainer/tie"
}}

JSON OUTPUT:"""

        messages = [
            {
                "role": "system", 
                "content": "You convert judge analyses into structured JSON verdicts. You use ONLY the information provided in the judge's analysis. You never add external information or make assumptions beyond what the judge stated. Output valid JSON only."
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use very low temperature to minimize creativity/hallucination
            response = self.client.generate_response(self.model, messages, temperature=0.1, max_tokens=600)
            
            # Enhanced JSON extraction with validation
            verdict = self._extract_and_validate_json(response, judge_summary)
            if verdict:
                return self._validate_verdict_against_source(verdict, judge_summary)
                
        except Exception as e:
            logger.error(f"Error parsing scoring response: {str(e)}")
        
        # Enhanced fallback scoring based only on judge's actual text
        return self._create_evidence_based_fallback(judge_summary, claim)
    
    def _extract_and_validate_json(self, response: str, judge_summary: str) -> Dict[str, Any]:
        """Extract JSON and validate against source material"""
        verdict = self._extract_json_verdict(response)
        if not verdict:
            return None
        
        # Validate that reasoning is grounded in judge's analysis
        reasoning = verdict.get('reasoning', '')
        if reasoning and len(reasoning) > 50:
            # Check if reasoning contains concepts actually mentioned in judge summary
            judge_lower = judge_summary.lower()
            reasoning_lower = reasoning.lower()
            
            # Flag if reasoning contains terms not in judge summary
            key_terms = reasoning_lower.split()
            unfounded_terms = 0
            
            for term in key_terms:
                if len(term) > 4 and term not in judge_lower:  # Skip short common words
                    unfounded_terms += 1
            
            # If too many unfounded terms, mark as uncertain
            if unfounded_terms > len(key_terms) * 0.3:
                verdict['confidence'] = min(verdict.get('confidence', 0.5), 0.4)
                verdict['reasoning'] = f"Analysis based on judge's summary with some interpretation uncertainty."
        
        return verdict
    
    def _validate_verdict_against_source(self, verdict: Dict[str, Any], judge_summary: str) -> Dict[str, Any]:
        """Validate verdict components against judge's actual analysis"""
        
        # Extract key evidence points
        key_evidence = self._extract_key_evidence(judge_summary)
        verdict['key_evidence'] = key_evidence
        
        # Rest of the existing validation code...
        judge_lower = judge_summary.lower()
        
        certainty_indicators = {
            'high': ['clearly', 'definitively', 'conclusively', 'overwhelmingly', 'strongly supports'],
            'medium': ['likely', 'appears', 'suggests', 'indicates', 'moderately'],
            'low': ['unclear', 'uncertain', 'mixed', 'inconclusive', 'difficult to determine']
        }
        
        detected_certainty = 'medium'  # default
        
        for level, indicators in certainty_indicators.items():
            if any(indicator in judge_lower for indicator in indicators):
                detected_certainty = level
                break
        
        # Adjust confidence based on detected certainty
        if detected_certainty == 'high' and verdict['confidence'] < 0.7:
            verdict['confidence'] = min(0.8, verdict['confidence'] + 0.2)
        elif detected_certainty == 'low' and verdict['confidence'] > 0.6:
            verdict['confidence'] = max(0.4, verdict['confidence'] - 0.2)
        
        # Ensure verdict consistency
        if verdict['verdict'] == 'INSUFFICIENT_EVIDENCE':
            verdict['confidence'] = min(verdict['confidence'], 0.6)
        
        return verdict
    
    def _create_evidence_based_fallback(self, judge_summary: str, claim: str) -> Dict[str, Any]:
        """Create fallback verdict based strictly on judge's text analysis"""
        
        if not judge_summary or len(judge_summary.strip()) < 50:
            return {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.1,
                "reasoning": "Judge's analysis was too brief or missing for proper evaluation.",
                "evidence_quality": "WEAK",
                "winning_side": "tie",
                "key_evidence": ["No analysis available to extract key evidence points"]
            }
        
        # Extract key evidence even for fallback scoring
        key_evidence = self._extract_key_evidence(judge_summary)
        
        judge_lower = judge_summary.lower()
        
        # Rest of the existing fallback code...
        strong_support_keywords = ['claim is true', 'evidence clearly supports', 'verifier presented compelling']
        moderate_support_keywords = ['supports the claim', 'evidence suggests', 'likely true']
        strong_challenge_keywords = ['claim is false', 'evidence contradicts', 'counter-explainer convincingly']
        moderate_challenge_keywords = ['challenges the claim', 'alternative explanation', 'questionable']
        uncertainty_keywords = ['insufficient evidence', 'unclear', 'both sides', 'mixed results', 'inconclusive']
        
        # Count actual occurrences in judge's text
        strong_support_count = sum(1 for kw in strong_support_keywords if kw in judge_lower)
        moderate_support_count = sum(1 for kw in moderate_support_keywords if kw in judge_lower)
        strong_challenge_count = sum(1 for kw in strong_challenge_keywords if kw in judge_lower)
        moderate_challenge_count = sum(1 for kw in moderate_challenge_keywords if kw in judge_lower)
        uncertainty_count = sum(1 for kw in uncertainty_keywords if kw in judge_lower)
        
        # Conservative scoring based on actual text content
        total_support = strong_support_count * 2 + moderate_support_count
        total_challenge = strong_challenge_count * 2 + moderate_challenge_count
        
        if uncertainty_count >= 2 or (total_support == 0 and total_challenge == 0):
            verdict = "INSUFFICIENT_EVIDENCE"
            confidence = 0.3 + min(0.3, uncertainty_count * 0.1)
            winning_side = "tie"
        elif total_support > total_challenge:
            verdict = "TRUE"
            confidence = 0.5 + min(0.3, (total_support - total_challenge) * 0.1)
            winning_side = "verifier"
        elif total_challenge > total_support:
            verdict = "FALSE"
            confidence = 0.5 + min(0.3, (total_challenge - total_support) * 0.1)
            winning_side = "counter_explainer"
        else:
            verdict = "INSUFFICIENT_EVIDENCE"
            confidence = 0.4
            winning_side = "tie"
        
        return {
            "verdict": verdict,
            "confidence": min(confidence, 0.8),  # Cap confidence for fallback scoring
            "reasoning": f"Analysis based on judge's summary containing {total_support} supporting and {total_challenge} challenging indicators.",
            "evidence_quality": "MODERATE" if total_support + total_challenge > 0 else "WEAK",
            "winning_side": winning_side,
            "key_evidence": key_evidence  # Add extracted key evidence
        }
    
    def _extract_json_verdict(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response with multiple strategies"""
        # Strategy 1: Find JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Look for JSON between markers
        markers = ['\`\`\`json', '\`\`\`', 'JSON:', 'json:']
        for marker in markers:
            if marker in response.lower():
                parts = response.lower().split(marker)
                if len(parts) > 1:
                    json_part = parts[1].split('\`\`\`')[0] if '\`\`\`' in parts[1] else parts[1]
                    try:
                        return json.loads(json_part)
                    except json.JSONDecodeError:
                        continue
        
        return None

    def _extract_key_evidence(self, judge_summary: str) -> List[str]:
        """Extract key evidence points from judge's summary"""
        # Split into sentences and look for quoted text and key phrases
        sentences = judge_summary.split('.')
        key_evidence = []
        
        # Look for sentences with quotes or key indicator phrases
        evidence_indicators = [
            'evidence shows', 'according to', 'demonstrates', 'shows that',
            'proves', 'indicates', 'quoted', 'stated', 'mentioned',
            'key point', 'important evidence', 'crucial finding',
            'significant fact', 'notable point'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip empty or very short sentences
            if len(sentence) < 10:
                continue
                
            # Check for quoted content
            if '"' in sentence or '"' in sentence or '"' in sentence:
                key_evidence.append(sentence)
                continue
            
            # Check for evidence indicator phrases
            if any(indicator in sentence.lower() for indicator in evidence_indicators):
                key_evidence.append(sentence)
                continue
        
        # If no key evidence found through quotes or indicators,
        # look for sentences that mention specific facts or findings
        if not key_evidence:
            fact_indicators = ['found', 'discovered', 'revealed', 'showed', 'confirmed']
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                if any(indicator in sentence.lower() for indicator in fact_indicators):
                    key_evidence.append(sentence)
        
        # If still no key evidence, take the most relevant-looking sentences
        if not key_evidence:
            relevant_sentences = [s.strip() for s in sentences if len(s.strip()) > 30 and 
                                ('evidence' in s.lower() or 'argument' in s.lower())][:3]
            key_evidence.extend(relevant_sentences)
        
        # Clean up the evidence points
        cleaned_evidence = []
        for evidence in key_evidence:
            # Remove common prefixes that might have been picked up
            evidence = re.sub(r'^(In addition,|Moreover,|Furthermore,|Additionally,)\s*', '', evidence)
            # Clean up any extra whitespace
            evidence = ' '.join(evidence.split())
            cleaned_evidence.append(evidence)
        
        # Limit to top 5 most relevant points and ensure they're unique
        unique_evidence = list(dict.fromkeys(cleaned_evidence))[:5]
        
        # If we somehow have no evidence points, add a fallback
        if not unique_evidence:
            unique_evidence = ["No specific key evidence points were identified in the judge's analysis"]
        
        return unique_evidence

class LangGraphClaimVerificationSystem:
    """LangGraph-based claim verification system"""
    
    def __init__(self):
        self.client = LMStudioClient()
        self.scraper = WebScraper()
        # Add memory saver for state persistence
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def validate_state(self, state: GraphState) -> bool:
        """Validate state transitions"""
        required_fields = ["claim", "urls", "current_round", "max_rounds"]
        return all(field in state for field in required_fields)
    
    def route_after_scraping(self, state: GraphState) -> Literal["success", "retry", "error"]:
        """Enhanced routing after scraping"""
        if state.get("error_message"):
            retry_count = state.get("retry_count", 0)
            if retry_count < 3:
                return "retry"
            else:
                return "error"
        
        successful_scrapes = [item for item in state.get("scraped_content", []) if item.get('status') == 'success']
        if successful_scrapes:
            return "success"
        else:
            return "retry" if state.get("retry_count", 0) < 3 else "error"

    def route_after_agent(self, state: GraphState) -> Literal["success", "retry", "error"]:
        """Enhanced routing after agent execution"""
        if state.get("error_message"):
            retry_count = state.get("retry_count", 0)
            return "retry" if retry_count < 3 else "error"
        return "success"

    def should_continue_debate(self, state: GraphState) -> Literal["continue", "end", "error"]:
        """Enhanced debate continuation logic"""
        if state.get("error_message"):
            return "error"
        
        current_round = state.get("current_round", 1)
        max_rounds = state.get("max_rounds", 2)
        
        # Ensure we have arguments from both sides
        verifier_args = len(state.get("verifier_arguments", []))
        opposer_args = len(state.get("opposer_arguments", []))
        
        if current_round > max_rounds:
            return "end"
        elif verifier_args == 0 or opposer_args == 0:
            return "continue"  # Need at least one argument from each side
        else:
            return "continue" if current_round <= max_rounds else "end"

    def route_retry(self, state: GraphState) -> Literal["scrape", "verifier", "counter_explainer", "error"]:
        """Route retry attempts to the appropriate node"""
        last_error_node = state.get("last_error_node")
        
        if last_error_node == "scrape_evidence":
            return "scrape"
        elif last_error_node == "verifier_turn":
            return "verifier"
        elif last_error_node == "counter_explainer_turn":
            return "counter_explainer"
        else:
            return "error"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with enhanced error handling and retries"""
        try:
            workflow = StateGraph(GraphState)
            
            # Add nodes including retry handler
            workflow.add_node("scrape_evidence", self.scrape_evidence_node)
            workflow.add_node("verifier_turn", self.verifier_node)
            workflow.add_node("counter_explainer_turn", self.counter_explainer_node)
            workflow.add_node("judge_decision", self.judge_node)
            workflow.add_node("check_rounds", self.check_rounds_node)
            workflow.add_node("error_handler", self.error_handler_node)
            workflow.add_node("retry_handler", self.retry_handler_node)
            
            # Define edges with retry logic
            workflow.add_edge(START, "scrape_evidence")
            
            workflow.add_conditional_edges(
                "scrape_evidence",
                self.route_after_scraping,
                {
                    "success": "verifier_turn",
                    "retry": "retry_handler",
                    "error": "error_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "verifier_turn",
                self.route_after_agent,
                {
                    "success": "counter_explainer_turn",
                    "retry": "retry_handler", 
                    "error": "error_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "counter_explainer_turn",
                self.route_after_agent,
                {
                    "success": "check_rounds",
                    "retry": "retry_handler",
                    "error": "error_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "check_rounds",
                self.should_continue_debate,
                {
                    "continue": "verifier_turn",
                    "end": "judge_decision",
                    "error": "error_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "retry_handler",
                self.route_retry,
                {
                    "scrape": "scrape_evidence",
                    "verifier": "verifier_turn", 
                    "counter_explainer": "counter_explainer_turn",
                    "error": "error_handler"
                }
            )
            
            workflow.add_edge("judge_decision", END)
            workflow.add_edge("error_handler", END)
            
            return workflow.compile(checkpointer=self.memory)
            
        except Exception as e:
            st.error(f"Failed to build LangGraph: {str(e)}")
            raise e
    
    def validate_state_node(self, state: GraphState) -> GraphState:
        """Node: Validate initial state"""
        if not self.validate_state(state):
            new_state = dict(state)
            new_state["error_message"] = "Invalid state: missing required fields"
            return new_state
        return state
    
    def check_state_valid(self, state: GraphState) -> Literal["valid", "invalid"]:
        """Check if state is valid"""
        return "valid" if not state.get("error_message") else "invalid"
    
    def scrape_evidence_node(self, state: GraphState) -> GraphState:
        """Node: Scrape evidence from URLs"""
        st.write("## 🔍 Scraping Evidence")
        
        try:
            scraped_content = self.scraper.scrape_urls(state["urls"])
            successful_scrapes = [item for item in scraped_content if item['status'] == 'success']
            st.write(f"✅ Successfully scraped {len(successful_scrapes)} out of {len(state['urls'])} URLs")
            
            return {
                **state,
                "scraped_content": scraped_content,
                "messages": state["messages"] + [HumanMessage(content=f"Scraped {len(successful_scrapes)} sources successfully")],
                "error_message": None,
                "retry_count": 0
            }
            
        except Exception as e:
            st.error(f"Scraping failed: {str(e)}")
            return {
                **state,
                "error_message": f"Scraping failed: {str(e)}",
                "scraped_content": [],
                "messages": state["messages"] + [HumanMessage(content=f"Scraping failed: {str(e)}")],
                "retry_count": state.get("retry_count", 0) + 1,
                "last_error_node": "scrape_evidence"
            }
    
    def verifier_node(self, state: GraphState) -> GraphState:
        """Node: Generate verifier argument"""
        round_num = state["current_round"]
        st.write(f"### Round {round_num}")
        
        try:
            with st.spinner("🟢 Verifier Agent thinking..."):
                verifier = VerifierAgent(AgentRole.VERIFIER, QWEN_MODEL, self.client)
                argument = verifier.generate_argument(
                    state["claim"], 
                    state["scraped_content"], 
                    state["opposer_arguments"], 
                    round_num
                )
            
            st.write("**🟢 Verifier (Supporting the claim):**")
            st.write(argument)
            
            return {
                **state,
                "verifier_arguments": state["verifier_arguments"] + [argument],
                "messages": state["messages"] + [AIMessage(content=f"Verifier Round {round_num}: {argument}")],
                "error_message": None,
                "retry_count": 0
            }
            
        except Exception as e:
            st.error(f"Verifier error: {str(e)}")
            return {
                **state,
                "error_message": f"Verifier error: {str(e)}",
                "verifier_arguments": state["verifier_arguments"] + [f"Error in round {round_num}: Unable to generate argument"],
                "retry_count": state.get("retry_count", 0) + 1,
                "last_error_node": "verifier_turn"
            }
    
    def counter_explainer_node(self, state: GraphState) -> GraphState:
        """Node: Generate counter-explainer analysis"""
        round_num = state["current_round"]
        
        try:
            with st.spinner("🔄 Counter-Explainer Agent analyzing..."):
                counter_explainer = CounterExplainerAgent(AgentRole.COUNTER_EXPLAINER, QWEN_MODEL, self.client)
                argument = counter_explainer.generate_argument(
                    state["claim"], 
                    state["scraped_content"], 
                    state["verifier_arguments"], 
                    round_num
                )
            
            st.write("**🔄 Counter-Explainer (Providing alternative perspectives):**")
            st.write(argument)
            
            return {
                **state,
                "opposer_arguments": state["opposer_arguments"] + [argument],
                "messages": state["messages"] + [AIMessage(content=f"Counter-Explainer Round {round_num}: {argument}")],
                "error_message": None,
                "retry_count": 0
            }
            
        except Exception as e:
            st.error(f"Counter-Explainer error: {str(e)}")
            return {
                **state,
                "error_message": f"Counter-Explainer error: {str(e)}",
                "opposer_arguments": state["opposer_arguments"] + [f"Error in round {round_num}: Unable to generate analysis"],
                "retry_count": state.get("retry_count", 0) + 1,
                "last_error_node": "counter_explainer_turn"
            }
    
    def check_rounds_node(self, state: GraphState) -> GraphState:
        """Node: Check if we should continue the debate"""
        return {
            **state,
            "current_round": state["current_round"] + 1,
            "round_complete": True,
            "error_message": None
        }
    
    def judge_node(self, state: GraphState) -> GraphState:
        """Node: Generate natural language summary and structured verdict"""
        st.write("## ⚖️ Final Judgment")
        
        try:
            with st.spinner("🧑‍⚖️ Judge Agent analyzing debate..."):
                judge = JudgeAgent(AgentRole.JUDGE, PHI_MODEL, self.client)
                judge_summary = judge.make_judgment(
                    state["claim"], 
                    state["scraped_content"], 
                    state["verifier_arguments"], 
                    state["opposer_arguments"]
                )
            
            st.write("### 📝 Judge's Analysis")
            display_judge_analysis(judge_summary)
            
            with st.spinner("📊 Scoring the debate..."):
                scoring_agent = ScoringAgent(self.client, PHI_MODEL)
                structured_verdict = scoring_agent.score_debate(judge_summary, state["claim"])
            
            # Combine judge summary with structured verdict
            final_judgment = {
                **structured_verdict,
                "judge_summary": judge_summary
            }
            
            return {
                **state,
                "final_judgment": final_judgment,
                "debate_complete": True,
                "messages": state["messages"] + [AIMessage(content=f"Final judgment: {structured_verdict['verdict']} with {structured_verdict['confidence']:.2f} confidence")],
                "error_message": None
            }
            
        except Exception as e:
            st.error(f"Judge/Scoring error: {str(e)}")
            fallback_judgment = {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.5,
                "reasoning": f"Technical error prevented proper judgment: {str(e)}",
                "evidence_quality": "MODERATE",
                "winning_side": "tie",
                "judge_summary": f"Analysis incomplete due to technical error: {str(e)}"
            }
            
            return {
                **state,
                "final_judgment": fallback_judgment,
                "debate_complete": True,
                "error_message": f"Judge error: {str(e)}",
                "messages": state["messages"] + [AIMessage(content="Final judgment completed with errors")]
            }
    
    def check_scraping_success(self, state: GraphState) -> Literal["success", "error"]:
        """Check if scraping was successful"""
        if state.get("error_message"):
            return "error"
        successful_scrapes = [item for item in state.get("scraped_content", []) if item.get('status') == 'success']
        return "success" if successful_scrapes else "error"
    
    def check_agent_success(self, state: GraphState) -> Literal["success", "error"]:  
        """Check if agent execution was successful"""
        return "error" if state.get("error_message") else "success"
    
    def error_handler_node(self, state: GraphState) -> GraphState:
        """Node: Handle errors gracefully"""
        error_msg = state.get("error_message", "Unknown error occurred")
        st.error(f"System Error: {error_msg}")
        
        # Create fallback judgment with updated structure
        fallback_judgment = {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "confidence": 0.0,
            "reasoning": f"Analysis failed due to system error: {error_msg}",
            "evidence_quality": "WEAK",
            "winning_side": "tie",
            "judge_summary": f"Analysis failed due to system error: {error_msg}"
        }
        
        return {
            "final_judgment": fallback_judgment,
            "debate_complete": True,
            "messages": [AIMessage(content=f"Process terminated due to error: {error_msg}")]
        }
    
    def retry_handler_node(self, state: GraphState) -> GraphState:
        """Node: Handle retries with exponential backoff"""
        retry_count = state.get("retry_count", 0)
        max_retries = 3
        
        if retry_count >= max_retries:
            return {
                **state,
                "error_message": f"Max retries ({max_retries}) exceeded for node: {state.get('last_error_node', 'unknown')}",
                "debate_complete": True
            }
        
        # Exponential backoff
        sleep_time = 2 ** retry_count
        st.info(f"Retrying in {sleep_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
        time.sleep(sleep_time)
        
        return state
    
    def run_verification(self, claim: str, urls: List[str], num_rounds: int = 2) -> Dict[str, Any]:
        """Run the complete verification process using LangGraph"""
        
        # Check dependencies first
        if not check_dependencies():
            return {
                'success': False,
                'error': 'Missing required dependencies'
            }
        
        # Initialize the state
        initial_state: GraphState = {
            "claim": claim,
            "urls": urls,
            "scraped_content": [],
            "verifier_arguments": [],
            "opposer_arguments": [],
            "current_round": 1,
            "max_rounds": num_rounds,
            "final_judgment": None,
            "messages": [HumanMessage(content=f"Starting verification for claim: {claim}")],
            "next_action": "scrape",
            "round_complete": False,
            "error_message": None,
            "debate_complete": False,
            "retry_count": 0,
            "last_error_node": None
        }
        
        try:
            st.write("🚀 **Starting LangGraph Execution**")
            
            # Create a unique thread ID for this verification session
            thread_id = f"verification_{int(time.time())}"
            config = RunnableConfig(configurable={"thread_id": thread_id})
            
            # Execute the graph with proper config
            final_state = self.graph.invoke(initial_state, config=config)
            
            st.success("✅ LangGraph execution completed successfully")
            
            return {
                'state': final_state,
                'judgment': final_state.get('final_judgment', {}),
                'scraped_content': final_state.get('scraped_content', []),
                'debate_history': self._extract_debate_history(final_state),
                'messages': final_state.get('messages', []),
                'success': True,
                'error': final_state.get('error_message')
            }
            
        except Exception as e:
            error_msg = f"LangGraph execution error: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)
            
            return {
                'state': initial_state,
                'judgment': {
                    "verdict": "INSUFFICIENT_EVIDENCE",
                    "confidence": 0.0,
                    "reasoning": error_msg,
                    "key_evidence": ["Analysis failed due to system error"],
                    "verifier_score": 0,
                    "opposer_score": 0,
                    "evidence_quality": "WEAK"
                },
                'scraped_content': [],
                'debate_history': [],
                'messages': [HumanMessage(content=error_msg)],
                'success': False,
                'error': error_msg
            }
    
    def _extract_debate_history(self, final_state: GraphState) -> List[Dict[str, Any]]:
        """Extract debate history from the final state"""
        history = []
        verifier_args = final_state.get('verifier_arguments', [])
        opposer_args = final_state.get('opposer_arguments', [])
        
        max_rounds = min(len(verifier_args), len(opposer_args))
        
        for i in range(max_rounds):
            history.append({
                'round': i + 1,
                'verifier_argument': verifier_args[i] if i < len(verifier_args) else "No argument generated",
                'opposer_argument': opposer_args[i] if i < len(opposer_args) else "No argument generated"
            })
        
        return history

def display_judge_analysis(judge_summary: str):
    """Display judge analysis with LaTeX support"""
    # Check if the response contains LaTeX formatting
    if r'\boxed{' in judge_summary or r'\text{' in judge_summary:
        # Try to render LaTeX
        try:
            # Split the text to find LaTeX portions
            import re
            
            # Find boxed content
            boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', judge_summary)
            
            if boxed_matches:
                # Clean the text for display
                clean_text = re.sub(r'\\boxed\{([^}]+)\}', r'**\1**', judge_summary)
                clean_text = re.sub(r'\\text\{([^}]+)\}', r'\1', clean_text)
                st.write(clean_text)
                
                # Display the boxed content prominently
                for match in boxed_matches:
                    clean_match = re.sub(r'\\text\{([^}]+)\}', r'\1', match)
                    st.success(f"**Final Conclusion:** {clean_match}")
            else:
                st.write(judge_summary)
        except Exception as e:
            # Fallback to plain text
            st.write(judge_summary)
    else:
        st.write(judge_summary)

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Claim Verification System",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ AI Claim Verification System")
    st.write("""
    This system uses AI agents to verify claims by searching for evidence online and conducting a structured debate.
    The agents will analyze the evidence, present arguments for and against the claim, and make a final judgment.
    """)
    
    # Input section
    st.write("## 📝 Input")
    claim = st.text_area("Enter the claim to verify:", height=100)
    
    # URL input
    urls_text = st.text_area(
        "Enter URLs with relevant evidence (one per line):",
        height=150,
        help="Enter each URL on a new line. The system will scrape these pages for evidence."
    )
    
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    
    # Number of debate rounds
    num_rounds = st.slider("Number of debate rounds:", min_value=1, max_value=5, value=2)
    
    # Verification button
    if st.button("🚀 Start Verification", type="primary", disabled=not (claim and urls)):
        if claim and urls:
            with st.spinner("Initializing LangGraph AI system..."):
                system = LangGraphClaimVerificationSystem()
                
            start_time = time.time()
            results = system.run_verification(claim, urls, num_rounds)
            end_time = time.time()
            
            st.success(f"✅ LangGraph verification completed in {end_time - start_time:.1f} seconds")
            
            # Display judgment
            judgment = results['judgment']
            if judgment:
                st.write("## 📊 Final Results")
                
                # Create columns for the verdict display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Verdict", judgment['verdict'])
                with col2:
                    st.metric("Confidence", f"{judgment['confidence']:.2%}")
                with col3:
                    st.metric("Evidence Quality", judgment['evidence_quality'])
            
            # Display scraped sources with full content
            st.write("## 📚 Sources")
            for i, source in enumerate(results['scraped_content']):
                # Create a more descriptive title for the expander
                source_title = source.get('title', 'No Title Available')
                if len(source_title) > 80:
                    source_title = source_title[:80] + "..."
                
                status_emoji = "✅" if source['status'] == 'success' else "❌"
                expander_title = f"{status_emoji} Source {i+1}: {source_title}"
                
                with st.expander(expander_title):
                    st.write(f"**URL:** {source['url']}")
                    st.write(f"**Title:** {source.get('title', 'No title available')}")
                    st.write(f"**Status:** {source['status']}")
                    st.write(f"**Scraped at:** {source.get('scraped_at', 'Unknown time')}")
                    
                    if source['status'] == 'success':
                        # Show content statistics
                        content_length = len(source.get('content', ''))
                        full_text_length = len(source.get('full_text', ''))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Content Length", f"{content_length:,} chars")
                        with col2:
                            st.metric("Full Text Length", f"{full_text_length:,} chars")
                        
                        # Display full content in expandable sections
                        st.write("### 📄 Main Content")
                        main_content = source.get('content', 'No content available')
                        if main_content and main_content.strip():
                            st.text_area(
                                "Main Content (used by AI agents):",
                                value=main_content,
                                height=300,
                                key=f"main_content_{i}",
                                help="This is the main content that was extracted and used by the AI agents for analysis."
                            )
                        else:
                            st.info("No main content was extracted from this source.")
                        
                        # Display full text if different from main content
                        full_text = source.get('full_text', '')
                        if full_text and full_text.strip() and full_text != main_content:
                            st.write("### 📋 Full Text")
                            st.text_area(
                                "Complete scraped text:",
                                value=full_text,
                                height=400,
                                key=f"full_text_{i}",
                                help="This is the complete text that was scraped from the webpage, including all content."
                            )
                        
                        # Show a preview of the raw HTML structure if available
                        if st.checkbox(f"Show technical details for Source {i+1}", key=f"tech_details_{i}"):
                            st.write("### 🔧 Technical Details")
                            st.json({
                                "URL": source.get('url', ''),
                                "Title": source.get('title', ''),
                                "Content Length": len(source.get('content', '')),
                                "Full Text Length": len(source.get('full_text', '')),
                                "Scraped At": source.get('scraped_at', ''),
                                "Status": source.get('status', '')
                            })
                    else:
                        st.error(f"**Error:** {source.get('error', 'Unknown error occurred during scraping')}")
                        st.info("This source could not be scraped and was not used in the analysis.")
        else:
            st.warning("Please enter a claim and at least one URL to verify.")

if __name__ == "__main__":
    main()
