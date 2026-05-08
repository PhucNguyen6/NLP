# -*- coding: utf-8 -*-
"""
LLM Client - Integration with LM Studio (Mistral) for sentiment explanation
Handles API calls, prompt engineering, and response processing
"""

import logging
import json
from typing import Dict, Optional, List
import time

import requests

from config import LLM_CONFIG, LOGGING_CONFIG

# Setup logging
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['level']),
                   format=LOGGING_CONFIG['format'])
logger = logging.getLogger(__name__)


class LMStudioClient:
    """Client for LM Studio API (OpenAI-compatible endpoint)"""
    
    def __init__(self, url: str = None, model_name: str = None):
        """
        Initialize LM Studio client
        
        Args:
            url: LM Studio API endpoint URL
            model_name: Model name to use
        """
        self.url = url or LLM_CONFIG['lm_studio_url']
        self.model_name = model_name or LLM_CONFIG['model_name']
        self.temperature = LLM_CONFIG['temperature']
        self.max_tokens = LLM_CONFIG['max_tokens']
        self.timeout = LLM_CONFIG['timeout']
        
        logger.info(f"Initializing LM Studio client: {self.url}")
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """
        Verify connection to LM Studio
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(
                f"{self.url.replace('/v1', '')}/models",
                timeout=5
            )
            if response.status_code == 200:
                logger.info("Connected to LM Studio successfully")
                return True
        except Exception as e:
            logger.warning(f"Failed to verify LM Studio connection: {e}")
        
        return False
    
    def generate_sentiment_explanation(self, query: str, context: str = None,
                                      sentiment_label: str = None) -> str:
        """
        Generate sentiment explanation using LLM
        
        Args:
            query: Original query/comment
            context: Retrieved context from RAG
            sentiment_label: Predicted sentiment label
            
        Returns:
            LLM-generated explanation
        """
        
        prompt = self._build_sentiment_prompt(
            query, context, sentiment_label
        )
        
        try:
            response = self._call_llm(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Error generating explanation."
    
    def _build_sentiment_prompt(self, query: str, context: str = None,
                               sentiment_label: str = None) -> str:
        """
        Build prompt for sentiment analysis
        
        Args:
            query: Original query
            context: Retrieved context
            sentiment_label: Predicted sentiment
            
        Returns:
            Formatted prompt
        """
        
        system_prompt = """You are an expert sentiment analysis assistant. 
Your task is to analyze Vietnamese text and provide clear, concise explanations of sentiment and emotions.
Focus on:
1. The primary emotion expressed
2. Key emotional words or phrases
3. Context and nuance
4. Recommendations for response"""
        
        sentiment_instruction = ""
        if sentiment_label:
            sentiment_instruction = f"\nPredicted Sentiment: {sentiment_label}\nPlease confirm or refine this assessment."
        
        context_section = ""
        if context:
            context_section = f"\n\nRelated Context:\n{context}"
        
        prompt = f"""{system_prompt}

Query/Comment:
{query}{sentiment_instruction}{context_section}

Please provide:
1. Sentiment Analysis: What emotion(s) are expressed?
2. Key Phrases: Which words/phrases convey sentiment?
3. Context: Any important nuances or background?
4. Response Suggestion: How should this be handled?

Analysis:"""
        
        return prompt
    
    def _call_llm(self, prompt: str, temperature: float = None,
                 max_tokens: int = None, top_p: float = None,
                 top_k: int = None, retries: int = 3) -> str:
        """
        Call LM Studio API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Max tokens in response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            retries: Number of retry attempts
            
        Returns:
            Generated text response
        """
        
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        top_p = top_p or LLM_CONFIG['top_p']
        top_k = top_k or LLM_CONFIG['top_k']
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.url}/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        text = data['choices'][0].get('text', '').strip()
                        logger.debug(f"LLM response: {text[:100]}...")
                        return text
                else:
                    logger.error(f"LLM API error: {response.status_code}")
            
            except requests.Timeout:
                logger.warning(f"LLM request timeout (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"LLM API error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error("Failed to get response from LLM after retries")
        return "Unable to generate response"
    
    def generate_emotion_keywords(self, text: str) -> List[str]:
        """
        Extract emotion keywords from text
        
        Args:
            text: Input text
            
        Returns:
            List of emotion keywords
        """
        
        prompt = f"""Analyze the following Vietnamese text and extract the key emotion/sentiment words or phrases that convey feeling:

Text: {text}

List only the emotion/sentiment words/phrases, one per line:"""
        
        try:
            response = self._call_llm(prompt, temperature=0.3, max_tokens=100)
            keywords = [line.strip() for line in response.split('\n') if line.strip()]
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def generate_context_summary(self, documents: List[str]) -> str:
        """
        Generate summary of context documents
        
        Args:
            documents: List of context documents
            
        Returns:
            Summary of context
        """
        
        doc_text = "\n".join([f"- {doc}" for doc in documents[:5]])  # Use top 5
        
        prompt = f"""Summarize the following sentiment examples to understand common themes and patterns:

{doc_text}

Summary of themes and patterns:"""
        
        try:
            response = self._call_llm(prompt, temperature=0.5, max_tokens=200)
            return response
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary"
    
    def compare_sentiments(self, text1: str, text2: str) -> Dict:
        """
        Compare sentiment between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison analysis
        """
        
        prompt = f"""Compare the sentiment and emotion in these two Vietnamese texts:

Text 1: {text1}

Text 2: {text2}

Provide:
1. Sentiment of Text 1
2. Sentiment of Text 2
3. Key differences
4. Similarities

Comparison:"""
        
        try:
            response = self._call_llm(prompt, max_tokens=300)
            return {'comparison': response}
        except Exception as e:
            logger.error(f"Error comparing sentiments: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        try:
            response = requests.get(
                f"{self.url}/models",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
        
        return {'model': self.model_name, 'status': 'unknown'}


# Global LLM client instance
_llm_client = None


def get_llm_client(url: str = None, model_name: str = None) -> LMStudioClient:
    """
    Get or create LLM client singleton
    
    Args:
        url: LM Studio API endpoint
        model_name: Model name
        
    Returns:
        LMStudioClient instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LMStudioClient(url, model_name)
    return _llm_client
