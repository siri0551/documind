"""
DocuMind - RAG Pipeline Module
==============================
Implements Retrieval-Augmented Generation (RAG) using LangChain and Google Gemini.
Retrieves contextually relevant document chunks from the FAISS vector store,
constructs a grounded prompt template with conversational history, and generates
accurate, cited responses while strictly guarding against hallucinations.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

import config

logger = logging.getLogger(__name__)


# ==============================================================================
# System Prompt & Templates
# ==============================================================================
SYSTEM_PROMPT = """You are DocuMind, an expert, professional, and helpful Generative AI document assistant.
Your goal is to answer the user's questions accurately and concisely based ONLY on the provided context retrieved from the user's uploaded PDF documents.

### Operating Rules:
1. Grounding: Answer the question using ONLY the provided document context. Do not invent facts, speculate, or introduce external information not found in the documents.
2. Context Missing: If the provided context does not contain enough information to answer the question, state politely and clearly:
   "Based on the provided document(s), I could not find information regarding [topic]. Please verify if this is covered in the uploaded PDF."
3. Citations: Where applicable, reference the specific document name and page number provided in the context tags (e.g., "[Doc: report.pdf, Page: 3]").
4. Tone & Style: Maintain an objective, professional, and clear tone. Use Markdown formatting (bullet points, bold highlights, tables, or numbered steps) when helpful for readability.
5. Scope: If greeted or asked about your capabilities, introduce yourself as DocuMind and explain that you can answer questions about the uploaded PDF documents.

### Retrieved Document Context:
{context}
"""


class RAGPipeline:
    """
    Orchestrates the complete RAG process:
    1. Semantic similarity retrieval from FAISS
    2. Context formatting with document source and page citations
    3. Grounded prompt assembly
    4. Google Gemini LLM invocation and source attribution
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: int = config.TOP_K_RESULTS
    ):
        """
        Initialize the RAG pipeline.

        Args:
            api_key (str, optional): Google Gemini API key. Defaults to env var.
            model_name (str, optional): Gemini model name.
            temperature (float, optional): Generation temperature.
            top_k (int): Number of top context chunks to retrieve.
        """
        self.api_key = api_key or config.get_google_api_key()
        self.model_name = model_name or config.DEFAULT_LLM_MODEL
        self.temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        self.top_k = top_k

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Instantiate the Google Gemini Chat model.

        Returns:
            ChatGoogleGenerativeAI: Initialized LangChain LLM instance.

        Raises:
            ValueError: If the Google API key is missing.
        """
        if not self.api_key:
            raise ValueError(
                "Google Gemini API Key is missing. "
                "Please configure GOOGLE_API_KEY in your .env file or enter it in the sidebar."
            )

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            convert_system_message_to_human=True
        )

    @staticmethod
    def format_context_chunks(chunks: List[Document]) -> str:
        """
        Formats retrieved Document chunks into a structured context string
        clearly identifying source documents and page numbers.

        Args:
            chunks (List[Document]): Retrieved documents.

        Returns:
            str: Formatted context string.
        """
        if not chunks:
            return "No relevant context found in documents."

        formatted_parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "Unknown Document")
            page = chunk.metadata.get("page", "N/A")
            formatted_parts.append(
                f"[Source Chunk #{i} | File: {source} | Page: {page}]\n{chunk.page_content}"
            )

        return "\n\n---\n\n".join(formatted_parts)

    def retrieve_relevant_chunks(
        self,
        query: str,
        vector_store: FAISS
    ) -> List[Document]:
        """
        Performs semantic similarity search on the FAISS vector index.

        Args:
            query (str): User question.
            vector_store (FAISS): FAISS index instance.

        Returns:
            List[Document]: Top-K most relevant chunks.
        """
        try:
            return vector_store.similarity_search(query, k=self.top_k)
        except Exception as e:
            logger.error(f"Error during vector retrieval: {e}")
            return []

    def answer_question(
        self,
        query: str,
        vector_store: FAISS,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end RAG pipeline for a given query:
        1. Retrieves relevant chunks.
        2. Formats context with source/page attribution.
        3. Constructs prompt and invokes Gemini LLM.
        4. Extracts clean source references for UI display.

        Args:
            query (str): The user's question.
            vector_store (FAISS): Active vector index.
            chat_history (List[Dict[str, str]], optional): Prior turns formatted as [{'role': ..., 'content': ...}].

        Returns:
            Dict[str, Any]: {
                "answer": str,
                "sources": List[Dict[str, Any]],  # [{'source': ..., 'page': ..., 'snippet': ...}]
                "error": Optional[str]
            }
        """
        # Step 1: Validate query
        query_clean = query.strip()
        if not query_clean:
            return {
                "answer": "Please ask a question about your uploaded document(s).",
                "sources": [],
                "error": None
            }

        # Step 2: Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query_clean, vector_store)
        if not relevant_chunks:
            return {
                "answer": (
                    "I could not find any relevant sections in your uploaded document(s) "
                    "matching this query. Please check your document contents or try rephrasing your question."
                ),
                "sources": [],
                "error": None
            }

        # Step 3: Format context
        context_str = self.format_context_chunks(relevant_chunks)

        # Step 4: Build prompt with conversational context
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Convert recent chat history into LangChain message pairs
        formatted_history = []
        if chat_history:
            # Keep the last 6 messages (3 turns) for context continuity
            for msg in chat_history[-6:]:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    formatted_history.append(("human", content))
                elif role == "assistant":
                    formatted_history.append(("ai", content))

        # Step 5: Invoke LLM Chain
        try:
            llm = self._get_llm()
            chain = prompt_template | llm | StrOutputParser()

            answer = chain.invoke({
                "context": context_str,
                "chat_history": formatted_history,
                "question": query_clean
            })

            # Format source references for UI display
            sources_metadata = []
            seen_sources = set()
            for chunk in relevant_chunks:
                source_name = chunk.metadata.get("source", "Document")
                page_num = chunk.metadata.get("page", 1)
                snippet = chunk.page_content.strip()
                # Create unique key to prevent duplicate citation cards
                citation_key = (source_name, page_num, snippet[:80])
                if citation_key not in seen_sources:
                    seen_sources.add(citation_key)
                    sources_metadata.append({
                        "source": source_name,
                        "page": page_num,
                        "snippet": snippet[:300] + ("..." if len(snippet) > 300 else "")
                    })

            return {
                "answer": answer,
                "sources": sources_metadata,
                "error": None
            }

        except Exception as e:
            err_msg = str(e)
            logger.error(f"RAG generation error: {err_msg}")

            # User-friendly diagnostic messaging
            if "API_KEY_INVALID" in err_msg or "invalid api key" in err_msg.lower():
                user_feedback = (
                    "Authentication Error: The provided Google Gemini API Key is invalid. "
                    "Please verify your API key in Google AI Studio."
                )
            elif "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                user_feedback = (
                    "Rate Limit Warning: Google Gemini API rate limit or quota exceeded. "
                    "Please wait a moment before trying again."
                )
            else:
                user_feedback = f"An error occurred while communicating with Gemini API: {err_msg}"

            return {
                "answer": user_feedback,
                "sources": [],
                "error": err_msg
            }
