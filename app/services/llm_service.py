from typing import List, Dict, Any
import requests
from config import Config


class LLMService:
    """
    Interface to the local llama.cpp server for text generation.

    Assembles grounded prompts from retrieved context and generates
    cited answers via the /completion endpoint.
    """

    SYSTEM_PROMPT = (
        "You are a helpful document assistant. Answer the user's question "
        "based ONLY on the provided context passages. "
        "Each context passage includes the LOCAL DOCUMENT it came from, and if found online, an INTERNET SOURCE URL. "
        "When answering, use the reference numbers (e.g., [1]) to cite passages. "
        "Clearly differentiate between the internal document name and the external internet URL. "
        "NEVER invent or hallucinate URLs or titles that are not explicitly provided in the context."
    )

    def __init__(self):
        self.base_url = Config.LLM_URL

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a grounded answer using the LLM.

        Args:
            question: The user's question.
            context_chunks: Retrieved chunks with text, filename, score, and web_source.

        Returns:
            The generated answer text.
        """
        if not context_chunks:
            return (
                "No relevant documents were found for your question. "
                "Please upload documents first and try again."
            )

        prompt = self._build_prompt(question, context_chunks)

        try:
            response = requests.post(
                f"{self.base_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 512,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stop": ["</s>", "\n\nQuestion:", "\n\nContext:"],
                },
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("content", "").strip()
        except requests.exceptions.Timeout:
            return "The LLM took too long to respond. Please try a shorter question."
        except requests.exceptions.RequestException as e:
            return f"Error communicating with the LLM service: {str(e)}"

    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        """Build the full prompt with system instructions, context, web sources, and question."""
        # Format context passages
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("filename", "unknown")
            text = chunk.get("text", "")
            web = chunk.get("web_source", "")
            if web:
                context_parts.append(f"[{i}] LOCAL DOCUMENT: {source}\nINTERNET SOURCE: {web}\nTEXT EXCERPT:\n{text}")
            else:
                context_parts.append(f"[{i}] LOCAL DOCUMENT: {source}\nINTERNET SOURCE: None found\nTEXT EXCERPT:\n{text}")

        context_str = "\n\n".join(context_parts)

        # Assemble the full prompt (Mistral instruct format)
        prompt = (
            f"[INST] {self.SYSTEM_PROMPT}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer: [/INST]\n"
        )

        return prompt

    def is_healthy(self) -> bool:
        """Check if the LLM server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
