"""src/rag/policy_assistant.py -- Grounded Policy Assistant (Part D Task 3)

1. Connects PolicyRetriever with mock_llm.chat to answer user questions.
2. Formats prompt with strict CONTEXT blocks and QUESTION section.
3. System message enforces POLICY ASSISTANT grounding rules and fallback phrasing.
"""

from pathlib import Path
import sys

# Ensure project root is available to import mock_llm
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mock_llm
from src.rag.retriever import PolicyRetriever


class PolicyAssistant:
    """Grounded Question-Answering Assistant over Instamart Policy Documents."""

    def __init__(self, retriever: PolicyRetriever):
        self.retriever = retriever

    def format_prompt(self, question: str, retrieved_chunks: list[dict]) -> tuple[str, str]:
        """Construct system and user messages adhering to the required structure."""
        sys_msg = (
            "You are an automated POLICY ASSISTANT for Instamart. "
            "Answer the question using only the supplied context. Do not invent policy details. "
            "If the answer is not supported by the context, respond with: "
            "I could not find that information in the policy documents."
        )

        context_lines = ["CONTEXT:"]
        for ch in retrieved_chunks:
            context_lines.append(f"[{ch['source']}]")
            context_lines.append(ch["text"])
        context_lines.append("QUESTION:")
        context_lines.append(question)

        user_msg = "\n".join(context_lines)
        return sys_msg, user_msg

    def answer(self, question: str, k: int = 3) -> tuple[str, list[str]]:
        """Retrieve top k chunks, format prompt, call mock_llm, and return answer with source filenames."""
        retrieved = self.retriever.retrieve(query=question, k=k)
        sources = [ch["source"] for ch in retrieved]

        sys_msg, user_msg = self.format_prompt(question, retrieved)
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        resp = mock_llm.chat(messages)
        answer_text = resp.choices[0].message.content
        return answer_text, sources

