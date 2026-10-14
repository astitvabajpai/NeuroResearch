"""
Writer Agent — synthesises research notes into a structured, cited report.

Prompt design principles:
- Role framing sets the quality bar: "experienced science journalist / analyst"
- Critique feedback is actionable: the model is told exactly how to use it
- Citation rules are explicit and repeated at point-of-use, not buried at the end
- Tone guidance prevents both dry bullet-dumps and hollow filler phrases
- Each section has a concrete description of what "good" looks like
- Iteration context: on revision, the model is reminded what changed and why
"""
import logging
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base_agent import BaseAgent
from src.state.research_state import ResearchState

logger = logging.getLogger(__name__)

MAX_NOTE_ITERATIONS = 2

STANDARD_PROMPT = ChatPromptTemplate.from_template("""\
You are an experienced analyst and science writer. Your task is to write a clear, \
well-structured research report for an educated but non-specialist audience.

TOPIC: {topic}

RESEARCH NOTES (sourced from live web searches — treat these as your only factual source):
{notes}

CRITIQUE TO ADDRESS IN THIS DRAFT:
{feedback}

If the critique is "First draft", write the best possible initial report.
If it lists specific problems, fix every one of them before anything else.

---

Write the report using these exact section headings in this order:

## Introduction
Write three paragraphs:
- Para 1: Define the topic precisely. What is it, and what problem does it solve or represent?
- Para 2: Why does it matter? Give concrete stakes — industries affected, scale of impact, or \
  why researchers care.
- Para 3: Scope this report — what will the reader learn from it?

Do NOT use filler phrases like "In today's rapidly evolving landscape" or \
"It is important to note". Start with a direct statement of fact.

## Key Findings
Write a numbered list of 5–7 findings.

Each finding must:
- Be 2–3 sentences long
- State a specific fact, statistic, mechanism, or example from the research notes
- End with an inline citation: [Source Title or Domain](url)
  → Use only URLs that appear in the research notes. If a note says "Source: unavailable", \
    do not invent a URL — omit the citation link and write (source unavailable).

Example of a well-written finding:
1. Transformer models process all tokens in parallel using self-attention, enabling training \
on much larger datasets than sequential RNNs. The original architecture, introduced in \
"Attention Is All You Need" (2017), achieved state-of-the-art results on translation tasks \
with significantly less training time. [Vaswani et al.](https://arxiv.org/abs/1706.03762)

## Analysis
Write three paragraphs that go beyond the findings:
- Para 1: What patterns connect the findings? What does the evidence collectively suggest?
- Para 2: What are the practical implications — for practitioners, researchers, or the \
  field at large?
- Para 3: What tensions or contradictions exist in the evidence? What remains unresolved?

Cite sources inline where specific claims come from the research notes.

## Conclusion
Write two paragraphs:
- Para 1: Synthesise the most important takeaways in 3–4 sentences. Be specific — name the \
  key mechanisms, numbers, or actors that matter most.
- Para 2: What should a reader do or think differently as a result of this report? \
  What open questions remain?

---

WRITING RULES:
- Minimum 500 words total
- Every Key Finding must have a citation
- Do not fabricate URLs or statistics not present in the research notes
- Write in third person, present tense for ongoing situations, past tense for completed events
- Avoid hedging phrases like "it seems", "it could be argued", "many experts believe"
- Be direct and specific at all times
""")

DEEP_PROMPT = ChatPromptTemplate.from_template("""\
You are a senior research analyst writing a comprehensive deep-dive report. Your audience is \
technically literate — researchers, engineers, or senior decision-makers. \
The report must be rigorous, specific, and citation-backed throughout.

TOPIC: {topic}

RESEARCH NOTES (sourced from live web searches — treat these as your only factual source):
{notes}

CRITIQUE TO ADDRESS IN THIS DRAFT:
{feedback}

If the critique is "First draft", write the best possible initial report.
If it lists specific problems, fix every one of them before anything else. \
Do not just reword sections — address the root cause of each critique point.

---

Write the report using these exact section headings in this order:

## Executive Summary
Three sentences only:
- Sentence 1: What is the topic and why is it significant right now?
- Sentence 2: What are the 2–3 most important findings?
- Sentence 3: What is the bottom-line takeaway or implication?

## Introduction
Four paragraphs:
- Para 1: Precise definition and origin of the topic
- Para 2: Why this matters — quantitative scale, who is affected, recent catalysts
- Para 3: Scope of this report — what it covers and what it deliberately excludes
- Para 4: How the report is structured and what the reader will gain

Do NOT open with clichés. Start Para 1 with the definition itself.

## Current State of the Field
Four paragraphs covering where things stand today:
- Para 1: Leading organisations, systems, or approaches with specific names and dates
- Para 2: Key benchmarks, metrics, or performance figures that define the current frontier
- Para 3: How the landscape has shifted in the last 12–24 months
- Para 4: What the mainstream adoption picture looks like

Cite sources inline: [Domain or Author](url)

## Key Findings
A numbered list of 8–10 findings.

Each finding must:
- Be 3–4 sentences
- Open with the most specific fact or claim, not a general statement
- Include at least one concrete detail: name, date, percentage, paper title, company, or metric
- End with an inline citation: [Source](url)
  → Only use URLs from the research notes. Write (source unavailable) if none exists.

## Technical Deep Dive
Three paragraphs that go deep on the hardest or most interesting technical aspects:
- Para 1: The core mechanism or algorithm — how does it actually work?
- Para 2: The key engineering or design tradeoffs involved
- Para 3: What makes this technically hard, and what breakthroughs have addressed it?

Use precise technical language. Cite sources for specific claims.

## Applications & Real-World Impact
Three paragraphs:
- Para 1: The most mature or impactful deployed applications with specific examples
- Para 2: Emerging applications and what enables them
- Para 3: Quantitative impact — cost savings, performance gains, scale of adoption

Name specific companies, products, or deployments. Cite sources.

## Challenges & Limitations
A numbered list of 4–5 challenges.

Each challenge must:
- Name the specific problem (not just "it is challenging")
- Explain why it is hard — what makes it technically or practically difficult?
- Note whether it is an active area of research or a known unsolved problem

## Future Directions
Three paragraphs:
- Para 1: What is the research community actively working on right now?
- Para 2: What will likely be possible in 2–5 years, based on current trajectories?
- Para 3: What paradigm shifts or breakthroughs could change the trajectory entirely?

Be specific. Mention named research programmes, published roadmaps, or announced timelines \
where they appear in the research notes.

## Conclusion
Three paragraphs:
- Para 1: Synthesise the most important findings — what do they collectively mean?
- Para 2: What are the practical implications for someone working in or adjacent to this field?
- Para 3: What are the most important open questions, and why do they matter?

---

WRITING RULES:
- Minimum 900 words total
- Every Key Finding and every factual claim in Current State must have a citation
- Do not fabricate URLs, statistics, or details not present in the research notes
- Write in an authoritative, precise academic-professional register
- Avoid hedging unless the evidence itself is genuinely uncertain
- Never use filler transitions ("Furthermore", "It is worth noting", "In conclusion")
- Be direct, specific, and concrete at all times
""")


class WriterAgent(BaseAgent):
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id
        self.llm      = None

    def _initialize(self, model_id: str | None = None, deep: bool = False):
        target = model_id or self.model_id
        self.model_id = target
        from src.tools.llm import get_llm
        self.llm = get_llm(self.model_id, deep=deep)

    def _get_rag_context(self, session_id: str, topic: str, deep: bool) -> str:
        """
        Query the per-session FAISS index for chunks most relevant to writing
        the report. Returns a formatted context block or empty string if RAG
        is unavailable or the index hasn't been built yet.
        """
        try:
            from src.tools.vector_store import query as vs_query, format_chunks_for_prompt
            # For deep mode, retrieve more chunks for broader coverage
            k = 8 if deep else 5
            chunks = vs_query(session_id, topic, k=k)
            if not chunks:
                return ""
            formatted = format_chunks_for_prompt(chunks)
            return (
                "\n\nADDITIONAL RETRIEVED CONTEXT (use for specific facts and citations):\n"
                f"{formatted}"
            )
        except Exception as exc:
            logger.debug("[WriterAgent] RAG query failed: %s", exc)
            return ""

    def invoke(self, state: ResearchState) -> dict:
        model_id   = state.get("writer_model") or self.model_id
        deep       = state.get("deep_research", False)
        session_id = state.get("session_id", "default")
        self._initialize(model_id, deep=deep)

        all_notes    = state["research_notes"]
        notes_to_use = all_notes[-MAX_NOTE_ITERATIONS:] if len(all_notes) > MAX_NOTE_ITERATIONS else all_notes
        if len(all_notes) > MAX_NOTE_ITERATIONS:
            logger.info("[WriterAgent] Using last %d of %d note batches to avoid context overflow",
                        MAX_NOTE_ITERATIONS, len(all_notes))

        # RAG: augment notes with retrieved chunks from the vector store
        rag_context = self._get_rag_context(session_id, state["topic"], deep)
        notes = "\n\n".join(notes_to_use) + rag_context

        feedback = state.get("critique_feedback", "").strip()
        if not feedback:
            feedback = "First draft — write the best possible report using all research notes."

        prompt    = DEEP_PROMPT if deep else STANDARD_PROMPT
        chain     = prompt | self.llm
        draft     = chain.invoke({"topic": state["topic"], "notes": notes, "feedback": feedback})
        draft_str = draft.content if hasattr(draft, "content") else str(draft)

        if not draft_str.strip():
            logger.warning("[WriterAgent] LLM returned empty draft")
            draft_str = (
                f"## Introduction\n\nResearch report on: {state['topic']}\n\n"
                "(Draft generation failed — please retry)"
            )

        return {"draft": draft_str}
