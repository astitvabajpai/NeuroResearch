"""
Eval dataset — 25 fixed benchmark topics.

Each fixture specifies:
  - topic          : the research query
  - difficulty     : "easy" | "medium" | "hard" | "adversarial"
  - gold_facts     : key facts the final report must mention (case-insensitive)
  - gold_sources   : expected source domains or keywords in retrieved content
  - must_contain   : exact substrings required in the draft
  - required_sections: ## headings expected
  - min_words      : minimum acceptable draft length
  - min_score      : minimum critique score
  - deep_research  : whether to run in deep mode
  - description    : human-readable note about what this fixture tests
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalFixture:
    topic: str
    difficulty: str = "medium"          # easy | medium | hard | adversarial
    deep_research: bool = False
    min_words: int = 400
    must_contain: List[str] = field(default_factory=list)
    required_sections: List[str] = field(default_factory=list)
    gold_facts: List[str] = field(default_factory=list)
    gold_sources: List[str] = field(default_factory=list)   # domain keywords
    min_score: float = 0.5
    description: str = ""


EVAL_DATASET: List[EvalFixture] = [

    # ── Easy: well-known, lots of data ───────────────────────────────────────

    EvalFixture(
        topic="Transformer architecture in deep learning",
        difficulty="easy",
        min_words=400,
        must_contain=["attention", "transformer"],
        required_sections=["introduction", "key findings", "conclusion"],
        gold_facts=["self-attention", "encoder", "decoder", "BERT", "GPT"],
        gold_sources=["arxiv", "attention is all you need", "vaswani"],
        min_score=0.5,
        description="Foundational ML topic — tests that core architectural facts appear.",
    ),

    EvalFixture(
        topic="CRISPR-Cas9 gene editing mechanism and applications",
        difficulty="easy",
        min_words=400,
        must_contain=["crispr", "gene"],
        required_sections=["introduction", "key findings", "conclusion"],
        gold_facts=["guide RNA", "Cas9 protein", "double-strand break", "Jennifer Doudna"],
        gold_sources=["nature", "science", "cell"],
        min_score=0.5,
        description="Bio topic with clear factual grounding.",
    ),

    EvalFixture(
        topic="How does gradient descent work in neural networks",
        difficulty="easy",
        min_words=400,
        must_contain=["gradient", "learning rate", "loss"],
        required_sections=["introduction", "key findings"],
        gold_facts=["backpropagation", "stochastic gradient descent", "local minimum"],
        gold_sources=["arxiv", "deeplearning"],
        min_score=0.5,
        description="Core ML concept — tests explanatory quality.",
    ),

    EvalFixture(
        topic="Large language model fine-tuning techniques",
        difficulty="easy",
        min_words=400,
        must_contain=["fine-tun"],
        required_sections=["introduction", "key findings"],
        gold_facts=["LoRA", "PEFT", "instruction tuning", "RLHF"],
        gold_sources=["arxiv", "huggingface"],
        min_score=0.5,
        description="Tests LoRA/PEFT coverage — popular topic with rich arxiv data.",
    ),

    EvalFixture(
        topic="Retrieval-Augmented Generation (RAG) for question answering",
        difficulty="easy",
        min_words=400,
        must_contain=["retrieval", "generation"],
        required_sections=["introduction", "key findings", "conclusion"],
        gold_facts=["dense retrieval", "FAISS", "vector store", "hallucination reduction"],
        gold_sources=["arxiv", "lewis", "facebook"],
        min_score=0.5,
        description="Tests RAG understanding — directly relevant to this project.",
    ),

    # ── Medium: requires synthesis across multiple sources ────────────────────

    EvalFixture(
        topic="Climate change mitigation strategies and carbon capture technology",
        difficulty="medium",
        min_words=400,
        must_contain=["carbon", "emission"],
        required_sections=["introduction", "key findings", "conclusion"],
        gold_facts=["net zero", "direct air capture", "Paris Agreement", "renewable energy"],
        gold_sources=["ipcc", "nature", "science"],
        min_score=0.5,
        description="Multi-domain topic requiring synthesis of policy and technology.",
    ),

    EvalFixture(
        topic="Protein structure prediction using AlphaFold",
        difficulty="medium",
        min_words=400,
        must_contain=["protein", "alphafold"],
        required_sections=["introduction", "key findings", "conclusion"],
        gold_facts=["DeepMind", "amino acid", "3D structure", "CASP competition"],
        gold_sources=["nature", "deepmind", "alphafold"],
        min_score=0.5,
        description="Tests that the agent finds the AlphaFold breakthrough specifically.",
    ),

    EvalFixture(
        topic="Federated learning for privacy-preserving machine learning",
        difficulty="medium",
        min_words=400,
        must_contain=["federated", "privacy"],
        required_sections=["introduction", "key findings"],
        gold_facts=["local training", "model aggregation", "differential privacy", "Google"],
        gold_sources=["arxiv", "google", "mcmahan"],
        min_score=0.5,
        description="Niche ML topic — tests retrieval depth.",
    ),

    EvalFixture(
        topic="Multimodal AI models combining vision and language",
        difficulty="medium",
        min_words=400,
        must_contain=["vision", "language"],
        required_sections=["introduction", "key findings"],
        gold_facts=["CLIP", "GPT-4V", "image captioning", "cross-modal"],
        gold_sources=["arxiv", "openai", "radford"],
        min_score=0.5,
        description="Tests multi-modal reasoning coverage.",
    ),

    EvalFixture(
        topic="Reinforcement learning from human feedback (RLHF)",
        difficulty="medium",
        min_words=400,
        must_contain=["reinforcement", "human feedback"],
        required_sections=["introduction", "key findings"],
        gold_facts=["reward model", "PPO", "InstructGPT", "Constitutional AI"],
        gold_sources=["arxiv", "openai", "anthropic"],
        min_score=0.5,
        description="Key alignment technique — tests awareness of InstructGPT/PPO.",
    ),

    EvalFixture(
        topic="Quantum computing applications in drug discovery",
        difficulty="medium",
        min_words=400,
        must_contain=["quantum", "drug"],
        required_sections=["introduction", "key findings", "conclusion"],
        gold_facts=["molecular simulation", "variational quantum eigensolver", "IBM", "qubit"],
        gold_sources=["nature", "arxiv", "ibm"],
        min_score=0.4,
        description="Cross-domain topic — quantum + biology.",
    ),

    EvalFixture(
        topic="Graph neural networks for molecular property prediction",
        difficulty="medium",
        min_words=400,
        must_contain=["graph", "molecular"],
        required_sections=["introduction", "key findings"],
        gold_facts=["message passing", "node embedding", "ChemBERTa", "drug candidate"],
        gold_sources=["arxiv", "nature chemistry"],
        min_score=0.4,
        description="Specialized GNN application domain.",
    ),

    # ── Hard: deep research mode, complex synthesis ───────────────────────────

    EvalFixture(
        topic="Quantum computing and post-quantum cryptography standards",
        difficulty="hard",
        deep_research=True,
        min_words=800,
        must_contain=["quantum", "encrypt"],
        required_sections=[
            "executive summary", "introduction", "key findings",
            "challenges", "future directions", "conclusion",
        ],
        gold_facts=["Shor's algorithm", "NIST PQC", "lattice-based", "CRYSTALS-Kyber"],
        gold_sources=["nist", "arxiv", "pqcrypto"],
        min_score=0.5,
        description="Deep mode — tests 9-section structure, length, and technical depth.",
    ),

    EvalFixture(
        topic="Mechanistic interpretability of large language models",
        difficulty="hard",
        deep_research=True,
        min_words=800,
        must_contain=["interpretability", "circuit"],
        required_sections=[
            "executive summary", "introduction", "key findings",
            "technical deep dive", "challenges", "conclusion",
        ],
        gold_facts=["superposition", "polysemanticity", "Anthropic", "activation patching"],
        gold_sources=["arxiv", "anthropic", "elhage"],
        min_score=0.4,
        description="Frontier research area — tests if agent retrieves cutting-edge papers.",
    ),

    EvalFixture(
        topic="Mixture of Experts architecture in large language models",
        difficulty="hard",
        deep_research=True,
        min_words=800,
        must_contain=["mixture of experts", "sparse"],
        required_sections=[
            "executive summary", "introduction", "key findings",
            "technical deep dive", "conclusion",
        ],
        gold_facts=["routing", "Mixtral", "Switch Transformer", "gating network"],
        gold_sources=["arxiv", "mistral", "google"],
        min_score=0.4,
        description="Tests deep mode on a specific architectural concept.",
    ),

    EvalFixture(
        topic="AI safety alignment approaches: RLHF, Constitutional AI, and scalable oversight",
        difficulty="hard",
        deep_research=True,
        min_words=800,
        must_contain=["alignment", "safety"],
        required_sections=[
            "executive summary", "introduction", "key findings",
            "challenges", "future directions", "conclusion",
        ],
        gold_facts=["Anthropic", "OpenAI", "reward hacking", "value alignment"],
        gold_sources=["arxiv", "anthropic", "openai"],
        min_score=0.4,
        description="Broad AI safety synthesis — tests comprehensiveness.",
    ),

    # ── Adversarial: ambiguous, controversial, or sparse data topics ──────────

    EvalFixture(
        topic="Does consciousness emerge from information integration in neural systems",
        difficulty="adversarial",
        min_words=400,
        must_contain=["consciousness"],
        required_sections=["introduction", "key findings"],
        gold_facts=["integrated information theory", "Tononi", "hard problem", "qualia"],
        gold_sources=["arxiv", "plos", "neuroscience"],
        min_score=0.3,
        description="Adversarial: speculative/philosophical topic — tests graceful handling.",
    ),

    EvalFixture(
        topic="Economic impact of artificial general intelligence",
        difficulty="adversarial",
        min_words=400,
        must_contain=["artificial general intelligence", "economic"],
        required_sections=["introduction", "key findings"],
        gold_facts=["productivity", "labor displacement", "GDP"],
        gold_sources=["arxiv", "economics", "brookings"],
        min_score=0.3,
        description="Speculative future topic — tests that agent stays grounded.",
    ),

    EvalFixture(
        topic="xyzplexor neural optimization 2024",
        difficulty="adversarial",
        min_words=50,           # very low bar — this topic doesn't exist
        must_contain=[],
        required_sections=[],
        gold_facts=[],
        gold_sources=[],
        min_score=0.0,
        description="Adversarial: nonsense topic — verifies graceful degradation, not crash.",
    ),

    EvalFixture(
        topic="Bias and fairness in machine learning hiring systems",
        difficulty="adversarial",
        min_words=400,
        must_contain=["bias", "fair"],
        required_sections=["introduction", "key findings"],
        gold_facts=["disparate impact", "Amazon recruiting tool", "protected attributes"],
        gold_sources=["arxiv", "acm", "fatconference"],
        min_score=0.3,
        description="Sensitive topic — tests that agent addresses controversy directly.",
    ),

    # ── Application-domain topics ─────────────────────────────────────────────

    EvalFixture(
        topic="LangGraph vs LangChain for building multi-agent AI systems",
        difficulty="medium",
        min_words=400,
        must_contain=["langgraph", "langchain"],
        required_sections=["introduction", "key findings"],
        gold_facts=["stateful graph", "agent loop", "node", "edge"],
        gold_sources=["langchain", "github", "docs"],
        min_score=0.4,
        description="Project-relevant topic — tests framework comparison quality.",
    ),

    EvalFixture(
        topic="Vector databases for semantic search: FAISS vs Pinecone vs Weaviate",
        difficulty="medium",
        min_words=400,
        must_contain=["vector", "embedding"],
        required_sections=["introduction", "key findings"],
        gold_facts=["cosine similarity", "ANN search", "index", "nearest neighbour"],
        gold_sources=["arxiv", "pinecone", "weaviate"],
        min_score=0.4,
        description="Practical tool comparison — tests structured comparison output.",
    ),

    EvalFixture(
        topic="Neural scaling laws: compute, data, and model size relationships",
        difficulty="hard",
        min_words=400,
        must_contain=["scaling", "compute"],
        required_sections=["introduction", "key findings", "conclusion"],
        gold_facts=["Chinchilla", "Kaplan", "power law", "tokens per parameter"],
        gold_sources=["arxiv", "deepmind", "openai"],
        min_score=0.4,
        description="Tests that agent retrieves Chinchilla/Kaplan laws specifically.",
    ),

    EvalFixture(
        topic="Diffusion models for image generation: DALL-E, Stable Diffusion, Midjourney",
        difficulty="medium",
        min_words=400,
        must_contain=["diffusion", "image generation"],
        required_sections=["introduction", "key findings", "conclusion"],
        gold_facts=["denoising", "latent space", "CLIP", "Stable Diffusion"],
        gold_sources=["arxiv", "openai", "stability"],
        min_score=0.4,
        description="Popular generative AI topic — tests factual coverage.",
    ),

    EvalFixture(
        topic="Model context protocol (MCP) for AI tool integration",
        difficulty="medium",
        min_words=400,
        must_contain=["model context protocol", "tool"],
        required_sections=["introduction", "key findings"],
        gold_facts=["Anthropic", "stdio", "SSE transport", "JSON-RPC"],
        gold_sources=["anthropic", "modelcontextprotocol", "github"],
        min_score=0.3,
        description="Meta-topic: tests if agent can research the MCP spec itself.",
    ),
]
