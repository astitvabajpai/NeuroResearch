"""Test what Qwen actually returns for each agent prompt."""
import requests

token = open('.env').read().split('HF_API_TOKEN=')[1].split()[0].strip()
model = 'Qwen/Qwen2.5-7B-Instruct'

def call(messages):
    for provider in ['featherless-ai', 'nebius', 'novita', 'together', 'sambanova']:
        url = f'https://router.huggingface.co/{provider}/v1/chat/completions'
        r = requests.post(url,
            headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'},
            json={'model': model, 'messages': messages, 'max_tokens': 512, 'temperature': 0.7},
            timeout=60)
        if r.ok:
            return r.json()['choices'][0]['message']['content']
        if 'not supported' not in r.text:
            return f'ERROR {r.status_code}: {r.text[:200]}'
    return 'All providers failed'

print('=== Research Agent prompt ===')
research_prompt = """You are a Research Analyst. Analyze search results and extract key findings.

Topic: Impact of AI on healthcare
Search Results:
AI is being used in medical imaging to detect cancer earlier. Studies show 94% accuracy.
Machine learning models predict patient readmission with 80% accuracy.
Natural language processing helps doctors with clinical notes.

Extract 3-5 key research findings or insights from the search results.
Format as bullet points."""

result = call([{'role': 'user', 'content': research_prompt}])
print(result[:500])
print()

print('=== Writer Agent prompt ===')
writer_prompt = """You are a Technical Writer. Write a clear, well-structured research report.

Topic: Impact of AI on healthcare
Research Notes:
- AI detects cancer in medical imaging with 94% accuracy
- ML predicts patient readmission with 80% accuracy
- NLP assists with clinical documentation

Write a comprehensive draft with:
- A clear Title
- Introduction
- Key Findings (numbered list)
- Analysis
- Conclusion

Use markdown formatting with ## headings."""

result = call([{'role': 'user', 'content': writer_prompt}])
print(result[:800])
print()

print('=== Critique Agent prompt ===')
critique_prompt = """You are a Quality Critic. Evaluate the research draft strictly.

Topic: Impact of AI on healthcare
Draft:
## Impact of AI on Healthcare

## Introduction
AI is transforming healthcare through improved diagnostics and efficiency.

## Key Findings
1. Medical imaging AI achieves 94% accuracy in cancer detection
2. ML models predict readmission with 80% accuracy

## Conclusion
AI shows significant promise in healthcare applications.

Rate on Accuracy, Completeness, and Clarity.
Respond EXACTLY in this format:
SCORE: <float between 0.0 and 1.0>
FEEDBACK: <specific improvements needed>"""

result = call([{'role': 'user', 'content': critique_prompt}])
print(result[:400])
