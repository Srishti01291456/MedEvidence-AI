
import os
import re

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"


# ==========================================
# GROQ CLIENT
# ==========================================

def get_groq_client():

    api_key = os.getenv("GROQ_API_KEY")

    # Streamlit Cloud secret support
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Check your local .env file or Streamlit Cloud secrets."
        )

    return Groq(api_key=api_key)


# ==========================================
# PREPARE MEDICAL EVIDENCE
# ==========================================

def prepare_context(retrieved_chunks):

    if retrieved_chunks is None:
        return "", []

    if hasattr(retrieved_chunks, "to_dict"):
        chunks = retrieved_chunks.to_dict("records")
    else:
        chunks = retrieved_chunks

    context = []
    sources = []

    evidence_number = 1

    for chunk in chunks:

        answer = chunk.get("answer_chunk", "")

        if not isinstance(answer, str) or not answer.strip():
            continue

        source_name = chunk.get("source_name", "Unknown")
        question = chunk.get("question", "Not available")
        chunk_id = chunk.get("chunk_id", "Unknown")
        source_url = chunk.get("source_url", "")

        evidence_text = f"""
[EVIDENCE {evidence_number}]

Source: {source_name}

Related Question: {question}

Medical Information:
{answer}
"""

        context.append(evidence_text)

        sources.append({
            "evidence_number": evidence_number,
            "chunk_id": chunk_id,
            "source_name": source_name,
            "source_url": source_url
        })

        evidence_number += 1

    return "\n".join(context), sources


# ==========================================
# CITATION VALIDATION
# ==========================================

def validate_citations(answer, sources):

    if not answer or not sources:
        return answer

    valid_numbers = {
        str(source["evidence_number"])
        for source in sources
    }

    cited_numbers = re.findall(r"\[(\d+)\]", answer)

    invalid_numbers = [
        number
        for number in cited_numbers
        if number not in valid_numbers
    ]

    for number in invalid_numbers:

        answer = re.sub(
            rf"\[{re.escape(number)}\]",
            "",
            answer
        )

    if invalid_numbers:
        print("Warning: Invalid citations were removed.")
    else:
        print("Citation number validation successful!")

    return answer


# ==========================================
# GENERATE MEDICAL ANSWER
# ==========================================

def generate_answer(user_query, retrieved_chunks):

    print("\nSending evidence to Groq...\n")

    if retrieved_chunks is None or len(retrieved_chunks) == 0:

        return {
            "answer": (
                "I could not find sufficiently relevant medical "
                "information to answer your question. "
                "Please consult a qualified healthcare professional."
            ),
            "sources": []
        }

    # Prepare evidence
    context, sources = prepare_context(retrieved_chunks)

    if not context.strip() or not sources:

        return {
            "answer": (
                "I could not find usable medical evidence "
                "to answer your question."
            ),
            "sources": []
        }

    # ======================================
    # SYSTEM PROMPT
    # ======================================

    system_prompt = """
You are MedEvidence-AI, an evidence-based medical information assistant.

Your task is to answer medical questions using ONLY the medical
evidence provided in the context.

IMPORTANT INSTRUCTIONS:

1. Use only the supplied evidence to answer the user's question.

2. Never invent medical facts, symptoms, causes, treatments,
   statistics, or recommendations.

3. If the supplied evidence does not contain enough information,
   clearly state that the available evidence is insufficient.

4. Explain medical information in simple, clear language.

5. Use headings and bullet points when they improve readability.

6. Cite medical claims using the evidence numbers provided.
   For example: [1] or [2].

7. Only use citation numbers that actually exist in the supplied
   evidence.

8. Place citations immediately after the claims they support.

9. Never cite a source for a claim that the source does not support.

10. Do not confuse symptoms, physical examination findings,
    diagnostic findings, and complications.

11. Never mix subtype-specific symptoms, complications, or
    findings with general condition information.

12. If evidence describes a specific subtype, explicitly name
    that subtype before presenting its information.

13. Do not assume that a feature reported for one subtype
    applies to every person with the general condition.

14. Never state that a symptom occurs in all forms or all patients
    unless the supplied evidence explicitly confirms this.

15. Use phrases such as "may include", "can occur", and
    "reported symptoms include" when describing variable symptoms.

16. When presenting subtype-specific information:
    - State the exact subtype.
    - Keep its symptoms and complications separate from
      general information about the condition.
    - Do not attribute subtype-specific findings to all forms
      of the condition.

17. Distinguish between:
    - General symptoms
    - Physical examination findings
    - Subtype-specific findings
    - Possible long-term complications

18. Do not present a possible complication as if it were
    a symptom experienced by every patient.

19. Do not diagnose the user or claim that the user has a
    particular medical condition.

20. Do not claim that your answer replaces professional
    medical advice.

21. If the question describes a potentially serious or
    emergency situation, advise the user to seek appropriate
    urgent medical care.

22. If evidence is available only for a particular subtype,
    clearly mention that limitation instead of generalizing
    the information to the entire condition.

23. Do not include information from unrelated evidence chunks.

24. If multiple sources provide overlapping information,
    combine it without unnecessarily repeating the same points.

25. Be accurate and cautious. Do not make unsupported
    conclusions merely to provide a longer answer.

26. Keep the answer concise but complete. Do not end
    mid-sentence or leave an unfinished list.

27. Never claim that citation-number validation proves
    that a source semantically supports a medical claim.

MEDICAL EVIDENCE:
"""

    # ======================================
    # USER PROMPT
    # ======================================

    user_prompt = f"""
Medical Question:
{user_query}

Retrieved Medical Evidence:
{context}

Instructions:
Answer the medical question using the supplied evidence.

Keep general condition information separate from subtype-specific
information.

Use accurate citations after the relevant statements.

If the evidence is insufficient, clearly mention this limitation.

Provide a clear, structured and concise answer.
"""

    try:

        # Create Groq client
        client = get_groq_client()

        # Generate response
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=1000
        )

        answer = completion.choices[0].message.content

        if not answer:
            raise ValueError("Groq returned an empty answer.")

        # Validate citation numbers
        answer = validate_citations(answer, sources)

        # Identify cited sources
        cited_numbers = set(
            re.findall(r"\[(\d+)\]", answer)
        )

        used_sources = [
            source
            for source in sources
            if str(source["evidence_number"]) in cited_numbers
        ]

        # Safe fallback
        if not used_sources:

            return {
                "answer": (
                    "I could not generate an answer with valid "
                    "evidence citations. Please consult a qualified "
                    "healthcare professional for reliable guidance."
                ),
                "sources": []
            }

        return {
            "answer": answer,
            "sources": used_sources
        }

    except Exception as error:

        print("\nError while generating answer:", str(error))

        return {
            "answer": (
                "Sorry, I encountered an error while generating "
                "your medical answer. Please try again."
            ),
            "sources": []
        }
