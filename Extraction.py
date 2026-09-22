from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from Document import docs

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


class Fact(BaseModel):
    entity_name_guess: str
    type: str
    extracted_value: str
    derivation_method: str
    confidence: float
    evidence_span: str


class FactList(BaseModel):
    observations: list[Fact]


structured_model = model.with_structured_output(FactList)


def clean(s):
    return s.replace("$", "").replace(",", "").strip()


def evidence_ok(span, text):
    return span in text or clean(span) in clean(text)


results = []

for doc_id, text in docs.items():
    prompt = (
        "Extract every fact from this document: dates, amounts, names, IDs, statuses, "
        "phone numbers, errors and priorities. Find at least 2 facts even in short documents. "
        "For evidence_span, copy the shortest exact substring that shows the fact, word for word.\n\n"
        + text
    )
    facts = structured_model.invoke(prompt).observations

    for f in facts:
        ok = evidence_ok(f.evidence_span, text)
        if not ok:
            print(f"{doc_id}: evidence not found -> {f.evidence_span}")
        results.append((doc_id, f.type, f.extracted_value, f.evidence_span, "PASS" if ok else "FAIL"))

print("\nDoc | Type | Value | Evidence | Result")
print("-" * 80)
for doc_id, ftype, value, span, result in results:
    print(f"{doc_id} | {ftype} | {value} | {span} | {result}")