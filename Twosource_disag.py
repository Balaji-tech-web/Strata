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
    evidence_span: str


class FactList(BaseModel):
    observations: list[Fact]


structured_model = model.with_structured_output(FactList)


def get_facts(text):
    prompt = (
        "Extract all facts from this document. For each one give the person's full name "
        "(no titles) as entity_name_guess, the type in snake_case (use phone_number for any "
        "phone or mobile number), the value, and the exact evidence_span from the text.\n\n" + text
    )
    return structured_model.invoke(prompt).observations


def fix_type(t):
    t = t.lower().replace(" ", "_")
    if "phone" in t or "mobile" in t:
        return "phone_number"
    return t


observations = []
n = 1
for doc_id in ["F", "G"]:
    for f in get_facts(docs[doc_id]):
        observations.append({
            "id": doc_id + "_" + str(n),
            "source": doc_id,
            "entity": f.entity_name_guess.strip(),
            "type": fix_type(f.type),
            "value": f.extracted_value,
            "status": "valid",
            "conflicts_with": [],
        })
        n += 1

groups = {}
for obs in observations:
    key = (obs["entity"], obs["type"])
    if key not in groups:
        groups[key] = []
    groups[key].append(obs)

for key in groups:
    group = groups[key]
    values = set()
    for obs in group:
        values.add(obs["value"])
    if len(values) > 1:
        for obs in group:
            obs["status"] = "contradictory"
            for other in group:
                if other["id"] != obs["id"]:
                    obs["conflicts_with"].append(other["id"])


def lookup(entity, field_type):
    return groups.get((entity, fix_type(field_type)), [])


for obs in observations:
    print(obs["id"], "|", obs["source"], "|", obs["entity"], "|", obs["type"], "|", obs["value"], "|", obs["status"])

print("\nlookup Raj Malhotra phone_number:")
result = lookup("Raj Malhotra", "phone_number")
sources = set()
all_conflict = True
for obs in result:
    print(obs["value"], obs["source"], obs["status"], obs["conflicts_with"])
    sources.add(obs["source"])
    if obs["status"] != "contradictory":
        all_conflict = False

if sources == {"F", "G"} and all_conflict:
    print("\nDONE")
else:
    print("\nNOT DONE - check names and types above")