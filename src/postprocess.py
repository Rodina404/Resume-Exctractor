from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class EducationItem:
    degree: str = ""
    university: str = ""
    date: str = ""

@dataclass
class ExperienceItem:
    job_title: str = ""
    company: str = ""
    date: str = ""

def merge_entities(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = []
    current = None

    for p in preds:
        label = p["entity_group"]
        word = p["word"].replace("##", "")
        start = p["start"]
        end = p["end"]

        if current and current["label"] == label and start <= current["end"] + 1:
            sep = "" if word.startswith("'") else " "
            current["text"] += sep + word
            current["end"] = end
            current["score"] = min(current["score"], float(p["score"]))
        else:
            if current:
                merged.append(current)
            current = {
                "label": label,
                "text": word,
                "start": start,
                "end": end,
                "score": float(p["score"]),
            }

    if current:
        merged.append(current)

    return merged

def to_final_output(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    skills = []
    education = []
    experience = []

    current_edu = EducationItem()
    current_exp = ExperienceItem()

    for ent in entities:
        label = ent["label"]
        text = ent["text"].strip()

        if label == "SKILL":
            if text not in skills:
                skills.append(text)

        elif label == "DEGREE":
            if current_edu.degree or current_edu.university or current_edu.date:
                education.append(asdict(current_edu))
                current_edu = EducationItem()
            current_edu.degree = text

        elif label == "UNIVERSITY":
            current_edu.university = text

        elif label == "JOB_TITLE":
            if current_exp.job_title or current_exp.company or current_exp.date:
                experience.append(asdict(current_exp))
                current_exp = ExperienceItem()
            current_exp.job_title = text

        elif label == "COMPANY":
            current_exp.company = text

        elif label == "DATE":
            if current_exp.job_title or current_exp.company:
                current_exp.date = text
            elif current_edu.degree or current_edu.university:
                current_edu.date = text

    if current_edu.degree or current_edu.university or current_edu.date:
        education.append(asdict(current_edu))

    if current_exp.job_title or current_exp.company or current_exp.date:
        experience.append(asdict(current_exp))

    return {
        "Skills": skills,
        "experience": experience,
        "education": education,
    }