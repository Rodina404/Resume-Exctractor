LABEL_LIST = [
    "O",
    "B-SKILL", "I-SKILL",
    "B-JOB_TITLE", "I-JOB_TITLE",
    "B-COMPANY", "I-COMPANY",
    "B-DEGREE", "I-DEGREE",
    "B-UNIVERSITY", "I-UNIVERSITY",
    "B-DATE", "I-DATE",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}