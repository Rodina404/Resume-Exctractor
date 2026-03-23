# Binary skill mapping for isolated model tasks
LABEL_LIST = [
    "O",
    "B-SKILL",
    "I-SKILL"
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
