import re

head_pattern  = re.compile(r"[0-9][\.|\)] ([^:\n\.-]+):", re.I)

def is_valid_enum(text, eos_token='</s>'):
    if len(head_pattern.findall(text)) == 0:
        return False
    if not text.endswith(eos_token):
        return False
    return True