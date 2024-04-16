import re

head_pattern  = re.compile(r"[0-9][\.|\)] ([^:\n\.-]+):", re.I)

def is_valid_enum(text, eos_token='</s>'):
    if len(head_pattern.findall(text)) == 0:
        return False
    if not text.endswith(eos_token):
        return False
    return True

def parse_enum(text):
    previous = None
    outputs = []
    for m in head_pattern.finditer(text):
        if previous:
            outputs.append({
                'header': previous.group(1).strip(),
                'argument': text[previous.end():m.start()].strip()
            })
        previous = m
    return outputs


def parse_enums(examples):
    outputs = []

    for example in examples:
        label = int(example['label_id'] == example['pred'])
        outputs.extend([{**x, 'label':label} for x in parse_enum(example['pred_expls']['enum_exp'])])

    return outputs