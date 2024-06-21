import json
import re

head_pattern = re.compile(r"[0-9][\.|\)] ([^:\n\.-]+):", re.I)
json_pattern = re.compile(r"```(?:json)?(.+)```", re.S + re.I)
text_pattern = re.compile(r"\{[\n\s\t]*\"text\":[\n\s\t]*\"(.+)\"[\n\s\t]*\}")


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
        if is_valid_enum(example['pred_expls']['enum_exp']):
            label = int(example['label_id'] == example['pred']) 
            outputs.extend([{**x, 'label':label} for x in parse_enum(example['pred_expls']['enum_exp'])])

    return outputs

def validate_text_answer(answer, min_size=5):
    text = text_pattern.findall(answer)
    return len(text) == 1 and len(text[0]) >= min_size

def parse_text_answer(answer):
    text = text_pattern.findall(answer)
    return '' if len(text) == 0 else text[0]

VALID_MAP = {
    'text': validate_text_answer,
}

PARSER_MAP = {
    'text': parse_text_answer,
}