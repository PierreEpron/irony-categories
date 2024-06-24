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
    matches = text_pattern.findall(answer)
    return len(matches) == 1 and len(matches[0]) >= min_size

def parse_text_answer(answer):
    matches = text_pattern.findall(answer)
    return '' if len(matches) == 0 else matches[0]


def validate_json_answer(answer):
    matches = json_pattern.findall(answer)
    if len(matches) > 0:
        try:
            json.loads(matches[0].strip())
            return True
        except:
            return False

def parse_json_answer(answer):
    matches = json_pattern.findall(answer)
    if len(matches) > 0:
        try:
            return json.loads(matches[0].strip())
        except:
            return {}


VALID_MAP = {
    'text': validate_text_answer,
    'json': validate_json_answer,
}

PARSER_MAP = {
    'text': parse_text_answer,
    'json': parse_json_answer,
}