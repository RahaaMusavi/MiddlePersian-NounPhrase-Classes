import os
import math
import pandas as pd
from collections import Counter
from conllu import parse
from conllu.parser import DEFAULT_FIELD_PARSERS

class MiddlePersianExtractor:
    def __init__(self):
        self.standardized_deprels = {
            "nm": "nmod", "nmod:poss": "nmod", "nmod:det": "nmod",
            "adjmod": "amod", "al:relcl": "acl:relcl", "acl:recl": "acl:relcl"
        }
        self.anchors = {'ān', 'ōy', 'ēn', 'ēd'}
        self.vowels = "aeiouʾywēīōūā"

    def convert_id(self, id_val):
        if isinstance(id_val, tuple):
            return float(f"{id_val[0]}.{id_val[2]}")
        try: return float(id_val)
        except: return None

    def get_np_depth(self, token_id, token_map, depth=0):
        deps = [t for t in token_map.values() if self.convert_id(t['head']) == token_id]
        if not deps or depth > 10: return depth
        return max([self.get_np_depth(self.convert_id(d['id']), token_map, depth + 1) for d in deps])

    def is_verbal(self, token):
        lemma = (token.get('lemma') or "").lower()
        form = (token.get('form') or "").lower()
        feats = token.get('feats') or {}
        return "VerbForm" in feats or any(x in lemma or x in form for x in ["išnīh", "ār", "tag", "dag"])

    def extract_features(self, conllu_path):
        with open(conllu_path, 'r', encoding='utf-8') as f:
            sentences = parse(f.read())
        
        records = []
        for sent in sentences:
            token_map = {self.convert_id(t['id']): t for t in sent if self.convert_id(t['id']) is not None}
            
            for t_id, token in token_map.items():
                if token['upos'] not in {'NOUN', 'PROPN', 'PRON'}: continue
                
                # Identify dependents
                dependents = [d for d in sent if self.convert_id(d['head']) == t_id]
                for dep in dependents:
                    dep_id = self.convert_id(dep['id'])
                    if dep['deprel'] not in ['nmod', 'amod', 'flat', 'appos'] or dep['lemma'] in ['ī', 'ud']:
                        continue

                    # Calculate Distance (ignoring ezafe)
                    between = [x for x in sent if min(t_id, dep_id) < self.convert_id(x['id']) < max(t_id, dep_id) and x['lemma'] != 'ī']
                    if len(between) > 7: continue

                    records.append({
                        'head_upos': token['upos'],
                        'modifier_upos': dep['upos'],
                        'head_is_plural': 1 if (token['feats'] or {}).get('Number') == 'Plur' else 0,
                        'np_depth': self.get_np_depth(t_id, token_map),
                        'num_dependents_head': len(dependents),
                        'num_dependents_modifier': len([x for x in sent if self.convert_id(x['head']) == dep_id]),
                        'distance': len(between),
                        'position': 1 if t_id < dep_id else 2, # 1: Initial, 2: Final
                        'has_anchor_dependent': 1 if any(d['deprel'] == 'det' and d['lemma'] in self.anchors for d in dependents) else 0,
                        'ezafe_label': 1 if any(x['lemma'] == 'ī' and self.convert_id(x['head']) == dep_id for x in sent) else 0,
                        'head_is_verbal': 1 if self.is_verbal(token) else 0,
                        'source_file': os.path.basename(conllu_path)
                    })
        return records