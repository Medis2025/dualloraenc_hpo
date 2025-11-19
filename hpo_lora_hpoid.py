#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPO Token-level NER with LoRA + Ontology Refinement (BIOES)
Two-head training (no loss mixing): NER (token/CRF) and HPO-ID head with Cross-Attention.
- Per-dataset holdout split + external val
- Separate validation: NER token/span F1 and HPO acc/topK
- tqdm progress bars
- TensorBoard logging

Fixes in this revision:
- init_encoder_from accepts HF dir OR .pt state_dict; loads only encoder weights (strict=False)
- Gold HPO ID is always kept in candidate lists even after refine_id_set (prevents ValueError)
- AMP API updated to torch.amp (no deprecation warnings)
"""

from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
#                      "expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:128")

import re
import json
import math
import time
import glob
import random
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Set
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from torchcrf import CRF
    HAS_CRF = True
except Exception:
    HAS_CRF = False

from torch.amp import GradScaler, autocast

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("HPO-LoRA-NER-Onto-Min")

SPAN = Tuple[int, int]  # [char_start, char_end]

# =============================================================================
# Config
# =============================================================================
@dataclass
class HPOConfig:
    # REQUIRED
    backbone: str = " /models/BioLinkBERT-base"
    model_dir: str = " /tuned/hpo_lora_onto"
    train_roots: List[str] = (
        "  data/GeneReviews",
        "  data/GSC+",
        "  data/ID-68",
    )
    val_root: str = "  data/val"
    hpo_json: str = "  data/hpo.json"

    # optional encoder init from finetuned model dir or .pt state_dict
    init_encoder_from: Optional[str] = " /tuned/intention"

    # training
    seed: int = 42
    max_len: int = 256
    stride: int = 48
    batch_size: int = 8
    epochs: int = 12
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("query","key","value","dense","intermediate.dense","output.dense")

    # CRF
    use_crf: bool = True  # respected only if torchcrf installed

    # BIOES labels
    label_O: str = "O"
    label_B: str = "B-PHENO"
    label_I: str = "I-PHENO"
    label_E: str = "E-PHENO"
    label_S: str = "S-PHENO"

    # ontology refine toggles
    do_refine_on_val: bool = True
    ancestor_backoff: bool = True
    dedup_parent_when_child: bool = True
    prefer_deeper_when_tie: bool = True

    # holdout split + example count
    per_root_val_ratio: float = 0.1
    per_root_example_limit: int = 3

    # HPO head & fusion
    span_proj_dim: int = 256
    hpo_pred_mix_p: float = 0.2      # probability to use predicted span instead of gold (if IoU >= 0.5)
    hpo_topk: int = 5                # report top-k for validation

    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_dirs(self):
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "config.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

# =============================================================================
# Ontology loader & helpers  (REVISED: Alt_id-aware expansion)
# =============================================================================
class HPOOntology:
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Dict] = json.load(f)

        # Normalize primary nodes to colon form and lists
        norm_data: Dict[str, Dict] = {}
        for k, v in raw.items():
            kk = k.replace("_", ":")
            vv = dict(v)
            vv["Id"] = vv.get("Id", kk).replace("_", ":")
            def _norm_keys(d: Dict) -> Set[str]:
                return set([x.replace("_", ":") for x in (d.keys() if isinstance(d, dict) else [])])
            vv["Son"]    = _norm_keys(vv.get("Son", {}))
            vv["Child"]  = _norm_keys(vv.get("Child", {}))
            vv["Father"] = _norm_keys(vv.get("Father", {}))
            # list-ify fields
            names = vv.get("Name", [])
            vv["Name"] = [str(x) for x in (names if isinstance(names, list) else [names])]
            syns = vv.get("Synonym", [])
            vv["Synonym"] = [str(x) for x in (syns if isinstance(syns, list) else [syns])]
            alts = vv.get("Alt_id", [])
            vv["Alt_id"] = [str(x).replace("_", ":") for x in (alts if isinstance(alts, list) else [alts])]
            norm_data[kk] = vv

        # Build alias -> primary map from Alt_id
        alias2primary: Dict[str, str] = {}
        for pid, v in norm_data.items():
            for alt in v.get("Alt_id", []):
                if alt and alt != pid:
                    alias2primary[alt] = pid

        # Expand aliases into data as cloned nodes so they exist as keys
        expanded_data: Dict[str, Dict] = dict(norm_data)
        for alt, pid in alias2primary.items():
            if alt in expanded_data:
                continue
            base = norm_data.get(pid, {})
            clone = dict(base)
            clone["Id"] = alt
            expanded_data[alt] = clone

        self.data = expanded_data
        self.alias2primary = alias2primary

        # Build parent/child maps from expanded data
        self.children: Dict[str, Set[str]] = {hid: set(v.get("Child", set())) for hid, v in self.data.items()}
        self.parents:  Dict[str, Set[str]] = {hid: set(v.get("Father", set())) for hid, v in self.data.items()}

        # Compute depths (BFS from root)
        self.depth: Dict[str, int] = {}
        if "HP:0000001" in self.data:
            dq = deque([("HP:0000001", 0)])
            seen = set()
            while dq:
                node, d = dq.popleft()
                if node in seen:
                    continue
                seen.add(node)
                self.depth[node] = d
                for ch in self.children.get(node, []):
                    dq.append((ch, d+1))

        # Lexicon (name/synonym -> ids)
        self.term2ids: Dict[str, Set[str]] = {}
        for hid, v in self.data.items():
            for t in v.get("Name", []):
                self.term2ids.setdefault(self._norm(t), set()).add(hid)
            for t in v.get("Synonym", []):
                self.term2ids.setdefault(self._norm(t), set()).add(hid)

    @staticmethod
    def _norm(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def resolve_id(self, hid: str) -> str:
        h = (hid or "").replace("_", ":")
        return self.alias2primary.get(h, h)

    def get_depth(self, hid: str) -> int:
        return self.depth.get(hid, 0)

    def is_ancestor(self, anc: str, desc: str) -> bool:
        cur = [desc]; visited = set()
        while cur:
            nxt = []
            for x in cur:
                if x == anc: return True
                for p in self.parents.get(x, []):
                    if p not in visited:
                        visited.add(p); nxt.append(p)
            cur = nxt
        return False

    def normalize_span(self, mention_text: str) -> List[str]:
        key = self._norm(mention_text)
        ids = list(self.term2ids.get(key, []))
        ids = sorted(ids, key=lambda x: self.get_depth(x), reverse=True)
        out, seen = [], set()
        for h in ids:
            if h not in seen:
                seen.add(h); out.append(h)
        return out

    def refine_id_set(self, ids: List[str]) -> List[str]:
        ids = [self.resolve_id(h) for h in ids]
        uniq = list(dict.fromkeys(ids))
        if len(uniq) > 1:
            keep = []
            s = set(uniq)
            for hid in uniq:
                drop = False
                for other in s:
                    if other == hid: 
                        continue
                    if self.is_ancestor(hid, other):
                        drop = True; break
                if not drop:
                    keep.append(hid)
            uniq = keep
        uniq = sorted(uniq, key=lambda x: self.get_depth(x), reverse=True)
        return uniq

# =============================================================================
# Data parsing + ann loaders
# =============================================================================
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_ann_tab(lines: List[str]) -> List[Tuple[SPAN, str, str]]:
    """GeneReviews / ID-68: '88 115 decrease in fetal movements HP:0001558'"""
    anns = []
    pat = re.compile(r"^(\d+)\s+(\d+)\s+(.*)\s+(HP[:_]\d+)\s*$")
    for raw in lines:
        line = raw.strip()
        if not line: continue
        m = pat.match(line)
        if not m:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and re.match(r"^HP[:_]\d+$", parts[-1] or ""):
                start = int(parts[0]); end = int(parts[1])
                hpo = parts[-1].replace("_", ":")
                mention = " ".join(parts[2:-1])
                anns.append(((start, end), mention, hpo))
        else:
            start, end, mention, hpo = m.groups()
            anns.append(((int(start), int(end)), mention, hpo.replace("_", ":")))
    return anns

def parse_ann_bracket(lines: List[str]) -> List[Tuple[SPAN, str, str]]:
    """GSC+: '[27::42]  HP_0000110 | renal dysplasia'"""
    anns = []
    pat = re.compile(r"^\[(\d+)::(\d+)\]\s*(HP[_:]\d+)\s*\|\s*(.+?)\s*$")
    for raw in lines:
        line = raw.strip()
        if not line: continue
        m = pat.match(line)
        if not m: continue
        start, end, hpo, mention = m.groups()
        anns.append(((int(start), int(end)), mention, hpo.replace("_", ":")))
    return anns

def load_ann_file(path: str) -> List[Tuple[SPAN, str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    probe = next((ln.strip() for ln in lines if ln.strip()), "")
    if not probe: return []
    if probe.startswith("[") and "::" in probe:
        return parse_ann_bracket(lines)
    return parse_ann_tab(lines)

# =============================================================================
# BIOES alignment helpers
# =============================================================================
def char_spans_to_token_labels(
    text: str,
    spans_only: List[SPAN],
    tokenizer: AutoTokenizer,
    max_len: int,
    stride: int,
    label_map: Dict[str, int],
) -> List[Dict]:
    enc_full = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
    input_ids_full = enc_full["input_ids"]
    offsets_full = enc_full["offset_mapping"]
    full_labels = [label_map["O"]] * len(input_ids_full)

    for (c0, c1) in spans_only:
        token_indices = []
        for ti, (a, b) in enumerate(offsets_full):
            if a == b:  # specials
                continue
            if not (b <= c0 or a >= c1):
                token_indices.append(ti)
        if not token_indices:
            continue
        if len(token_indices) == 1:
            full_labels[token_indices[0]] = label_map["S-PHENO"]
        else:
            full_labels[token_indices[0]] = label_map["B-PHENO"]
            for ti in token_indices[1:-1]:
                full_labels[ti] = label_map["I-PHENO"]
            full_labels[token_indices[-1]] = label_map["E-PHENO"]

    chunks = []
    start = 0
    while start < len(input_ids_full):
        end = min(start + max_len, len(input_ids_full))
        ids = input_ids_full[start:end]
        offs = offsets_full[start:end]
        labs = full_labels[start:end]
        chunks.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(ids), dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
            "offsets": offs,
            "slice": (start, end),
        })
        if end == len(input_ids_full): break
        start = max(0, end - stride)
        if start >= len(input_ids_full): break
    return chunks

# =============================================================================
# Dataset
# =============================================================================
class HPONERDataset(Dataset):
    def __init__(self, roots: List[str], tokenizer: AutoTokenizer, max_len: int, stride: int, label_map: Dict[str, int]):
        self.examples: List[Dict] = []
        for root in roots:
            ann_dir = os.path.join(root, "ann")
            corpus_dir = os.path.join(root, "corpus")
            if not (os.path.isdir(ann_dir) and os.path.isdir(corpus_dir)):
                logger.warning(f"Skip root (missing ann/corpus): {root}")
                continue
            for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*"))):
                if not os.path.isfile(ann_path) or ann_path.endswith("~"): continue
                base = os.path.basename(ann_path)
                text_path = os.path.join(corpus_dir, base)
                if not os.path.isfile(text_path):
                    alt = os.path.splitext(text_path)[0] + ".txt"
                    if os.path.isfile(alt): text_path = alt
                    else:
                        logger.warning(f"Missing corpus for ann: {ann_path}")
                        continue
                text = read_text(text_path)
                anns = load_ann_file(ann_path)  # [(char_span, mention, hpo_id)]
                spans_only = [sp for sp, _, _ in anns]
                chunks = char_spans_to_token_labels(text, spans_only, tokenizer, max_len, stride, label_map)
                for ch in chunks:
                    ch["meta"] = {
                        "root": root,
                        "ann_path": ann_path,
                        "text_path": text_path,
                        "base": base,
                        "full_text": text,
                        "anns": [{"span": sp, "mention": m, "hpo_id": hid} for (sp, m, hid) in anns],
                    }
                    self.examples.append(ch)
        logger.info(f"Built dataset with {len(self.examples)} windows from {len(roots)} roots.")

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return {
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
            "labels": x["labels"],
            "offsets": x["offsets"],
            "slice": x["slice"],
            "meta": x["meta"],
        }

def collate_pad(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(seq, val, L): return seq + [val]*(L - len(seq)
)
    input_ids = torch.tensor([pad(x["input_ids"].tolist(), pad_id, max_len) for x in batch], dtype=torch.long)
    attn      = torch.tensor([pad(x["attention_mask"].tolist(), 0, max_len) for x in batch], dtype=torch.long)
    labels    = torch.tensor([pad(x["labels"].tolist(), -100, max_len) for x in batch], dtype=torch.long)
    offsets   = [x["offsets"] for x in batch]
    slices    = [x["slice"] for x in batch]
    metas     = [x["meta"]  for x in batch]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels,
            "offsets": offsets, "slices": slices, "metas": metas}

# =============================================================================
# Model with optional CRF
# =============================================================================
class TokenCRFWrapper(nn.Module):
    def __init__(self, base_model: AutoModelForTokenClassification, num_labels: int, use_crf: bool):
        super().__init__()
        self.base = base_model
        self.num_labels = num_labels
        self.use_crf = use_crf and HAS_CRF
        if self.use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, output_hidden: bool=False):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=None, output_hidden_states=output_hidden)
        logits = outputs.logits
        hidden = outputs.hidden_states[-1] if output_hidden else None
        loss = None
        if labels is not None:
            if self.use_crf:
                mask = labels.ne(-100)
                labels_crf = labels.clone()
                labels_crf[~mask] = 0
                loss = - self.crf(logits, labels_crf, mask=mask, reduction="mean")
            else:
                loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits, "hidden": hidden}

    @torch.no_grad()
    def decode(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits
        if self.use_crf:
            mask = attention_mask.bool()
            paths = self.crf.decode(logits, mask=mask)
            maxL = logits.size(1)
            padded = []
            for seq in paths:
                seq = seq + [0]*(maxL - len(seq))
                padded.append(seq)
            return torch.tensor(padded, dtype=torch.long, device=logits.device)
        else:
            return logits.argmax(-1)

# =============================================================================
# Span/HPO head & Cross-Attention
# =============================================================================
class CrossAttnFuse(nn.Module):
    def __init__(self, h_size: int, d: int, nhead: int = 4):
        super().__init__()
        self.q = nn.Linear(h_size, d)
        self.k = nn.Linear(h_size, d)
        self.v = nn.Linear(h_size, d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.out = nn.Linear(d, d)

    def forward(self, span_vec_B1H, seq_H_BLH, key_padding_mask=None):
        Q = self.q(span_vec_B1H)        # [B,1,d]
        K = self.k(seq_H_BLH)           # [B,L,d]
        V = self.v(seq_H_BLH)           # [B,L,d]
        ctx,_ = self.attn(Q, K, V, key_padding_mask=key_padding_mask)  # [B,1,d]
        return self.out(ctx)            # [B,1,d]

class SpanHPOHead(nn.Module):
    def __init__(self, h_size: int, d: int, id_vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(h_size + d, d)  # concat(span, ctx) → d
        self.id_embed = nn.Embedding(id_vocab_size, d)

    def score(self, z_B1D, cand_idx_BK: torch.Tensor) -> torch.Tensor:
        E = self.id_embed(cand_idx_BK)     # [B,K,d]
        z = z_B1D.squeeze(1)               # [B,d]
        return torch.einsum("bd,bkd->bk", z, E)  # [B,K]

# =============================================================================
# Helpers
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_label_map(cfg: HPOConfig) -> Dict[str, int]:
    labels = [cfg.label_O, cfg.label_B, cfg.label_I, cfg.label_E, cfg.label_S]
    return {lab: i for i, lab in enumerate(labels)}

def spans_from_bioes(seq_ids: List[int], id2label: Dict[int, str]) -> List[Tuple[int, int]]:
    spans = []
    s = -1
    for i, y in enumerate(seq_ids):
        tag = id2label.get(y, "O")
        if tag == "O":
            if s != -1: s = -1
            continue
        if tag.startswith("S-"):
            spans.append((i, i))
        elif tag.startswith("B-"):
            s = i
        elif tag.startswith("E-"):
            if s != -1:
                spans.append((s, i)); s = -1
    return spans

def token_span_to_char_span(token_span: Tuple[int,int], offsets: List[Tuple[int,int]]) -> SPAN:
    i0, i1 = token_span
    a = offsets[i0][0]
    b = offsets[i1][1]
    return (a, b)

def postprocess_text_span(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^[\s\.,;:()\[\]\{\}]+", "", t)
    t = re.sub(r"[\s\.,;:()\[\]\{\}]+$", "", t)
    return t.strip()

def iou_tok(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    ai0, ai1 = a; bi0, bi1 = b
    inter = max(0, min(ai1,bi1) - max(ai0,bi0) + 1)
    if inter == 0: return 0.0
    ua = (ai1-ai0+1)+(bi1-bi0+1)-inter
    return inter/ua if ua>0 else 0.0

# =============================================================================
# Validation: NER
# =============================================================================
@torch.no_grad()
def evaluate_and_refine(model: TokenCRFWrapper,
                        loader: DataLoader,
                        device: torch.device,
                        label_map: Dict[str,int],
                        ontology: Optional[HPOOntology],
                        cfg: HPOConfig,
                        collect_examples: bool = False,
                        example_limit: int = 3) -> Dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support

    model.eval()
    all_gold: List[int] = []
    all_pred: List[int] = []

    span_tp = span_fp = span_fn = 0
    id2label = {i: l for l, i in label_map.items()}
    out_jsonl = os.path.join(cfg.model_dir, "val_refine.jsonl")
    if cfg.do_refine_on_val:
        open(out_jsonl, "w", encoding="utf-8").close()

    examples_tp, examples_fp, examples_fn = [], [], []

    pbar = tqdm(loader, desc="Eval NER", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attn      = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        preds = model.decode(input_ids=input_ids, attention_mask=attn)

        mask = labels.ne(-100)
        g = labels[mask].view(-1).cpu().tolist()
        p = preds[mask].view(-1).cpu().tolist()
        all_gold.extend(g); all_pred.extend(p)

        B = input_ids.size(0)
        for b in range(B):
            valid = mask[b].cpu().tolist()
            gold_seq = labels[b][valid].cpu().tolist()
            pred_seq = preds[b][valid].cpu().tolist()
            offs = [o for i,o in enumerate(batch["offsets"][b]) if valid[i]]

            meta_b = batch["metas"][b]
            text = meta_b["full_text"]
            anns_full = meta_b["anns"]

            gold_spans_tok = spans_from_bioes(gold_seq, id2label)
            pred_spans_tok = spans_from_bioes(pred_seq, id2label)
            gold_spans_char = { token_span_to_char_span(ts, offs) for ts in gold_spans_tok }
            pred_spans_char = [ token_span_to_char_span(ts, offs) for ts in pred_spans_tok ]

            set_pred = set(pred_spans_char)
            set_gold = set(gold_spans_char)
            tp_set = set_pred & set_gold
            fp_set = set_pred - set_gold
            fn_set = set_gold - set_pred

            span_tp += len(tp_set); span_fp += len(fp_set); span_fn += len(fn_set)

            if collect_examples:
                def _clip(a,b2): return postprocess_text_span(text[a:b2])
                for (a,b2) in list(tp_set)[:max(0, example_limit - len(examples_tp))]:
                    examples_tp.append({"root": meta_b["root"], "file": meta_b["base"],
                                        "span": [a,b2], "text": _clip(a,b2)})
                for (a,b2) in list(fp_set)[:max(0, example_limit - len(examples_fp))]:
                    examples_fp.append({"root": meta_b["root"], "file": meta_b["base"],
                                        "span": [a,b2], "text": _clip(a,b2)})
                for (a,b2) in list(fn_set)[:max(0, example_limit - len(examples_fn))]:
                    examples_fn.append({"root": meta_b["root"], "file": meta_b["base"],
                                        "span": [a,b2], "text": _clip(a,b2)})

            if cfg.do_refine_on_val and ontology is not None:
                refined_items = []
                for (a,b2) in pred_spans_char:
                    mention = postprocess_text_span(text[a:b2])
                    if not mention: continue
                    cands = ontology.normalize_span(mention)
                    if cands: cands = ontology.refine_id_set(cands)
                    refined_items.append({
                        "span": [a,b2], "mention": mention, "candidates": cands,
                        "picked": cands[0] if cands else None,
                        "depth": ontology.get_depth(cands[0]) if cands else None,
                    })
                with open(out_jsonl, "a", encoding="utf-8") as f:
                    rec = {
                        "file": meta_b["base"],
                        "slice": batch["slices"][b],
                        "pred_spans": [[int(a),int(b2)] for (a,b2) in pred_spans_char],
                        "refined": refined_items,
                        "gold": [{"span": list(x["span"]), "hpo_id": x["hpo_id"], "mention": x["mention"]} for x in anns_full],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    p, r, f1, _ = precision_recall_fscore_support(all_gold, all_pred, average="micro", zero_division=0)
    span_p = span_tp / (span_tp + span_fp + 1e-9)
    span_r = span_tp / (span_tp + span_fn + 1e-9)
    span_f1 = 2*span_p*span_r/(span_p + span_r + 1e-9)

    metrics = {"token_p": float(p), "token_r": float(r), "token_f1": float(f1),
               "span_p": float(span_p), "span_r": float(span_r), "span_f1": float(span_f1)}

    if collect_examples:
        metrics["_examples_tp"] = examples_tp[:example_limit]
        metrics["_examples_fp"] = examples_fp[:example_limit]
        metrics["_examples_fn"] = examples_fn[:example_limit]
    return metrics

# =============================================================================
# Validation: HPO-ID
# =============================================================================
@torch.no_grad()
def evaluate_hpo_id(model: TokenCRFWrapper,
                    cross_fuse: CrossAttnFuse,
                    span_head: SpanHPOHead,
                    loader: DataLoader,
                    ontology: HPOOntology,
                    id2idx: Dict[str,int],
                    cfg: HPOConfig,
                    device: torch.device,
                    iou_th: float = 0.5) -> Dict[str,float]:
    model.eval(); cross_fuse.eval(); span_head.eval()

    correct = total = topk_hits = 0
    Krep = cfg.hpo_topk

    label_map = build_label_map(cfg)
    id2label = {i:l for l,i in label_map.items()}

    pbar = tqdm(loader, desc="Eval HPO", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        metas     = batch["metas"]
        offs_list = batch["offsets"]

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=None, output_hidden=True)
        H = out["hidden"]                    # [B,L,H]
        preds_tok = model.decode(input_ids, attn_mask)

        B = input_ids.size(0)
        for b in range(B):
            offs = offs_list[b]
            def char2tok(a,b2):
                idx = [ti for ti,(x,y) in enumerate(offs) if (x!=y) and not (y<=a or x>=b2)]
                return (idx[0], idx[-1]) if idx else None

            gold_tok = []
            for ann in metas[b]["anns"]:
                tok = char2tok(ann["span"][0], ann["span"][1])
                if tok:
                    gold_tok.append((tok, ann["hpo_id"].replace("_", ":"), ann["mention"]))

            pred_tok = spans_from_bioes(preds_tok[b].tolist(), id2label)

            for ( (gi0,gi1), gold_id, mention ) in gold_tok:
                best = None; best_i = 0.0
                for (pi0,pi1) in pred_tok:
                    i = iou_tok((gi0,gi1), (pi0,pi1))
                    if i > best_i: best_i, best = i, (pi0,pi1)
                if best is None or best_i < iou_th:
                    continue

                cands = ontology.normalize_span(mention)
                gold_id = gold_id.replace("_", ":")
                if gold_id not in cands:
                    cands = [gold_id] + cands
                cands = ontology.refine_id_set(cands)
                if gold_id not in cands:
                    cands = [gold_id] + [h for h in cands if h != gold_id]
                cands = [h for h in cands if h in id2idx]
                if not cands:
                    if gold_id in id2idx: cands = [gold_id]
                    else: continue

                cand_idx = torch.tensor([[id2idx[h] for h in cands]], device=device)
                seqH = H[b:b+1]                                 # [1,L,H]
                span_vec = seqH[0, best[0]:best[1]+1].mean(dim=0).unsqueeze(0).unsqueeze(1)  # [1,1,H]
                ctx = cross_fuse(span_vec, seqH, key_padding_mask=(attn_mask[b:b+1]==0))
                z = torch.cat([span_vec, ctx], dim=-1)          # [1,1,H+d]
                z = span_head.proj(z)                           # <<< NEW: project to d
                logits = span_head.score(z, cand_idx)           # [1,K]
                probs = logits.softmax(-1).squeeze(0)           # [K]
                topk = probs.topk(min(Krep, probs.numel()))
                pred_id = cands[int(topk.indices[0])]
                total += 1
                if pred_id == gold_id: correct += 1
                picked = [cands[int(i)] for i in topk.indices.tolist()]
                if gold_id in picked: topk_hits += 1

    return {"hpo_acc": correct/max(1,total), "hpo_topk": topk_hits/max(1,total), "count": total}

# =============================================================================
# Save checkpoints (NER head/LoRA); HPO head saved separately
# =============================================================================
def save_checkpoint(cfg: HPOConfig,
                    model: TokenCRFWrapper,
                    tokenizer: AutoTokenizer,
                    cross_fuse: CrossAttnFuse,
                    span_head: SpanHPOHead,
                    epoch: int,
                    val_ner: Dict[str,float],
                    val_hpo: Dict[str,float],
                    is_best: bool):
    tag = "best" if is_best else f"epoch-{epoch}"
    outdir = os.path.join(cfg.model_dir, tag)
    os.makedirs(outdir, exist_ok=True)
    model.base.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    torch.save({"cross_fuse": cross_fuse.state_dict(),
                "span_head": span_head.state_dict()},
               os.path.join(outdir, "hpo_head.pt"))
    meta = {
        "epoch": epoch,
        "val_ner": val_ner,
        "val_hpo": val_hpo,
        "has_crf": model.use_crf,
        "label_schema": ["O","B-PHENO","I-PHENO","E-PHENO","S-PHENO"],
        "backbone": cfg.backbone,
        "init_encoder_from": cfg.init_encoder_from,
        "config": asdict(cfg),
        "timestamp": int(time.time()),
    }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# =============================================================================
# Encoder init: supports HF dir OR .pt state_dict
# =============================================================================
def maybe_init_encoder_from(base_tc: AutoModelForTokenClassification,
                            init_path: Optional[str]) -> None:
    """
    Initialize ONLY the encoder weights of `base_tc` from either:
      1) a HF directory (has config.json), or
      2) a .pt/.pth/.bin checkpoint file saved by RouterFirstMultiHead, or
      3) a directory containing best_model.pt / last_model.pt.

    It safely strips the 'encoder.' prefix and loads with strict=False.
    """
    if not init_path:
        logger.info("No init_encoder_from provided; using backbone as-is.")
        return
    if not os.path.exists(init_path):
        logger.warning(f"init_encoder_from path does not exist: {init_path}. Using backbone as-is.")
        return

    def _load_from_state_dict_file(sd_path: str):
        import torch
        from collections import OrderedDict
        logger.info(f"Initializing encoder from state_dict: {sd_path}")
        ckpt = torch.load(sd_path, map_location="cpu")
        # your intention script saves {"model_state": ..., ...}
        state = ckpt.get("model_state", ckpt)

        enc_only = OrderedDict()
        for k, v in state.items():
            if k.startswith("encoder."):
                enc_only[k[len("encoder."):]] = v  # strip prefix
            elif k.startswith("base_model."):
                # extremely rare case: someone saved TC model directly
                sub = k[len("base_model."):]
                if not sub.startswith("classifier."):  # skip heads
                    enc_only[sub] = v

        missing, unexpected = base_tc.base_model.load_state_dict(enc_only, strict=False)
        logger.info(f"Loaded encoder weights. Missing={len(missing)} Unexpected={len(unexpected)}")

    try:
        if os.path.isdir(init_path):
            # Case A: HF directory
            if os.path.isfile(os.path.join(init_path, "config.json")):
                logger.info(f"Initializing encoder from HF directory: {init_path}")
                enc_src = AutoModel.from_pretrained(init_path)
                missing, unexpected = base_tc.base_model.load_state_dict(enc_src.state_dict(), strict=False)
                logger.info(f"Loaded encoder weights. Missing={len(missing)} Unexpected={len(unexpected)}")
                return
            # Case B: directory with .pt
            for cand in ("best_model.pt", "last_model.pt"):
                cand_path = os.path.join(init_path, cand)
                if os.path.isfile(cand_path):
                    _load_from_state_dict_file(cand_path)
                    return
            # No usable files
            raise RuntimeError("No HF config.json and no best_model.pt/last_model.pt found.")
        else:
            # File path: expect a .pt/.pth/.bin
            if os.path.splitext(init_path)[1].lower() in (".pt", ".pth", ".bin"):
                _load_from_state_dict_file(init_path)
                return
            # Anything else -> try HF loader (will likely fail)
            logger.info(f"Trying HF loader for file: {init_path}")
            enc_src = AutoModel.from_pretrained(init_path)
            missing, unexpected = base_tc.base_model.load_state_dict(enc_src.state_dict(), strict=False)
            logger.info(f"Loaded encoder weights. Missing={len(missing)} Unexpected={len(unexpected)}")
    except Exception as e:
        logger.warning(f"Failed to init encoder from {init_path}: {e}. Using backbone as-is.")

def _make_loader(ds: Dataset, batch_size: int, pad_id: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: collate_pad(b, pad_id))

def _log_examples(tag: str, m: Dict, limit: int):
    logger.info(f"[{tag}] tokenF1={m['token_f1']:.4f} (P={m['token_p']:.4f}, R={m['token_r']:.4f}) | spanF1={m['span_f1']:.4f}")
    for label, key in (("TP", "_examples_tp"), ("FP", "_examples_fp"), ("FN", "_examples_fn")):
        ex = m.get(key, [])[:limit]
        if not ex: continue
        logger.info(f"[{tag}] {label} examples:")
        for e in ex:
            logger.info(f"  - file={e.get('file')} | span={e.get('span')} | text={e.get('text')!r}")

def _sanitize_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

# =============================================================================
# Train
# =============================================================================
def train_and_validate(cfg: HPOConfig):
    cfg.make_dirs()
    set_seed(cfg.seed)
    device = cfg.device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, use_fast=True)

    tb_dir = os.path.join(cfg.model_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    label_map = build_label_map(cfg)
    id2label = {i: l for l, i in label_map.items()}
    with open(os.path.join(cfg.model_dir, "labels.json"), "w") as f:
        json.dump({"label2id": label_map, "id2label": id2label}, f, indent=2)

    # Per-root datasets
    train_subsets = []
    per_root_val_loaders: Dict[str, DataLoader] = {}
    rnd = random.Random(cfg.seed)

    for root in list(cfg.train_roots):
        root_ds = HPONERDataset([root], tokenizer, cfg.max_len, cfg.stride, label_map)
        n = len(root_ds)
        if n == 0:
            logger.warning(f"Empty dataset for root: {root}")
            continue
        idxs = list(range(n))
        rnd.shuffle(idxs)
        k_val = max(1, int(round(cfg.per_root_val_ratio * n)))
        val_idx = idxs[:k_val]
        trn_idx = idxs[k_val:]
        train_subsets.append(Subset(root_ds, trn_idx))
        per_root_val_loaders[root] = _make_loader(Subset(root_ds, val_idx), cfg.batch_size, tokenizer.pad_token_id, shuffle=False)
        logger.info(f"[Split] {root}: train={len(trn_idx)} | val={len(val_idx)} (ratio={cfg.per_root_val_ratio})")
        root_tag = _sanitize_tag(root)
        writer.add_scalar(f"data/{root_tag}_train_windows", len(trn_idx), 0)
        writer.add_scalar(f"data/{root_tag}_val_windows", len(val_idx), 0)

    if not train_subsets:
        writer.close()
        raise RuntimeError("No training data after per-root splitting.")
    train_ds = ConcatDataset(train_subsets)

    # External validation set
    val_ds = HPONERDataset([cfg.val_root], tokenizer, cfg.max_len, cfg.stride, label_map)

    # loaders
    train_loader = _make_loader(train_ds, cfg.batch_size, tokenizer.pad_token_id, shuffle=True)
    val_loader   = _make_loader(val_ds, cfg.batch_size, tokenizer.pad_token_id, shuffle=False)

    # model + optional encoder init + LoRA
    base = AutoModelForTokenClassification.from_pretrained(
        cfg.backbone,
        num_labels=len(label_map),
        id2label=id2label,
        label2id=label_map,
    )
    maybe_init_encoder_from(base, cfg.init_encoder_from)

    # Freeze encoder params (we train LoRA + token head)
    for p in base.base_model.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none", task_type="TOKEN_CLS",
    )
    base = get_peft_model(base, lora_cfg)
    model = TokenCRFWrapper(base, num_labels=len(label_map), use_crf=cfg.use_crf).to(device)

    # HPO fusion + head
    hidden_size = model.base.config.hidden_size
    ontology = HPOOntology(cfg.hpo_json)
    id_list = sorted(list(ontology.data.keys()))
    id2idx = {h:i for i,h in enumerate(id_list)}

    cross_fuse = CrossAttnFuse(hidden_size, cfg.span_proj_dim, nhead=4).to(device)
    span_head  = SpanHPOHead(hidden_size, cfg.span_proj_dim, id_vocab_size=len(id_list)).to(device)

    # Params and optimizers (separate)
    ner_params = [p for p in model.parameters() if p.requires_grad]  # LoRA + token head(+CRF)
    hpo_params = list(cross_fuse.parameters()) + list(span_head.parameters())  # isolate HPO branch

    opt_ner = torch.optim.AdamW(ner_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_hpo = torch.optim.AdamW(hpo_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    # Schedulers (one for each)
    total_steps = cfg.epochs * math.ceil(len(train_loader)/max(1,cfg.gradient_accumulation_steps))
    warmup = int(cfg.warmup_ratio * total_steps)

    sch_ner = get_linear_schedule_with_warmup(opt_ner, warmup, total_steps)
    sch_hpo = get_linear_schedule_with_warmup(opt_hpo, warmup, total_steps)

    # TensorBoard param stats
    num_total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in cross_fuse.parameters()) + sum(p.numel() for p in span_head.parameters())
    num_train = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in cross_fuse.parameters() if p.requires_grad) + sum(p.numel() for p in span_head.parameters() if p.requires_grad)
    writer.add_scalar("model/params_total_millions", num_total/1e6, 0)
    writer.add_scalar("model/params_trainable_millions", num_train/1e6, 0)

    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    best_token_f1 = -1.0
    for epoch in range(1, cfg.epochs+1):
        model.train(); cross_fuse.train(); span_head.train()
        ner_running = 0.0
        hpo_running = 0.0

        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Train epoch {epoch}", leave=True)
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            # ===== A) NER step =====
            opt_ner.zero_grad(set_to_none=True)
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                outA = model(input_ids=input_ids, attention_mask=attn, labels=labels, output_hidden=False)
                token_loss = outA["loss"] / cfg.gradient_accumulation_steps
            scaler.scale(token_loss).backward()
            if step % cfg.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(ner_params, cfg.max_grad_norm)
                scaler.step(opt_ner); scaler.update()
                sch_ner.step()
            ner_running += float(token_loss.item()) * cfg.gradient_accumulation_steps

            # ===== B) HPO step =====
            opt_hpo.zero_grad(set_to_none=True)

            with torch.no_grad():
                pred_paths = model.decode(input_ids=input_ids, attention_mask=attn)

            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                outB = model(input_ids=input_ids, attention_mask=attn, labels=None, output_hidden=True)
                H_last = outB["hidden"].detach()
                key_pad = (attn == 0)

                span_items = []
                for b in range(input_ids.size(0)):
                    offs = batch["offsets"][b]
                    meta = batch["metas"][b]
                    text = meta["full_text"]

                    def char2tok(a,b2):
                        idx = [ti for ti,(x,y) in enumerate(offs) if (x!=y) and not (y<=a or x>=b2)]
                        return (idx[0], idx[-1]) if idx else None

                    gold_list = []
                    for ann in meta["anns"]:
                        a,b2 = ann["span"]
                        tt = char2tok(a,b2)
                        if tt:
                            gold_list.append( (tt[0], tt[1], ann["mention"], ann["hpo_id"].replace("_", ":")) )

                    label_map_local = build_label_map(cfg)
                    id2label_local = {i:l for l,i in label_map_local.items()}
                    pred_tok = spans_from_bioes(pred_paths[b].tolist(), id2label_local)

                    for (gi0,gi1,mention,gold_hid) in gold_list:
                        i0, i1 = gi0, gi1
                        if random.random() < cfg.hpo_pred_mix_p and len(pred_tok) > 0:
                            best, best_i = (i0,i1), 0.0
                            for (pi0,pi1) in pred_tok:
                                v = iou_tok((gi0,gi1),(pi0,pi1))
                                if v > best_i:
                                    best, best_i = (pi0,pi1), v
                            if best_i >= 0.5:
                                i0, i1 = best
                        span_items.append((b,i0,i1,mention,gold_hid))

                if len(span_items) > 0:
                    B2 = len(span_items)
                    span_vecs = []
                    cand_lists = []
                    gold_pos = []

                    for (b,i0,i1,mention,gold_hid) in span_items:
                        seqH = H_last[b]
                        sv   = seqH[i0:i1+1].mean(dim=0)
                        span_vecs.append(sv)

                        cands = ontology.normalize_span(mention)
                        gold_hid = gold_hid.replace("_", ":")
                        if gold_hid not in cands:
                            cands = [gold_hid] + cands
                        cands = ontology.refine_id_set(cands)
                        if gold_hid not in cands:
                            cands = [gold_hid] + [h for h in cands if h != gold_hid]
                        cands = [h for h in cands if h in id2idx]
                        if not cands:
                            if gold_hid in id2idx: cands = [gold_hid]
                            else: continue

                        cand_lists.append([id2idx[h] for h in cands])
                        gold_pos.append(cands.index(gold_hid))

                    if len(cand_lists) > 0:
                        Kmax = max(len(x) for x in cand_lists)
                        cand_idx = torch.full((len(cand_lists), Kmax), fill_value=0, device=device, dtype=torch.long)
                        mask_c   = torch.zeros((len(cand_lists), Kmax), device=device, dtype=torch.bool)
                        for i,ids in enumerate(cand_lists):
                            cand_idx[i,:len(ids)] = torch.tensor(ids, device=device)
                            mask_c[i,:len(ids)]   = 1

                        span_vecs = torch.stack(span_vecs, dim=0).unsqueeze(1)   # [B2,1,H]
                        ctx_list = []
                        for idx,(b,_,_,_,_) in enumerate(span_items):
                            ctx = cross_fuse(span_vecs[idx:idx+1], H_last[b:b+1], key_padding_mask=key_pad[b:b+1])
                            ctx_list.append(ctx)                                  # [1,1,d]
                        ctx_cat = torch.cat(ctx_list, dim=0)                      # [B2,1,d]
                        z = torch.cat([span_vecs, ctx_cat], dim=-1)               # [B2,1,H+d]
                        z = span_head.proj(z)                                      # <<< NEW: project to d
                        logits = span_head.score(z, cand_idx)                     # [B2,Kmax]
                        logits[~mask_c] = -1e9
                        gold_idx = torch.tensor(gold_pos, device=device, dtype=torch.long)
                        hpo_loss = ce(logits, gold_idx) / cfg.gradient_accumulation_steps
                    else:
                        hpo_loss = torch.tensor(0.0, device=device)
                else:
                    hpo_loss = torch.tensor(0.0, device=device)

            scaler.scale(hpo_loss).backward()
            if step % cfg.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(hpo_params, cfg.max_grad_norm)
                scaler.step(opt_hpo); scaler.update()
                sch_hpo.step()

            hpo_running += float(hpo_loss.item()) * cfg.gradient_accumulation_steps

            pbar.set_postfix(ner_loss=f"{ner_running/max(1,step):.4f}",
                             hpo_loss=f"{hpo_running/max(1,step):.4f}",
                             lr_ner=f"{sch_ner.get_last_lr()[0]:.2e}",
                             lr_hpo=f"{sch_hpo.get_last_lr()[0]:.2e}")

        # ===== Validation per epoch =====
        writer.add_scalar("train/ner_loss", ner_running/len(train_loader), epoch)
        writer.add_scalar("train/hpo_loss", hpo_running/len(train_loader), epoch)
        writer.add_scalar("train/lr_ner", sch_ner.get_last_lr()[0], epoch)
        writer.add_scalar("train/lr_hpo", sch_hpo.get_last_lr()[0], epoch)

        val_ner = evaluate_and_refine(model, val_loader, device, label_map, ontology, cfg,
                                      collect_examples=True, example_limit=cfg.per_root_example_limit)
        logger.info(f"[EXT] epoch={epoch} "
                    f"tokenF1={val_ner['token_f1']:.4f} (P={val_ner['token_p']:.4f}, R={val_ner['token_r']:.4f}) "
                    f"| spanF1={val_ner['span_f1']:.4f}")
        _log_examples("ext-val", val_ner, cfg.per_root_example_limit)

        writer.add_scalar("val_ext/token_f1", val_ner["token_f1"], epoch)
        writer.add_scalar("val_ext/token_p",  val_ner["token_p"], epoch)
        writer.add_scalar("val_ext/token_r",  val_ner["token_r"], epoch)
        writer.add_scalar("val_ext/span_f1",  val_ner["span_f1"], epoch)
        writer.add_scalar("val_ext/span_p",   val_ner["span_p"], epoch)
        writer.add_scalar("val_ext/span_r",   val_ner["span_r"], epoch)

        val_hpo = evaluate_hpo_id(model, cross_fuse, span_head, val_loader, ontology, id2idx, cfg, device)
        logger.info(f"[EXT-HPO] epoch={epoch} hpo_acc={val_hpo['hpo_acc']:.4f} | hpo_top{cfg.hpo_topk}={val_hpo['hpo_topk']:.4f} | n={val_hpo['count']}")
        writer.add_scalar("val_ext/hpo_acc",  val_hpo["hpo_acc"], epoch)
        writer.add_scalar("val_ext/hpo_topk", val_hpo["hpo_topk"], epoch)

        is_best = val_ner["token_f1"] > best_token_f1
        save_checkpoint(cfg, model, tokenizer, cross_fuse, span_head, epoch, val_ner, val_hpo, is_best=is_best)
        if is_best:
            best_token_f1 = val_ner["token_f1"]
            writer.add_scalar("val_ext/best_token_f1", best_token_f1, epoch)

        for root, r_loader in per_root_val_loaders.items():
            r_ner = evaluate_and_refine(model, r_loader, device, label_map, ontology, cfg,
                                        collect_examples=True, example_limit=cfg.per_root_example_limit)
            tag = f"root-holdout::{root}"
            _log_examples(tag, r_ner, cfg.per_root_example_limit)
            root_tag = _sanitize_tag(root)
            writer.add_scalar(f"val_root/{root_tag}/token_f1", r_ner["token_f1"], epoch)
            writer.add_scalar(f"val_root/{root_tag}/token_p",  r_ner["token_p"], epoch)
            writer.add_scalar(f"val_root/{root_tag}/token_r",  r_ner["token_r"], epoch)
            writer.add_scalar(f"val_root/{root_tag}/span_f1",  r_ner["span_f1"], epoch)
            writer.add_scalar(f"val_root/{root_tag}/span_p",   r_ner["span_p"], epoch)
            writer.add_scalar(f"val_root/{root_tag}/span_r",   r_ner["span_r"], epoch)
            r_hpo = evaluate_hpo_id(model, cross_fuse, span_head, r_loader, ontology, id2idx, cfg, device)
            writer.add_scalar(f"hpo/{root_tag}/acc",  r_hpo["hpo_acc"], epoch)
            writer.add_scalar(f"hpo/{root_tag}/top{cfg.hpo_topk}", r_hpo["hpo_topk"], epoch)

    logger.info(f"Done. Best external token F1={best_token_f1:.4f}. Artifacts: {cfg.model_dir}")
    writer.close()

# =============================================================================
# CLI
# =============================================================================
def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="HPO token-level NER (BIOES+CRF) with LoRA + Cross-Attn HPO mapping head (no loss mixing)")
    ap.add_argument("--backbone", type=str, default=" /models/BioLinkBERT-base")
    ap.add_argument("--init_encoder_from", type=str, default=" /tuned/intention/best_model.pt",
                    help="Either a HF model dir or a .pt state_dict file fine-tuned on the same BERT family.")
    ap.add_argument("--model_dir", type=str, default=" /tuned/hpo_lora_onto")
    ap.add_argument("--train_roots", type=str, nargs="+", default=[
        "  data/GeneReviews",
        "  data/GSC+",
        "  data/ID-68",
    ])
    ap.add_argument("--val_root", type=str, default="  data/val")
    ap.add_argument("--hpo_json", type=str, default="  data/hpo.json")

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--target_modules", type=str, nargs="+",
                    default=["query","key","value","dense","intermediate.dense","output.dense"])

    ap.add_argument("--use_crf", action="store_true", help="Enable CRF if torchcrf is installed")
    ap.add_argument("--no_refine_on_val", action="store_true")
    ap.add_argument("--no_ancestor_backoff", action="store_true")
    ap.add_argument("--no_dedup_parent", action="store_true")
    ap.add_argument("--no_prefer_deeper", action="store_true")

    ap.add_argument("--per_root_val_ratio", type=float, default=0.1)
    ap.add_argument("--per_root_example_limit", type=int, default=3)

    ap.add_argument("--span_proj_dim", type=int, default=256)
    ap.add_argument("--hpo_pred_mix_p", type=float, default=0.2)
    ap.add_argument("--hpo_topk", type=int, default=5)
    return ap.parse_args()

def main():
    args = _parse_args()
    cfg = HPOConfig(
        backbone=args.backbone,
        init_encoder_from=args.init_encoder_from if args.init_encoder_from else None,
        model_dir=args.model_dir,
        train_roots=args.train_roots,
        val_root=args.val_root,
        hpo_json=args.hpo_json,
        seed=args.seed,
        max_len=args.max_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_crf=args.use_crf,
        target_modules=tuple(args.target_modules),
        do_refine_on_val=not args.no_refine_on_val,
        ancestor_backoff=not args.no_ancestor_backoff,
        dedup_parent_when_child=not args.no_dedup_parent,
        prefer_deeper_when_tie=not args.no_prefer_deeper,
        per_root_val_ratio=args.per_root_val_ratio,
        per_root_example_limit=args.per_root_example_limit,
        span_proj_dim=args.span_proj_dim,
        hpo_pred_mix_p=args.hpo_pred_mix_p,
        hpo_topk=args.hpo_topk,
    )
    logger.info("Config:\n" + json.dumps(asdict(cfg), indent=2))
    train_and_validate(cfg)

if __name__ == "__main__":
    main()


"""
python  /process_bert/add_head/HPO/hpo_lora_hpoid.py   --backbone  /models/BioLinkBERT-base  
 --init_encoder_from  /tuned/intention   --model_dir  /tuned/hpo_lora_onto_Dhead 
     --train_roots         data/GeneReviews       
       data/GSC+         data/ID-68  
       --val_root   data/val  
 --hpo_json   data/hpo.json   --use_crf

"""