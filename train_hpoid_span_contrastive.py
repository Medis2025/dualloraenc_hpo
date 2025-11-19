#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Span-level HPO-ID Contrastive Training
- 利用真实 NER 标注 (GeneReviews / GSC+ / ID-68) + hpo.json
- 构造 mention span ↔ gold Name span 对
- 共用已训练好的 NER encoder 权重
- 新增一个 span projection head，优化 cos-sim 距离
- 记录到 TensorBoard，并在验证集上给出 HPO-ID top-1/top-5 指标

Run (example):

python train_hpoid_span_contrastive.py \
  --train_roots \
     /data/GeneReviews \
     /data/GSC+ \
     /data/ID-68 \
  --val_root  /data/val \
  --hpo_json  /data/hpo.json \
  --model_dir  /NER/tuned/hpo_lora_onto_Dhead/best \
  --backbone  /models/BioLinkBERT-base \
  --init_encoder_from  /NER/tuned/intention \
  --out_dir  /NER/tuned/hpoid_span_contrastive \
  --epochs 5 --batch_size 16 --max_len 512
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
#                      "expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:128")
import re
import json
import glob
import time
import math
import random
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

# --------- import original helpers / configs / wrappers ---------
from hpo_lora_hpoid import (
    HPOConfig,
    HPOOntology,
    TokenCRFWrapper,
    build_label_map,
    maybe_init_encoder_from,
)

# If your original module also exposes read_text / load_ann_file, you can import them.
# If not, we reimplement minimal versions here (compatible with the snippet you showed):

SPAN = Tuple[int, int]

logger = logging.getLogger("HPOIDSpanTrain")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)


# =============================================================================
# Ann / corpus helpers (compatible with your HPONERDataset snippet)
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
        if not line:
            continue
        m = pat.match(line)
        if not m:
            parts = line.split()
            if (
                len(parts) >= 3
                and parts[0].isdigit()
                and parts[1].isdigit()
                and re.match(r"^HP[:_]\d+$", parts[-1] or "")
            ):
                start = int(parts[0])
                end = int(parts[1])
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
        if not line:
            continue
        m = pat.match(line)
        if not m:
            continue
        start, end, hpo, mention = m.groups()
        anns.append(((int(start), int(end)), mention, hpo.replace("_", ":")))
    return anns


def load_ann_file(path: str) -> List[Tuple[SPAN, str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    probe = next((ln.strip() for ln in lines if ln.strip()), "")
    if not probe:
        return []
    if probe.startswith("[") and "::" in probe:
        return parse_ann_bracket(lines)
    return parse_ann_tab(lines)


# =============================================================================
# Gold sentence builder (Name + Def + Synonym)  ·  Name span as "gold span"
# =============================================================================
def build_hpo_gold_sentence_and_span(
    ontology: HPOOntology, hpo_id: str, max_syn: int = 3
) -> Tuple[str, SPAN]:
    """
    构造 gold sentence，以及 Name 在 sentence 中的 span（用于 span pooling）

    gold_text: "{Name}. {Def} Synonyms: a, b, c."
    name_span: (0, len(Name))
    """
    hid = ontology.resolve_id(hpo_id)
    rec = ontology.data.get(hid, {}) or {}

    names = rec.get("Name", [])
    name = names[0] if names else hid

    defs = rec.get("Def", [])
    definition = defs[0] if defs else ""

    syns = rec.get("Synonym", []) or []
    syns = [s for s in syns if isinstance(s, str)]
    syn_part = ""
    if syns:
        syn_show = syns[:max_syn]
        syn_part = " Synonyms: " + ", ".join(syn_show) + "."

    if definition:
        gold = f"{name}. {definition}{syn_part}"
    else:
        gold = f"{name}.{syn_part}"

    gold = gold.strip()
    name_span = (0, len(name))
    return gold, name_span


# =============================================================================
# Context window extractor (mention span → 截取上下文，并更新 span 坐标)
# =============================================================================
def extract_window(
    text: str, span: SPAN, max_window: int = 256
) -> Tuple[str, SPAN]:
    """
    从全文中截取一段包含 mention 的子串 (上下各 max_window/2)，并返回新的 span 坐标。
    """
    n = len(text)
    c0, c1 = span
    c0 = max(0, min(c0, n))
    c1 = max(0, min(c1, n))

    left = max(0, c0 - max_window // 2)
    right = min(n, c1 + max_window // 2)

    sub = text[left:right]
    new_span = (c0 - left, c1 - left)
    return sub, new_span


# =============================================================================
# HPO-ID Span Pair Dataset
# =============================================================================
class HPOIDSpanPairDataset(Dataset):
    """
    - roots 下的 ann/corpus 路径（GeneReviews / GSC+ / ID-68 / val）
    - 使用 NER 标注: (span, mention, hpo_id)
    - 使用 HPOOntology: 构造 gold(side) sentence + name span

    每个样本：
        {
          "left_text": str,        # mention 上下文
          "left_span": (c0,c1),    # left_text 内的 span
          "gold_text": str,        # HPO gold sentence
          "gold_span": (c0,c1),    # gold_text 内的 name span
          "hpo_id":    "HP:...",
          "dataset":   basename(root),
        }
    """

    def __init__(
        self,
        roots: List[str],
        ontology: HPOOntology,
        max_context_chars: int = 256,
        max_syn: int = 3,
    ):
        self.examples: List[Dict] = []
        self.ontology = ontology
        self.max_context_chars = max_context_chars
        self.max_syn = max_syn

        for root in roots:
            ann_dir = os.path.join(root, "ann")
            corpus_dir = os.path.join(root, "corpus")
            if not (os.path.isdir(ann_dir) and os.path.isdir(corpus_dir)):
                logger.warning(f"[HPOIDSpanPairDataset] Skip root (missing ann/corpus): {root}")
                continue

            dataset_name = os.path.basename(root.rstrip("/"))

            for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*"))):
                if not os.path.isfile(ann_path) or ann_path.endswith("~"):
                    continue
                base = os.path.basename(ann_path)

                text_path = os.path.join(corpus_dir, base)
                if not os.path.isfile(text_path):
                    alt = os.path.splitext(text_path)[0] + ".txt"
                    if os.path.isfile(alt):
                        text_path = alt
                    else:
                        logger.warning(f"[HPOIDSpanPairDataset] Missing corpus for ann: {ann_path}")
                        continue

                text = read_text(text_path)
                anns = load_ann_file(ann_path)  # [(span, mention, hpo_id)]

                if not anns:
                    continue

                for span, mention, hid_raw in anns:
                    hid = ontology.resolve_id(hid_raw)
                    if hid not in ontology.data:
                        continue

                    # 左侧：截上下文
                    left_text, left_span = extract_window(
                        text, span, max_window=self.max_context_chars
                    )

                    # 右侧：gold sentence + name span
                    gold_text, gold_span = build_hpo_gold_sentence_and_span(
                        ontology, hid, max_syn=self.max_syn
                    )

                    self.examples.append(
                        {
                            "left_text": left_text,
                            "left_span": left_span,
                            "gold_text": gold_text,
                            "gold_span": gold_span,
                            "hpo_id": hid,
                            "dataset": dataset_name,
                        }
                    )

        logger.info(
            f"[HPOIDSpanPairDataset] Built {len(self.examples)} span pairs from {len(roots)} root(s)."
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


# =============================================================================
# Span → token indices helper
# =============================================================================
def span_to_token_indices(
    offsets: List[Tuple[int, int]],
    span: SPAN,
    fallback_to_whole: bool = True,
) -> SPAN:
    c0, c1 = span
    token_indices = []
    for ti, (a, b) in enumerate(offsets):
        if a == b:
            continue  # special tokens
        # overlap check
        if not (b <= c0 or a >= c1):
            token_indices.append(ti)
    if not token_indices:
        if fallback_to_whole:
            # fallback: use all non-special tokens
            valid = [ti for ti, (a, b) in enumerate(offsets) if a != b]
            if not valid:
                return 0, 0
            return valid[0], valid[-1]
        else:
            return 0, 0

    if len(token_indices) == 1:
        return token_indices[0], token_indices[0]
    return token_indices[0], token_indices[-1]


# =============================================================================
# Simple span projection head
# =============================================================================
class SpanProj(nn.Module):
    """
    只在 encoder 输出的 span 向量上加一个线性投影 + dropout + L2 normalize。
    共用 NER encoder 的所有参数，单独训练这个头以调整 cos 距离。
    """

    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H]
        x = self.dropout(x)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


# =============================================================================
# Encode a batch of span texts using TokenCRFWrapper backbone
# =============================================================================
def encode_spans(
    model_tc: TokenCRFWrapper,
    span_proj: SpanProj,
    tokenizer: AutoTokenizer,
    texts: List[str],
    spans: List[SPAN],
    device: torch.device,
    max_len: int,
) -> torch.Tensor:
    """
    texts: list of sentences
    spans: list of char-level span in each sentence
    Return: [B, D] span embeddings
    """
    enc = tokenizer(
        texts,
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=max_len,
    )

    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device)
    attn_mask = torch.tensor(enc["attention_mask"], dtype=torch.long, device=device)
    offsets_all = enc["offset_mapping"]  # list of list

    # forward through NER encoder (shared)
    model_tc.train()  # keep dropout for training; will be set to eval() for validation
    out = model_tc(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=None,
        output_hidden=True,
    )
    H_last = out["hidden"]  # [B, L, H]

    B, L, H = H_last.shape
    span_vecs = []
    for i in range(B):
        offsets = offsets_all[i]
        i0, i1 = span_to_token_indices(offsets, spans[i])
        i0 = max(0, min(i0, L - 1))
        i1 = max(0, min(i1, L - 1))
        if i1 < i0:
            i1 = i0
        vec = H_last[i, i0 : i1 + 1, :].mean(dim=0)  # [H]
        span_vecs.append(vec)

    span_vecs = torch.stack(span_vecs, dim=0)  # [B, H]
    z = span_proj(span_vecs)  # [B, D] L2-normalized
    return z


@torch.no_grad()
def encode_hpo_gold_table(
    model_tc: TokenCRFWrapper,
    span_proj: SpanProj,
    tokenizer: AutoTokenizer,
    ontology: HPOOntology,
    hpo_ids: List[str],
    device: torch.device,
    max_len: int,
    max_syn: int = 3,
) -> Tuple[torch.Tensor, List[str]]:
    """
    为一组 HPO ID 构建 gold sentence + name span，并 encode 得到 [N_hpo, D] 向量表。
    """
    texts = []
    spans = []
    valid_ids = []

    for hid in hpo_ids:
        if hid not in ontology.data:
            continue
        gold_text, gold_span = build_hpo_gold_sentence_and_span(
            ontology, hid, max_syn=max_syn
        )
        texts.append(gold_text)
        spans.append(gold_span)
        valid_ids.append(hid)

    if not texts:
        return torch.empty(0, span_proj.proj.out_features, device=device), []

    model_tc.eval()
    span_proj.eval()
    enc = tokenizer(
        texts,
        return_offsets_mapping=True,
        truncation=True,
        padding=True,
        max_length=max_len,
    )
    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device)
    attn_mask = torch.tensor(enc["attention_mask"], dtype=torch.long, device=device)
    offsets_all = enc["offset_mapping"]

    out = model_tc(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=None,
        output_hidden=True,
    )
    H_last = out["hidden"]  # [N, L, H]
    N, L, H = H_last.shape

    span_vecs = []
    for i in range(N):
        offsets = offsets_all[i]
        i0, i1 = span_to_token_indices(offsets, spans[i])
        i0 = max(0, min(i0, L - 1))
        i1 = max(0, min(i1, L - 1))
        if i1 < i0:
            i1 = i0
        vec = H_last[i, i0 : i1 + 1, :].mean(dim=0)
        span_vecs.append(vec)

    span_vecs = torch.stack(span_vecs, dim=0)
    z = span_proj(span_vecs)
    return z, valid_ids


# =============================================================================
# Training & Evaluation
# =============================================================================
def info_nce_loss(z_left: torch.Tensor, z_right: torch.Tensor, temperature: float = 0.05):
    """
    z_left  : [B, D]
    z_right : [B, D]
    InfoNCE with in-batch negatives (双向)
    """
    z_left = F.normalize(z_left, dim=-1)
    z_right = F.normalize(z_right, dim=-1)

    logits = z_left @ z_right.t()  # [B, B]
    logits = logits / temperature
    labels = torch.arange(z_left.size(0), device=z_left.device)

    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_1 + loss_2)


def train_one_epoch(
    epoch: int,
    model_tc: TokenCRFWrapper,
    span_proj: SpanProj,
    tokenizer: AutoTokenizer,
    train_loader: DataLoader,
    cfg: HPOConfig,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
) -> int:
    model_tc.train()
    span_proj.train()

    optim = train_one_epoch.optim  # type: ignore
    temperature = getattr(cfg, "hpoid_temp", 0.05)

    pbar = tqdm(
        train_loader,
        desc=f"Train epoch {epoch}",
        total=len(train_loader),
        leave=True,
    )

    for step, batch in enumerate(pbar, start=1):
        # batch is a list[dict] because collate_fn=lambda x: x
        left_texts = [ex["left_text"] for ex in batch]
        left_spans = [ex["left_span"] for ex in batch]
        gold_texts = [ex["gold_text"] for ex in batch]
        gold_spans = [ex["gold_span"] for ex in batch]

        if len(left_texts) == 0:
            continue

        z_left = encode_spans(
            model_tc, span_proj, tokenizer, left_texts, left_spans, device, cfg.max_len
        )
        z_right = encode_spans(
            model_tc, span_proj, tokenizer, gold_texts, gold_spans, device, cfg.max_len
        )

        loss = info_nce_loss(z_left, z_right, temperature=temperature)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(span_proj.parameters(), max_norm=5.0)
        optim.step()

        global_step += 1
        writer.add_scalar("train/loss", loss.item(), global_step)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return global_step


@torch.no_grad()
def evaluate_hpoid(
    epoch: int,
    model_tc: TokenCRFWrapper,
    span_proj: SpanProj,
    tokenizer: AutoTokenizer,
    val_loader: DataLoader,
    ontology: HPOOntology,
    cfg: HPOConfig,
    device: torch.device,
    writer: SummaryWriter,
) -> Dict[str, float]:
    """
    HPO-ID level 验证：
    - 对 validation span（左边）encode
    - 对所有出现过的 HPO ID 构建 gold table（右边）
    - 计算每个 span 对所有 HPO gold 的 cos 相似度
    - 统计 top-1 / top-5 正确率
    """
    model_tc.eval()
    span_proj.eval()

    # 收集 val 的所有样本，先拿出 hpo_id 列表
    all_examples: List[Dict] = []
    for batch in val_loader:
        all_examples.extend(batch)

    if not all_examples:
        logger.warning("[Eval] Empty validation set.")
        metrics = {"val/top1": 0.0, "val/top5": 0.0}
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)
        return metrics

    # 统计 val 中出现过的 HPO ID
    hpo_ids_set = {ex["hpo_id"] for ex in all_examples}
    hpo_ids_table = sorted(hpo_ids_set)
    logger.info(f"[Eval] Building HPO gold table for {len(hpo_ids_table)} IDs...")

    z_hpo, hpo_ids_vec = encode_hpo_gold_table(
        model_tc,
        span_proj,
        tokenizer,
        ontology,
        hpo_ids_table,
        device=device,
        max_len=cfg.max_len,
    )
    if z_hpo.numel() == 0:
        logger.warning("[Eval] Empty HPO gold embedding table.")
        metrics = {"val/top1": 0.0, "val/top5": 0.0}
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)
        return metrics

    # 构建 id → index 映射
    id2idx = {hid: i for i, hid in enumerate(hpo_ids_vec)}
    logger.info(f"[Eval] HPO table size: {len(id2idx)}")

    top1_hits = 0
    top5_hits = 0
    total = 0

    # 分 batch encode 左侧 spans
    BATCH = cfg.batch_size
    for i in tqdm(
        range(0, len(all_examples), BATCH),
        desc=f"Eval epoch {epoch}",
        leave=False,
    ):
        chunk = all_examples[i : i + BATCH]
        left_texts = [ex["left_text"] for ex in chunk]
        left_spans = [ex["left_span"] for ex in chunk]
        hids_gold = [ex["hpo_id"] for ex in chunk]

        z_left = encode_spans(
            model_tc, span_proj, tokenizer, left_texts, left_spans, device, cfg.max_len
        )  # [b, D]

        # cos sim to all HPO gold vectors
        # z_left: [b, D], z_hpo: [N, D]; both normalized
        sims = z_left @ z_hpo.t()  # [b, N]

        # top-5 indices
        topv, topi = torch.topk(sims, k=min(5, sims.size(1)), dim=-1)

        for row, hid_true in enumerate(hids_gold):
            total += 1
            if hid_true not in id2idx:
                continue
            true_idx = id2idx[hid_true]
            preds = topi[row].tolist()
            if preds and preds[0] == true_idx:
                top1_hits += 1
            if true_idx in preds:
                top5_hits += 1

    top1 = top1_hits / max(1, total)
    top5 = top5_hits / max(1, total)

    metrics = {"val/top1": top1, "val/top5": top5}
    logger.info(
        f"[Eval] epoch={epoch} HPO-ID top1={top1:.4f}, top5={top5:.4f} (total={total})"
    )

    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)

    return metrics


# =============================================================================
# Model / config loader
# =============================================================================
def load_ner_tc_and_tokenizer(
    backbone: str,
    init_encoder_from: Optional[str],
    model_dir: str,
    cfg: HPOConfig,
) -> Tuple[AutoTokenizer, TokenCRFWrapper, Dict]:
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    label_map = build_label_map(cfg)
    id2label = {i: l for l, i in label_map.items()}

    base = AutoModelForTokenClassification.from_pretrained(
        backbone,
        num_labels=len(label_map),
        id2label=id2label,
        label2id=label_map,
    )
    if init_encoder_from:
        maybe_init_encoder_from(base, init_encoder_from)

    base = PeftModel.from_pretrained(base, model_dir)

    meta_path = os.path.join(model_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
    use_crf = bool(meta.get("has_crf", True))

    tc = TokenCRFWrapper(base, num_labels=len(label_map), use_crf=use_crf)
    return tokenizer, tc, meta


# =============================================================================
# CLI & main
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Span-level HPO-ID contrastive training using NER encoder + hpo.json"
    )
    ap.add_argument(
        "--train_roots",
        type=str,
        nargs="+",
        required=True,
        help="Training roots (each with ann/ and corpus/)",
    )
    ap.add_argument(
        "--val_root",
        type=str,
        required=True,
        help="Validation root (with ann/ and corpus/)",
    )
    ap.add_argument(
        "--hpo_json",
        type=str,
        required=True,
        help="Path to hpo.json",
    )
    ap.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="NER LoRA model dir (for TokenCRFWrapper / PeftModel)",
    )
    ap.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="HF backbone path (e.g., BioLinkBERT-base)",
    )
    ap.add_argument(
        "--init_encoder_from",
        type=str,
        default=None,
        help="Optional encoder init checkpoint (e.g., NER tuned intention)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output dir to save span projection weights, logs, etc.",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_context_chars", type=int, default=256)
    ap.add_argument("--hpoid_dim", type=int, default=256)
    ap.add_argument("--hpoid_temp", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tb_dir = os.path.join(args.out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # HPO config (reuse HPOConfig for backbone / json / max_len / batch_size)
    cfg = HPOConfig(
        backbone=args.backbone,
        init_encoder_from=args.init_encoder_from,
        model_dir=args.model_dir,
        hpo_json=args.hpo_json,
        max_len=args.max_len,
        batch_size=args.batch_size,
        stride=0,
        hpo_topk=5,
    )
    # attach extra fields used above
    setattr(cfg, "hpoid_temp", args.hpoid_temp)

    # Ontology
    ontology = HPOOntology(args.hpo_json)
    logger.info(f"Loaded HPO ontology from {args.hpo_json} with {len(ontology.data)} nodes.")

    # NER encoder + tokenizer
    tokenizer, model_tc, meta = load_ner_tc_and_tokenizer(
        args.backbone, args.init_encoder_from, args.model_dir, cfg
    )
    model_tc.to(device)

    # (Optionally) freeze NER encoder params,只训练 span_proj
    for p in model_tc.parameters():
        p.requires_grad = False

    # Span projection head
    hidden_size = model_tc.base.config.hidden_size
    span_proj = SpanProj(in_dim=hidden_size, out_dim=args.hpoid_dim, dropout=0.1).to(
        device
    )

    # Datasets
    train_ds = HPOIDSpanPairDataset(
        roots=args.train_roots,
        ontology=ontology,
        max_context_chars=args.max_context_chars,
        max_syn=3,
    )
    val_ds = HPOIDSpanPairDataset(
        roots=[args.val_root],
        ontology=ontology,
        max_context_chars=args.max_context_chars,
        max_syn=3,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,  # keep list[dict]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    # Optimizer (only span_proj)
    optim = torch.optim.AdamW(span_proj.parameters(), lr=args.lr)
    train_one_epoch.optim = optim  # hacky attach

    # Save config
    cfg_path = os.path.join(args.out_dir, "hpoid_span_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cli": vars(args),
                "hpo_cfg": asdict(cfg),
                "meta": meta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved config to {cfg_path}")

    global_step = 0
    best_top1 = -1.0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"===== Epoch {epoch}/{args.epochs} =====")
        t0 = time.time()
        global_step = train_one_epoch(
            epoch,
            model_tc,
            span_proj,
            tokenizer,
            train_loader,
            cfg,
            device,
            writer,
            global_step,
        )
        t1 = time.time()
        logger.info(f"Train epoch {epoch} done in {t1 - t0:.1f}s")

        # Eval
        metrics = evaluate_hpoid(
            epoch,
            model_tc,
            span_proj,
            tokenizer,
            val_loader,
            ontology,
            cfg,
            device,
            writer,
        )

        top1 = metrics.get("val/top1", 0.0)
        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.out_dir, f"hpoid_span_epoch{epoch:02d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "span_proj_state": span_proj.state_dict(),
                "cfg": asdict(cfg),
                "metrics": metrics,
                "cli": vars(args),
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint: {ckpt_path}")

        # Save best
        if top1 > best_top1:
            best_top1 = top1
            best_path = os.path.join(args.out_dir, "hpoid_span_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "span_proj_state": span_proj.state_dict(),
                    "cfg": asdict(cfg),
                    "metrics": metrics,
                    "cli": vars(args),
                },
                best_path,
            )
            logger.info(f"[BEST] Updated best top1={best_top1:.4f} at epoch {epoch}")

    writer.close()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()



"""
python train_hpoid_span_contrastive.py \
  --train_roots \
     /data/GeneReviews \
     /data/GSC+ \
     /data/ID-68 \
  --val_root  /data/val \
  --hpo_json  /data/hpo.json \
  --model_dir  /NER/tuned/hpo_lora_onto_Dhead/best \
  --backbone  /models/BioLinkBERT-base \
  --init_encoder_from  /NER/tuned/intention \
  --out_dir  /NER/tuned/hpoid_span_contrastive \
  --epochs 5 \
  --batch_size 16 \
  --max_len 512

"""