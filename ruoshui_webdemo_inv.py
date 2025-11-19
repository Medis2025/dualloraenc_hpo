#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backend for AquaFlow HPO NER demo (dualenc.html)

- 使用 span-contrastive 训练脚本里的组件：
  * TokenCRFWrapper (NER backbone)
  * SpanProj + encode_spans (DualLoRA span encoder)
  * HPOOntology + Stage-A (exact+fuzzy+direct) + 层级剪枝

- 前端 dualenc.html 的行为：
  * POST /api/hpo_demo，body:
      {
        "text": "...",
        "mode": "duallora" | "fuzzy" | "pipeline",
        "threshold": 0.3   # 0~1 之间
      }
  * 根据返回 spans[].start / spans[].end 精确高亮字符区域
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import logging
from typing import List, Dict, Tuple, Any

from flask import Flask, request, jsonify, send_from_directory

import torch

# 来自你的 span-contrastive 训练脚本
from train_hpoid_span_contrastive import (
    HPOConfig,
    HPOOntology,
    TokenCRFWrapper,
    HPOIDSpanPairDataset,
    SpanProj,
    encode_spans,
    encode_hpo_gold_table,
    load_ner_tc_and_tokenizer,
)

# rapidfuzz (可选)
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

logger = logging.getLogger("DualEncBackend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)

SPAN = Tuple[int, int]

# =============================================================================
# BASE PATHS (相对当前 web 目录)
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
WEB_MODEL_ROOT = os.path.join(BASE_DIR, "model")

# =============================================================================
# CONFIG（与 eval_hpoid_span_contrastive.py 示例一致，但使用相对路径）
# =============================================================================

EVAL_ROOTS = [
    os.path.join(WEB_DATASET_ROOT, "GeneReviews"),
    os.path.join(WEB_DATASET_ROOT, "GSC+"),
    os.path.join(WEB_DATASET_ROOT, "ID-68"),
]

HPO_JSON = os.path.join(WEB_MODEL_ROOT, "data", "hpo.json")
MODEL_DIR = os.path.join(WEB_MODEL_ROOT, "tuned", "hpo_lora_onto_Dhead", "best")
BACKBONE = os.path.join(WEB_MODEL_ROOT, "models", "BioLinkBERT-base")
INIT_ENCODER_FROM = os.path.join(WEB_MODEL_ROOT, "tuned", "intention")
OUT_DIR = os.path.join(BASE_DIR, "VAL")
CKPT_PATH = os.path.join(WEB_MODEL_ROOT, "tuned", "hpoid_span_contrastive", "hpoid_span_best.pt")

BATCH_SIZE = 32
MAX_LEN = 512
STAGEA_TOPK = 15

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 一些通用小 helper（BIOES + token→char 映射）
# =============================================================================

def spans_from_bioes(seq_ids: List[int], id2label: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    从 BIOES 标签序列还原 token-level spans（闭区间 [i0, i1]）。
    """
    spans: List[Tuple[int, int]] = []
    s = -1
    for i, y in enumerate(seq_ids):
        tag = id2label.get(int(y), "O")
        if tag.startswith("S-"):
            spans.append((i, i))
            s = -1
        elif tag.startswith("B-"):
            if s != -1:
                spans.append((s, i - 1))
            s = i
        elif tag.startswith("I-"):
            if s == -1:
                s = i
        elif tag.startswith("E-"):
            if s == -1:
                s = i
            spans.append((s, i))
            s = -1
        else:  # "O" or others
            if s != -1:
                spans.append((s, i - 1))
                s = -1
    if s != -1:
        spans.append((s, len(seq_ids) - 1))
    return spans


def token_span_to_char_span(token_span: Tuple[int, int], offsets: List[Tuple[int, int]]) -> SPAN:
    i0, i1 = token_span
    a = offsets[i0][0]
    b = offsets[i1][1]
    return (a, b)


def postprocess_text_span(text: str) -> str:
    """
    去掉 span 两侧多余标点/空格。
    """
    import re
    t = text.strip()
    t = re.sub(r"^[\s\.,;:()\[\]\{\}]+", "", t)
    t = re.sub(r"[\s\.,;:()\[\]\{\}]+$", "", t)
    return t.strip()

# =============================================================================
# 层级 Helper（与 eval 脚本同逻辑）
# =============================================================================
class HPOHierarchyHelper:
    def __init__(
        self,
        ontology: HPOOntology,
        depth_margin: float = 0.05,
        child_delta: float = 0.02,
        child_penalty: float = 0.05,
        depth_lambda: float = 0.10,
    ):
        self.onto = ontology
        self.depth_margin = float(depth_margin)
        self.child_delta = float(child_delta)
        self.child_penalty = float(child_penalty)
        self.depth_lambda = float(depth_lambda)

        if hasattr(self.onto, "depth") and self.onto.depth:
            self.max_depth = max(self.onto.depth.values())
        else:
            self.max_depth = 1

    def _depth_norm(self, hid: str) -> float:
        try:
            d = self.onto.get_depth(hid)
        except Exception:
            d = 0
        if self.max_depth <= 0:
            return 0.0
        return float(d) / float(self.max_depth)

    def prune_candidates(
        self,
        scored: List[Tuple[str, float]],
        max_k: int,
    ) -> List[str]:
        if not scored:
            return []

        norm_scored: List[Tuple[str, float]] = []
        for hid, s in scored:
            try:
                h_norm = self.onto.resolve_id(hid)
            except Exception:
                h_norm = hid
            norm_scored.append((h_norm, float(s)))

        norm_scored.sort(
            key=lambda x: (x[1], self._depth_norm(x[0])),
            reverse=True,
        )

        kept: List[Tuple[str, float]] = []
        for hid, s in norm_scored:
            drop = False
            for hk, sk in kept:
                try:
                    if self.onto.is_ancestor(hid, hk) and sk >= s - self.depth_margin:
                        drop = True
                        break
                except Exception:
                    continue
            if not drop:
                kept.append((hid, s))
            if len(kept) >= max_k:
                break

        out: List[str] = []
        seen: set = set()
        for h, _ in kept:
            if h in seen:
                continue
            seen.add(h)
            out.append(h)
            if len(out) >= max_k:
                break
        return out

    def rerank_indices(
        self,
        cand_indices: List[int],
        sims_tensor: torch.Tensor,
        hpo_ids_vec: List[str],
    ) -> List[int]:
        if not cand_indices or sims_tensor.numel() == 0:
            return cand_indices

        sims = sims_tensor.detach().cpu().tolist()
        if len(sims) != len(cand_indices):
            return cand_indices

        base_scores: Dict[int, float] = {
            idx: float(score) for idx, score in zip(cand_indices, sims)
        }

        depth_bias: Dict[int, float] = {}
        for idx in cand_indices:
            hid = hpo_ids_vec[idx]
            depth_bias[idx] = self.depth_lambda * self._depth_norm(hid)

        penalty: Dict[int, float] = {idx: 0.0 for idx in cand_indices}
        cand_hids: Dict[int, str] = {idx: hpo_ids_vec[idx] for idx in cand_indices}

        for idx_a in cand_indices:
            h_a = cand_hids[idx_a]
            s_a = base_scores[idx_a] + depth_bias.get(idx_a, 0.0)
            for idx_b in cand_indices:
                if idx_a == idx_b:
                    continue
                h_b = cand_hids[idx_b]
                s_b = base_scores[idx_b] + depth_bias.get(idx_b, 0.0)
                if s_b < s_a + self.child_delta:
                    continue
                try:
                    if self.onto.is_ancestor(h_a, h_b):
                        penalty[idx_a] += self.child_penalty
                        break
                except Exception:
                    continue

        final_scores: List[Tuple[int, float]] = []
        for idx in cand_indices:
            fs = base_scores[idx] + depth_bias.get(idx, 0.0) - penalty.get(idx, 0.0)
            final_scores.append((idx, fs))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in final_scores]


# =============================================================================
# Stage-A 候选（exact + fuzzy + direct_NER_bert）
#   mode 映射：
#     - "pipeline" / "all" / 其它默认：exact + fuzzy + direct
#     - "duallora"（DualLoRAEnc only）：仅 direct_NER_bert
#     - "fuzzy"（Fuzzy only）：exact + fuzzy（不使用 direct_NER_bert）
# =============================================================================
def ensure_fuzzy_cache(ontology: HPOOntology):
    if hasattr(ontology, "_keys"):
        return
    keys = list(getattr(ontology, "term2ids", {}).keys())
    ontology._keys = keys
    logger.info(f"[StageA] Built NER_FUZZY key cache with {len(keys)} keys from term2ids.")


def stageA_ranked(
    mention_text: str,
    ontology: HPOOntology,
    fuzzy_limit: int,
    z_left_row: torch.Tensor,
    z_hpo: torch.Tensor,
    hpo_ids_vec: List[str],
    mode: str = "all",
) -> List[Tuple[str, float]]:
    """
    返回 [(hpo_id, stageA_score)]，综合 exact / fuzzy / direct_NER_bert。

    mode 语义：
      - "pipeline" / "all" / 其它默认：exact + fuzzy + direct_NER_bert
      - "duallora"：仅 direct_NER_bert（DualLoRAEnc-only）
      - "fuzzy"：仅 exact + fuzzy，不用 direct_NER_bert
    """
    mention_text = (mention_text or "").strip()
    if not mention_text:
        return []

    mode = (mode or "all").lower()

    # 控制三种信号
    use_exact = True
    use_fuzzy = True
    use_direct = True

    if mode in ("duallora", "ner_only", "direct_only"):
        use_exact = False
        use_fuzzy = False
        use_direct = True
    elif mode in ("fuzzy", "fuzzy_only", "lex_fuzzy_only"):
        use_exact = True
        use_fuzzy = True
        use_direct = False
    else:
        # "pipeline" / "all" / 其它默认
        use_exact = True
        use_fuzzy = True
        use_direct = True

    # ----- EXACT ----- (Lexical exact)
    exact_map: Dict[str, float] = {}
    if use_exact:
        try:
            norm_key = ontology._norm(mention_text)
            for hid in ontology.term2ids.get(norm_key, set()):
                exact_map[hid] = 1.0
        except Exception:
            exact_map = {}

    # ----- FUZZY -----（RapidFuzz）
    fuzzy_map: Dict[str, float] = {}
    if use_fuzzy and HAS_RAPIDFUZZ:
        try:
            ensure_fuzzy_cache(ontology)
            if hasattr(ontology, "_norm_adv"):
                key = ontology._norm_adv(mention_text)
            else:
                key = ontology._norm(mention_text)
            hits = rf_process.extract(
                key,
                ontology._keys,
                scorer=rf_fuzz.token_set_ratio,
                limit=min(fuzzy_limit, max(1, len(ontology._keys))),
            )
            thr = 88.0
            for cand, rf_score, _ in hits:
                if rf_score < thr:
                    continue
                for hid in ontology.term2ids.get(cand, []):
                    s = (max(rf_score, 0.0) / 100.0) * 0.5
                    prev = fuzzy_map.get(hid, 0.0)
                    if s > prev:
                        fuzzy_map[hid] = s
        except Exception:
            fuzzy_map = {}

    # ----- direct_NER_bert -----（DualLoRAEnc span encoder）
    direct_map: Dict[str, float] = {}
    if use_direct:
        try:
            if z_left_row.dim() == 1:
                sims = torch.mv(z_hpo, z_left_row)
            else:
                sims = torch.matmul(z_hpo, z_left_row.squeeze(0))
            sims_list = sims.detach().cpu().tolist()
            for idx, hid in enumerate(hpo_ids_vec):
                direct_map[hid] = float(sims_list[idx])
        except Exception:
            direct_map = {}

    all_ids = set(exact_map.keys()) | set(fuzzy_map.keys()) | set(direct_map.keys())
    scored: List[Tuple[str, float]] = []
    for hid in all_ids:
        s_exact = exact_map.get(hid, None)
        s_fuzzy = fuzzy_map.get(hid, None)
        s_direct = direct_map.get(hid, None)
        scores = [x for x in (s_exact, s_fuzzy) if x is not None]
        if not scores:
            continue
        combined = max(scores)
        scored.append((hid, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# =============================================================================
# HPO 元数据
# =============================================================================
def get_hpo_metadata(ontology: HPOOntology, hpo_id: str) -> Dict[str, Any]:
    rec = {}
    try:
        rec = ontology.data.get(hpo_id, {})
    except Exception:
        rec = {}

    name = (
        rec.get("name")
        or rec.get("Name")
        or rec.get("label")
        or None
    )
    if not name and hasattr(ontology, "names"):
        name = getattr(ontology, "names").get(hpo_id, hpo_id)
    if not name:
        name = hpo_id

    syn = (
        rec.get("synonyms")
        or rec.get("Synonym")
        or rec.get("syns")
        or []
    )
    if isinstance(syn, str):
        syn = [syn]
    if not isinstance(syn, list):
        syn = []

    definition = (
        rec.get("def")
        or rec.get("definition")
        or rec.get("Definition")
        or ""
    )
    if isinstance(definition, dict) and "val" in definition:
        definition = definition["val"]

    return {
        "name": name,
        "synonyms": syn,
        "definition": definition,
    }


# =============================================================================
# NER span 抽取：先用 NER → 再 fallback 到 heuristic
# =============================================================================
def _heuristic_spans(text: str) -> List[SPAN]:
    """
    兜底 heuristic：句子 + ',' ';' 'and' 切片。
    只有在模型 decode 有问题时才会用到。
    """
    import re
    spans: List[SPAN] = []
    if not text.strip():
        return spans

    sentences = re.split(r'[\n。!?]+', text)
    offset = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            offset += 1
            continue

        idx = text.find(sent, offset)
        if idx == -1:
            idx = offset
        sent_start = idx
        sent_end = idx + len(sent)

        chunks = re.split(r'[;,]| and ', sent)
        running = sent_start
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            ch_idx = text.find(chunk, running, sent_end)
            if ch_idx == -1:
                ch_idx = running
            start = ch_idx
            end = ch_idx + len(chunk)
            running = end

            length = end - start
            if 4 <= length <= 120:
                spans.append((start, end))

        offset = sent_end + 1

    spans = sorted(set(spans), key=lambda x: (x[0], x[1]))
    return spans


def extract_mentions_from_text(
    text: str,
    tokenizer,
    model_tc: TokenCRFWrapper,
    device: torch.device,
) -> List[SPAN]:
    """
    返回【字符级】span 列表，只高亮真正 NER 出来的区域。

    步骤：
      1) tokenizer(text, return_offsets_mapping=True)
      2) model_tc.decode(...) 得到标签 id 序列
      3) 用 config.id2label 做 BIOES 解码
      4) token span → char span

    若任何一步失败，则 fallback 到 heuristic。
    """
    text = (text or "").strip()
    if not text:
        return []

    try:
        enc = tokenizer(
            text,
            max_length=MAX_LEN,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"][0].tolist()  # List[(char_start, char_end)]

        # 1) decode token labels
        with torch.no_grad():
            if not hasattr(model_tc, "decode"):
                raise RuntimeError("TokenCRFWrapper has no .decode method.")
            decoded = model_tc.decode(input_ids=input_ids, attention_mask=attention_mask)

        # decoded 通常是 [B,L] 或 list[list[int]]
        if isinstance(decoded, torch.Tensor):
            tag_ids = decoded[0].tolist()
        else:
            tag_ids = list(decoded[0])

        # 2) 取 id2label：优先 wrapper 自带，其次 base.config.id2label
        id2label: Dict[int, str] = {}
        if hasattr(model_tc, "id2label") and isinstance(model_tc.id2label, dict):
            for k, v in model_tc.id2label.items():
                id2label[int(k)] = str(v)
        elif hasattr(model_tc, "base") and hasattr(model_tc.base, "config"):
            cfg = model_tc.base.config
            if hasattr(cfg, "id2label") and isinstance(cfg.id2label, dict):
                for k, v in cfg.id2label.items():
                    id2label[int(k)] = str(v)

        if not id2label:
            raise RuntimeError("No id2label mapping found on model_tc.")

        # 3) BIOES 解码成 token spans，再映射到 char spans
        tok_spans = spans_from_bioes(tag_ids, id2label)
        spans_char: List[SPAN] = []
        for (i0, i1) in tok_spans:
            if i0 < 0 or i1 >= len(offsets):
                continue
            s_char, e_char = token_span_to_char_span((i0, i1), offsets)
            s_char = max(0, min(s_char, len(text)))
            e_char = max(s_char, min(e_char, len(text)))
            if e_char > s_char:
                spans_char.append((s_char, e_char))

        spans_char = sorted(set(spans_char), key=lambda x: (x[0], x[1]))
        if spans_char:
            logger.debug(f"[NER] token decode -> {len(spans_char)} span(s).")
            return spans_char

    except Exception:
        logger.exception("[NER] decode to char spans failed, fallback to heuristic.")

    # fallback
    spans = _heuristic_spans(text)
    logger.debug(f"[NER] heuristic spans -> {len(spans)} span(s).")
    return spans


# =============================================================================
# Flask app + 延迟模型加载
# =============================================================================

STATIC_HTML_DIR = os.path.join(BASE_DIR, "html")

app = Flask(
    __name__,
    static_folder=STATIC_HTML_DIR,
    static_url_path="/html",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局状态：延迟加载
STATE: Dict[str, Any] = {
    "init_ok": False,
    "init_error": "",
    "ontology": None,
    "cfg": None,
    "tokenizer": None,
    "model_tc": None,
    "span_proj": None,
    "z_hpo": None,
    "hpo_ids_vec": None,
    "id2idx": None,
    "hier": None,
}


def init_models_if_needed() -> None:
    """
    延迟初始化：第一次访问 /ping 或 /api/hpo_demo 时加载。
    """
    if STATE["init_ok"] or STATE["init_error"]:
        return

    logger.info("[Backend] Initializing HPO NER models...")
    try:
        ontology = HPOOntology(HPO_JSON)
        logger.info(f"[Backend] Loaded HPO ontology with {len(ontology.data)} nodes.")

        cfg = HPOConfig(
            backbone=BACKBONE,
            init_encoder_from=INIT_ENCODER_FROM,
            model_dir=MODEL_DIR,
            hpo_json=HPO_JSON,
            max_len=MAX_LEN,
            batch_size=BATCH_SIZE,
            stride=0,
            hpo_topk=5,
        )

        tokenizer, model_tc, meta = load_ner_tc_and_tokenizer(
            BACKBONE, INIT_ENCODER_FROM, MODEL_DIR, cfg
        )
        model_tc.to(device)
        for p in model_tc.parameters():
            p.requires_grad = False
        model_tc.eval()

        # TokenCRFWrapper 里 base 指向 HF 模型
        hidden_size = model_tc.base.config.hidden_size

        if not os.path.isfile(CKPT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        span_dim = ckpt.get("cfg", {}).get("hpoid_dim", 256)
        span_proj = SpanProj(in_dim=hidden_size, out_dim=span_dim, dropout=0.0).to(device)
        span_proj.load_state_dict(ckpt["span_proj_state"])
        span_proj.eval()

        # 构造 HPO table
        all_ds = HPOIDSpanPairDataset(
            roots=EVAL_ROOTS,
            ontology=ontology,
            max_context_chars=256,
            max_syn=3,
        )
        hpo_ids_table = sorted({ex["hpo_id"] for ex in all_ds if ex["hpo_id"] in ontology.data})
        logger.info(f"[Backend] Union of HPO IDs across eval_roots: {len(hpo_ids_table)}")

        z_hpo, hpo_ids_vec = encode_hpo_gold_table(
            model_tc,
            span_proj,
            tokenizer,
            ontology,
            hpo_ids_table,
            device=device,
            max_len=MAX_LEN,
        )
        if z_hpo.numel() == 0:
            logger.warning("[Backend] Empty HPO embedding table.")

        id2idx = {hid: i for i, hid in enumerate(hpo_ids_vec)}
        hier = HPOHierarchyHelper(ontology)

        STATE.update(
            init_ok=True,
            ontology=ontology,
            cfg=cfg,
            tokenizer=tokenizer,
            model_tc=model_tc,
            span_proj=span_proj,
            z_hpo=z_hpo,
            hpo_ids_vec=hpo_ids_vec,
            id2idx=id2idx,
            hier=hier,
        )
        logger.info("[Backend] Initialization done.")

    except Exception as e:
        logger.exception("[Backend] Initialization failed.")
        STATE["init_error"] = str(e)


# =============================================================================
# 核心推理函数：text -> spans + HPO Top1/Top5
#   mode: "duallora" | "fuzzy" | "pipeline"
#   threshold: 0~1，过滤 Top1 置信度过低的 span
# =============================================================================
def run_hpo_demo(text: str, mode: str = "pipeline", threshold: float = 0.0) -> Dict[str, Any]:
    init_models_if_needed()
    if not STATE["init_ok"]:
        return {
            "text": text,
            "spans": [],
            "mode": mode,
            "threshold": float(threshold),
            "error": f"backend_init_failed: {STATE['init_error']}",
        }

    ontology: HPOOntology = STATE["ontology"]
    tokenizer = STATE["tokenizer"]
    model_tc: TokenCRFWrapper = STATE["model_tc"]
    span_proj: SpanProj = STATE["span_proj"]
    z_hpo: torch.Tensor = STATE["z_hpo"]
    hpo_ids_vec: List[str] = STATE["hpo_ids_vec"]
    id2idx: Dict[str, int] = STATE["id2idx"]
    hier: HPOHierarchyHelper = STATE["hier"]

    text = (text or "").strip("\n")
    if not text:
        return {"text": text, "spans": [], "mode": mode, "threshold": float(threshold)}

    # 防御性 clamp 阈值
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.0
    if threshold < 0.0:
        threshold = 0.0
    if threshold > 1.0:
        threshold = 1.0

    # 1) 用 NER 模型抽取字符级 spans（只高亮这些）
    spans_char: List[SPAN] = extract_mentions_from_text(
        text,
        tokenizer=tokenizer,
        model_tc=model_tc,
        device=device,
    )
    if not spans_char:
        return {"text": text, "spans": [], "mode": mode, "threshold": float(threshold)}

    # 2) 将这些 span 通过 DualLoRA span encoder 编码
    left_texts = [text for _ in spans_char]
    left_spans = spans_char
    with torch.no_grad():
        z_left = encode_spans(
            model_tc, span_proj, tokenizer, left_texts, left_spans, device, MAX_LEN
        )  # [B, D]

    out_spans: List[Dict[str, Any]] = []

    # 将前端三种模式映射到 Stage-A mode
    stageA_mode = (mode or "pipeline").lower()
    # "duallora" / "fuzzy" / "pipeline" 在 stageA_ranked 中直接识别

    for row, (start, end) in enumerate(spans_char):
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        mention_raw = text[start:end]
        mention = postprocess_text_span(mention_raw)
        if not mention:
            continue

        # Stage-A: lexical/fuzzy/direct 综合候选（带 mode）
        ranked = stageA_ranked(
            mention_text=mention,
            ontology=ontology,
            fuzzy_limit=256,
            z_left_row=z_left[row],
            z_hpo=z_hpo,
            hpo_ids_vec=hpo_ids_vec,
            mode=stageA_mode,
        )
        if not ranked:
            continue

        # 层级剪枝
        cands = hier.prune_candidates(ranked, max_k=STAGEA_TOPK)
        if not cands:
            continue

        cand_indices = [id2idx[h] for h in cands if h in id2idx]
        if not cand_indices:
            continue

        with torch.no_grad():
            sims = z_left[row : row + 1, :] @ z_hpo[cand_indices, :].t()  # [1, K]
            sims = sims.squeeze(0)  # [K]

        # global_idx -> local_pos
        idx2pos = {g_idx: pos for pos, g_idx in enumerate(cand_indices)}

        new_order = hier.rerank_indices(
            cand_indices=cand_indices,
            sims_tensor=sims,
            hpo_ids_vec=hpo_ids_vec,
        )
        if not new_order:
            continue

        # Top-5
        top_indices = new_order[: min(5, len(new_order))]
        top5: List[Dict[str, Any]] = []
        for g_idx in top_indices:
            local_pos = idx2pos.get(g_idx, None)
            score = float(sims[local_pos]) if local_pos is not None else 0.0
            hid = hpo_ids_vec[g_idx]
            meta = get_hpo_metadata(ontology, hid)
            top5.append({
                "id": hid,
                "name": meta["name"],
                "score": float(score),
            })

        if not top_indices:
            continue

        # Top-1
        top1_idx = top_indices[0]
        top1_pos = idx2pos.get(top1_idx, None)
        hid_top1 = hpo_ids_vec[top1_idx]
        meta_top1 = get_hpo_metadata(ontology, hid_top1)
        score_top1 = float(sims[top1_pos]) if top1_pos is not None else 0.0

        # 应用置信度阈值：低于 threshold 的 span 直接丢弃
        if threshold > 0.0 and score_top1 < threshold:
            continue

        # NOTE:
        # - top5: 前端 dualenc.html 默认读取的字段（tooltip / 卡片）
        # - candidates / topk: 兼容旧前端或其他 demo 的字段名
        out_spans.append({
            "start": start,
            "end": end,
            "text": mention,
            "hpo_id": hid_top1,
            "name": meta_top1["name"],
            "score": float(score_top1),
            "synonyms": meta_top1["synonyms"],
            "definition": meta_top1["definition"],
            "top5": top5,
            "candidates": top5,
            "topk": top5,
        })

    return {
        "text": text,
        "spans": out_spans,
        "mode": mode,
        "threshold": float(threshold),
    }


# =============================================================================
# Routes
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    if not os.path.isdir(STATIC_HTML_DIR):
        logger.warning(f"[Backend] STATIC_HTML_DIR not found: {STATIC_HTML_DIR}")
        return "html directory not found", 500
    html_path = os.path.join(STATIC_HTML_DIR, "dualenc.html")
    if not os.path.isfile(html_path):
        logger.warning(f"[Backend] dualenc.html not found at {html_path}")
        return "dualenc.html not found", 500
    return send_from_directory(STATIC_HTML_DIR, "dualenc.html")


@app.route("/ruoshui", methods=["GET"])
def ruoshui():
    return index()


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return "", 204


@app.route("/ping", methods=["GET"])
def ping():
    init_models_if_needed()
    return jsonify({
        "status": "ok",
        "device": str(device),
        "init_ok": bool(STATE["init_ok"]),
        "init_error": STATE["init_error"],
        "hpo_size": int(len(STATE["id2idx"])) if STATE["id2idx"] else 0,
    })


@app.route("/api/hpo_demo", methods=["POST"])
def api_hpo_demo():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "JSON body must be an object"}), 400

    text = data.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "`text` must be a string"}), 400

    # 模式与阈值
    mode = data.get("mode", "pipeline")
    threshold = data.get("threshold", 0.0)

    try:
        result = run_hpo_demo(text, mode=mode, threshold=threshold)
    except Exception as e:
        logger.exception("Error running HPO demo")
        return jsonify({"error": f"backend exception: {e}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    logger.info(f"[Backend] Starting on 0.0.0.0:8008 (device={device})")
    app.run(host="0.0.0.0", port=8008, debug=False)
