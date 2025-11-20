import os
import sys
import json
import time
import argparse
import random
import html
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Any

# NOTE: Ancienne implémentation locale HF NER supprimée.
# L'évaluation s'appuie désormais sur l'ensemble NER (GLiNER + HF fallback)
# géré par l'orchestrateur via overrides (gliner_models, etc.).

from dotenv import load_dotenv
from tqdm import tqdm

BEST_GLINER_MODELS = [
    "EmergentMethods/gliner_medium_news-v2.1",
    "numind/NuNER_Zero-span",
    "urchade/gliner_large-v2.1",
    "urchade/gliner_multi-v2.1",
]

def add_to_syspath(p: str):
    pth = Path(p).expanduser().resolve()
    if pth.exists() and str(pth) not in sys.path:
        sys.path.insert(0, str(pth))

def import_anonymize():
    candidates = [
        "orchestrator",
        "src.orchestrator",
        "main",
        "anonymizer",
        "src.anonymizer",
    ]
    last_err = None
    for mod in candidates:
        try:
            m = __import__(mod, fromlist=["anonymize_text"])
            if hasattr(m, "anonymize_text"):
                return m.anonymize_text
        except Exception as e:
            last_err = e
    raise ImportError(
        f"Impossible d'importer anonymize_text depuis {candidates}. "
        f"Assurez-vous que votre projet est sur PYTHONPATH. Dernière erreur: {last_err}"
    )

def import_tab_eval(tab_repo_root: str):
    add_to_syspath(tab_repo_root)
    try:
        import evaluation as tab_eval
        return tab_eval
    except Exception as e:
        raise ImportError(
            f"Impossible d'importer evaluation.py depuis {tab_repo_root}. "
            f"Vérifiez le chemin vers le repo TAB. Erreur: {e}"
        )

def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = []
    cur_start, cur_end = spans[0]
    for s, e in spans[1:]:
        if s <= cur_end:  # chevauchement ou contiguïté
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged

def spans_from_diff(original: str, anonymized: str) -> List[Tuple[int, int]]:
    sm = SequenceMatcher(a=original, b=anonymized, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            if i2 > i1:
                spans.append((i1, i2))
    return merge_spans(spans)

def spans_from_audit(result: dict) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    audit = (result or {}).get("audit", {}) or {}
    for e in audit.get("entities", []):
        if e.get("source") == "policy-generalization":
            continue
        if str(e.get("etype") or e.get("type") or "").upper() == "DATE":
            continue
        if e.get("replacement"):
            s = e.get("start") if isinstance(e.get("start"), int) else e.get("start_offset")
            e2 = e.get("end") if isinstance(e.get("end"), int) else e.get("end_offset")
            if isinstance(s, int) and isinstance(e2, int) and e2 > s:
                spans.append((s, e2))
    return merge_spans(spans)

def extract_gold_spans_from_doc(doc: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Extraction robuste des spans gold à partir des structures TAB/ECHR:
    - annotations -> (quality_checked ou tous les annotateurs) -> entity_mentions[*]
      avec start_offset / end_offset
    - Exclut par défaut les mentions 'NO_MASK'
    - Conserve aussi les chemins "legacy" (entities, mentions, spans, etc.)
    """
    spans: List[Tuple[int, int]] = []
    INCLUDE_NO_MASK = False  # passer à True si on veut inclure NO_MASK

    def add(s, e):
        try:
            s = int(s); e = int(e)
        except (TypeError, ValueError):
            return
        if 0 <= s < e:
            spans.append((s, e))

    # 1) Parcours spécifique annotations -> entity_mentions
    anns = doc.get("annotations")
    if isinstance(anns, dict):
        qc = doc.get("quality_checked")
        annotator_keys = [k for k in qc if k in anns] if isinstance(qc, list) and qc else list(anns.keys())
        for ak in annotator_keys:
            a = anns.get(ak)
            if not isinstance(a, dict):
                continue
            ems = a.get("entity_mentions")
            if not isinstance(ems, list):
                continue
            for it in ems:
                if not isinstance(it, dict):
                    continue
                idt = str(it.get("identifier_type") or "").upper()
                if not INCLUDE_NO_MASK and idt == "NO_MASK":
                    continue
                s = it.get("start_offset") or it.get("start") or it.get("begin")
                e = it.get("end_offset")   or it.get("end")   or it.get("stop") or it.get("finish")
                if s is not None and e is not None:
                    add(s, e)

    # 2) Fallback des clés usuelles
    candidate_keys = [
        "entities", "mentions", "gold_mentions",
        "labels", "annotations",  # si annotations dict déjà traité ci-dessus
        "spans", "masked_spans", "gold_spans",
        "targets", "gold",
    ]
    for key in candidate_keys:
        obj = doc.get(key)
        if not obj:
            continue
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    span = it.get("span") or {}
                    s = (it.get("start") or it.get("start_offset") or span.get("start") or it.get("char_start") or it.get("begin"))
                    e = (it.get("end") or it.get("end_offset") or span.get("end") or it.get("char_end") or it.get("stop") or it.get("finish"))
                    if s is not None and e is not None:
                        add(s, e)
                elif isinstance(it, (list, tuple)) and len(it) == 2:
                    add(it[0], it[1])
        elif isinstance(obj, dict):
            nested = obj.get("spans") or obj.get("gold_spans")
            if isinstance(nested, list):
                for sp in nested:
                    if isinstance(sp, dict):
                        add(sp.get("start"), sp.get("end"))
                    elif isinstance(sp, (list, tuple)) and len(sp) == 2:
                        add(sp[0], sp[1])

    return sorted(set(spans))

def get_gold_spans_map(annotated_docs: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
    m = {}
    for d in annotated_docs:
        doc_id = d.get("doc_id")
        if not doc_id:
            continue
        m[doc_id] = extract_gold_spans_from_doc(d)
    return m

def exact_matches(gold: List[Tuple[int,int]], pred: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    gset = set(gold); pset = set(pred)
    return sorted(gset.intersection(pset))

def overlap_chars(gold: List[Tuple[int,int]], pred: List[Tuple[int,int]]) -> int:
    total = 0; i = 0; j = 0
    gold = sorted(gold); pred = sorted(pred)
    while i < len(gold) and j < len(pred):
        gs, ge = gold[i]; ps, pe = pred[j]
        start = max(gs, ps); end = min(ge, pe)
        if end > start: total += (end - start)
        if ge < pe: i += 1
        else: j += 1
    return total

def render_highlight_html(text: str, gold: List[Tuple[int,int]], pred: List[Tuple[int,int]]) -> str:
    n = len(text); lab = [0]*n
    for s,e in gold:
        s = max(0,min(n,s)); e = max(0,min(n,e))
        for k in range(s,e): lab[k] |= 1
    for s,e in pred:
        s = max(0,min(n,s)); e = max(0,min(n,e))
        for k in range(s,e): lab[k] |= 2
    def span_style(code):
        if code == 1: return 'background: rgba(46, 204, 113, 0.35);'
        if code == 2: return 'background: rgba(52, 152, 219, 0.35);'
        if code == 3: return 'background: rgba(155, 89, 182, 0.45);'
        return ''
    out = []; i = 0
    while i < n:
        code = lab[i]; j = i+1
        while j < n and lab[j] == code: j += 1
        chunk = html.escape(text[i:j])
        if code: out.append(f'<span style="{span_style(code)}">{chunk}</span>')
        else: out.append(chunk)
        i = j
    legend = """
    <div style="font-family: sans-serif; margin-bottom: 8px;">
      <span style="background: rgba(46, 204, 113, 0.35); padding:2px 4px; border-radius:3px;">Gold</span>
      <span style="background: rgba(52, 152, 219, 0.35); padding:2px 4px; border-radius:3px; margin-left:8px;">Prédits</span>
      <span style="background: rgba(155, 89, 182, 0.45); padding:2px 4px; border-radius:3px; margin-left:8px;">Recouvrement</span>
    </div>
    """
    return legend + '<div style="white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;">' + ''.join(out) + '</div>'

def evaluate_pipeline_on_tab(
    tab_repo_root: str,
    gold_json_path: str,
    level: str = "L1",
    split_filter: str = None,
    max_docs: int = None,
    save_preds_path: str = None,
    rate_limit_s: float = 0.0,
    prefer_audit_spans: bool = False,
    disable_paraphrase: bool = True,
    secret_salt: str = "tab_eval_secret",
    scope_mode: str = "doc_id",
    dump_dir: str = None,
    dump_k: int = 20,
    dump_strategy: str = "errors",
    use_gliner_best: bool = True,
    disable_internal_ner: bool = False,
):
    load_dotenv()
    anonymize_text = import_anonymize()
    tab_eval = import_tab_eval(tab_repo_root)

    with open(gold_json_path, "r", encoding="utf-8") as f:
        annotated_docs = json.load(f)

    if split_filter:
        annotated_docs = [d for d in annotated_docs if d.get("dataset_type") == split_filter]
    if max_docs:
        annotated_docs = annotated_docs[:max_docs]

    print(f"Documents à traiter: {len(annotated_docs)} (level={level}, split={split_filter})")

    has_llm = bool(os.getenv("OPENROUTER_API_KEY"))
    if level == "L1" and not has_llm:
        print("Aucune clé OPENROUTER_API_KEY trouvée: passage en mode sans LLM (niveau L0 effectif).")

    preds: Dict[str, List[Tuple[int, int]]] = {}
    anon_texts: Dict[str, str] = {}
    originals: Dict[str, str] = {}

    base_overrides = {
        "llm_paraphrase": False if disable_paraphrase else True,
        "date_granularity": "none",  # stabilité offsets
    }
    if disable_internal_ner:
        base_overrides["disable_internal_ner"] = True
    else:
        # Activer GLiNER best si demandé et pas déjà configuré via env
        if use_gliner_best and not os.getenv("GLINER_PRESET"):
            base_overrides.setdefault("ner_use_gliner", True)
            base_overrides.setdefault("gliner_models", BEST_GLINER_MODELS)

    for doc in tqdm(annotated_docs, desc="Anonymisation"):
        doc_id = doc["doc_id"]
        text = doc["text"]
        originals[doc_id] = text
        scope_id = doc_id if scope_mode == "doc_id" else scope_mode

        try:
            res = anonymize_text(
                value=text,
                scope_id=scope_id,
                secret_salt=secret_salt,
                level=level,
                ner_results=[],  # laisser l'orchestrateur gérer l'ensemble
                overrides=base_overrides,
            )
        except TypeError:
            res = anonymize_text(text, scope_id, secret_salt, level=level, ner_results=[], overrides=base_overrides)
        except Exception as e:
            print(f"Erreur anonymisation doc_id={doc_id}: {e}")
            continue

        spans = []
        if prefer_audit_spans:
            try:
                spans = spans_from_audit(res)
            except Exception:
                spans = []
        if not spans:
            anon_text = res.get("text") if isinstance(res, dict) else None
            if not isinstance(anon_text, str):
                raise RuntimeError("Le résultat du pipeline ne contient pas 'text' (chaîne).")
            spans = spans_from_diff(text, anon_text)
            anon_texts[doc_id] = anon_text
        else:
            anon_texts[doc_id] = (res.get("text") if isinstance(res, dict) else None) or ""
        preds[doc_id] = [(int(s), int(e)) for (s, e) in spans]

        if rate_limit_s > 0:
            time.sleep(rate_limit_s)

    if save_preds_path:
        save_path = Path(save_preds_path).expanduser().resolve()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in preds.items()}, f)
        print(f"Prédictions sauvegardées: {save_path}")

    gold_corpus = tab_eval.GoldCorpus(gold_json_path)
    masked_docs = [tab_eval.MaskedDocument(doc_id=k, masked_spans=v) for k, v in preds.items()]
    measures = tab_eval.evaluate(
        gold_corpus=gold_corpus,
        masked_docs=masked_docs,
        use_bert=False,
        verbose=False
    )

    print("======== Résultats (TAB) ========")
    for k, v in measures.items():
        print(f"{k}: {v}")

    if dump_dir:
        dump_root = Path(dump_dir).expanduser().resolve()
        (dump_root / "html").mkdir(parents=True, exist_ok=True)
        jsonl_path = dump_root / "samples.jsonl"
        index_path = dump_root / "html" / "index.html"
        gold_map = get_gold_spans_map(annotated_docs)
        missing = [d["doc_id"] for d in annotated_docs if not gold_map.get(d["doc_id"])]
        if missing:
            print(f"AVERTISSEMENT: {len(missing)} doc(s) sans gold extrait(s), ex: {missing[:5]}")
        per_doc_stats = []
        for doc in annotated_docs:
            doc_id = doc["doc_id"]
            g = gold_map.get(doc_id, [])
            p = preds.get(doc_id, [])
            em = exact_matches(g, p)
            ov = overlap_chars(g, p)
            per_doc_stats.append({
                "doc_id": doc_id,
                "num_gold": len(g),
                "num_pred": len(p),
                "num_exact": len(em),
                "overlap_chars": ov
            })
        selected_ids = []
        if dump_strategy == "errors":
            err = [s for s in per_doc_stats if s["num_gold"] > 0 and s["num_exact"] == 0]
            err.sort(key=lambda x: (x["overlap_chars"], x["num_pred"]), reverse=True)
            selected_ids = [s["doc_id"] for s in err[:dump_k]]
            if len(selected_ids) < dump_k:
                rest = [s for s in per_doc_stats if s["doc_id"] not in selected_ids]
                selected_ids += [s["doc_id"] for s in rest[: (dump_k - len(selected_ids))]]
        elif dump_strategy == "first":
            selected_ids = [d["doc_id"] for d in annotated_docs[:dump_k]]
        elif dump_strategy == "random":
            all_ids = [d["doc_id"] for d in annotated_docs]
            random.shuffle(all_ids)
            selected_ids = all_ids[:dump_k]
        else:
            selected_ids = [d["doc_id"] for d in annotated_docs[:dump_k]]

        with open(jsonl_path, "w", encoding="utf-8") as jf:
            pass
        links = []
        for doc_id in selected_ids:
            text = originals.get(doc_id, "")
            anon = anon_texts.get(doc_id, "")
            g = gold_map.get(doc_id, [])
            p = preds.get(doc_id, [])
            em = exact_matches(g, p)
            rec = {
                "doc_id": doc_id,
                "text": text,
                "anonymized_text": anon,
                "gold_spans": g,
                "pred_spans": p,
                "exact_matches": em
            }
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            html_body = f"""
            <h3>Doc: {html.escape(doc_id)}</h3>
            <p><strong>Gold mentions:</strong> {len(g)} | <strong>Pred:</strong> {len(p)} | <strong>Exact:</strong> {len(em)}</p>
            <h4>Texte original (surlignage)</h4>
            {render_highlight_html(text, g, p)}
            <h4>Texte anonymisé</h4>
            <div style=\"white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;\">{html.escape(anon)}</div>
            """
            page = f"""<!doctype html>
            <html><head><meta charset=\"utf-8\"><title>{html.escape(doc_id)}</title></head>
            <body style=\"margin:16px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;\">
            {html_body}
            </body></html>
            """
            page_path = dump_root / "html" / f"{doc_id}.html"
            with open(page_path, "w", encoding="utf-8") as f:
                f.write(page)
            links.append(f'<li><a href="{html.escape(page_path.name)}">{html.escape(doc_id)}</a></li>')
        index = f"""<!doctype html>
        <html><head><meta charset=\"utf-8\"><title>Samples</title></head>
        <body style=\"margin:16px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;\">
        <h3>Échantillons ({len(selected_ids)})</h3>
        <ul>
        {''.join(links)}
        </ul>
        <p>JSONL: {html.escape(str(jsonl_path))}</p>
        </body></html>
        """
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index)
        print(f"Dump écrit dans: {dump_root} (HTML: {index_path}, JSONL: {jsonl_path})")

    return measures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue le pipeline d'anonymisation sur TAB (avec dump)")
    parser.add_argument("--tab_repo", type=str, required=True, help="Chemin vers le repo TAB (racine contenant evaluation.py)")
    parser.add_argument("--gold_json", type=str, required=True, help="Chemin vers echr_train/dev/test.json")
    parser.add_argument("--level", type=str, default="L1", choices=["L0", "L1", "L2"])  # L2 possible
    parser.add_argument("--split", type=str, default=None, choices=["train", "dev", "test"])
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--save_preds", type=str, default="preds_tab.json")
    parser.add_argument("--rate_limit_s", type=float, default=0.0)
    parser.add_argument("--prefer_audit_spans", action="store_true")
    parser.add_argument("--no_paraphrase", action="store_true")
    parser.add_argument("--secret_salt", type=str, default="tab_eval_secret")
    parser.add_argument("--scope_mode", type=str, default="doc_id", choices=["doc_id", "fixed"])
    parser.add_argument("--dump_dir", type=str, default=None, help="Dossier de sortie (active le dump HTML/JSONL)")
    parser.add_argument("--dump_k", type=int, default=20, help="Nombre d’échantillons à sauvegarder")
    parser.add_argument("--dump_strategy", type=str, default="errors", choices=["errors", "first", "random"], help="Stratégie de sélection")
    parser.add_argument("--no_gliner_best", action="store_true", help="Désactive l'injection automatique du preset GLiNER best")
    parser.add_argument("--disable_internal_ner", action="store_true", help="Désactive entièrement le NER interne (pour tests ablation)")
    args = parser.parse_args()

    evaluate_pipeline_on_tab(
        tab_repo_root=args.tab_repo,
        gold_json_path=args.gold_json,
        level=args.level,
        split_filter=args.split,
        max_docs=args.max_docs,
        save_preds_path=args.save_preds,
        rate_limit_s=args.rate_limit_s,
        prefer_audit_spans=args.prefer_audit_spans,
        disable_paraphrase=args.no_paraphrase,
        secret_salt=args.secret_salt,
        scope_mode=args.scope_mode,
        dump_dir=args.dump_dir,
        dump_k=args.dump_k,
        dump_strategy=args.dump_strategy,
        use_gliner_best=not args.no_gliner_best,
        disable_internal_ner=args.disable_internal_ner,
    )