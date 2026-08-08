import gc
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. STRUKTURA RAPORTU AUDYTU (Pydantic Schema) — bez zmian
# ---------------------------------------------------------------------------

class AuditIssue(BaseModel):
    section: str
    rule_name: str
    severity: str
    detected_in: str
    impact_or_reason: str
    recommendation: str
    verified_by_human: bool = False
    relevance_score: float = 0.0
    matched_by_keywords: bool = False

class ProfileAuditReport(BaseModel):
    profile_name: str
    total_issues: int
    critical_issues_count: int
    overall_score: float
    detected_issues: List[AuditIssue]
    positive_highlights: List[str]


# ---------------------------------------------------------------------------
# 2. HELPERY I MAPOWANIA
# ---------------------------------------------------------------------------

def _priority_score_to_severity(score: int) -> str:
    if score >= 80:
        return "CRITICAL"
    elif score >= 60:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def _keyword_overlap_score(query_text: str, trigger_keywords_joined: str) -> float:
    """Twardy sygnał leksykalny — bez zmian względem oryginału."""
    if not trigger_keywords_joined:
        return 0.0

    keywords = [kw.strip().lower() for kw in trigger_keywords_joined.split("|||") if kw.strip()]
    if not keywords:
        return 0.0

    query_lower = query_text.lower()
    hits = sum(1 for kw in keywords if kw in query_lower)

    return min(1.0, hits / 2.0)


# ---------------------------------------------------------------------------
# 3. LEKKI INDEKS TF-IDF (ZASTĘPUJE ChromaDB + sentence-transformers/PyTorch)
# ---------------------------------------------------------------------------
#
# UWAGA DOT. JAKOŚCI DOPASOWANIA (świadomy kompromis RAM vs jakość):
# Poprzedni silnik używał modelu "paraphrase-multilingual-mpnet-base-v2"
# (transformer, setki MB parametrów + PyTorch w RAM), który dobrze radzi
# sobie z parafrazami ("brak uśmiechu" ~ "poważna mina"). TF-IDF jest
# metodą LEKSYKALNĄ — dobrze łapie dosłowne/pokrewne słowa, gorzej czyste
# parafrazy bez wspólnych rdzeni słów. Dlatego próg trafności i wagi
# poniżej są przeliczone tak, by silniej opierać się na sygnale keyword
# (Audit.py i tak już miał "kotwicę" słów kluczowych na wypadek słabego
# embeddingu — tutaj ta kotwica dźwiga więcej). Jeśli po testach na
# realnych danych jakość dopasowań okaże się niewystarczająca, zalecany
# upgrade to embeddingi przez Gemini API (`client.models.embed_content`)
# zamiast lokalnego modelu — to nadal nie obciąża RAM-u serwera, bo
# liczenie odbywa się zdalnie, a tutaj trzeba by tylko cache'ować wektory
# na dysku. Nie wdrożyłem tego domyślnie, żeby nie dodawać zależności
# sieciowej do KAŻDEGO zapytania audytu bez Twojej zgody.

_TOKEN_RE = re.compile(r"[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class _TfidfIndex:
    """Indeks TF-IDF działający wyłącznie na stdlib. Dla ~600 krótkich
    dokumentów reguł zajmuje pojedyncze MB RAM zamiast setek MB wymaganych
    przez lokalny model embeddingowy + PyTorch."""

    def __init__(self):
        self._idf: Dict[str, float] = {}
        self._doc_vectors: List[Dict[str, float]] = []
        self._doc_norms: List[float] = []

    def build(self, docs: List[str]):
        n_docs = len(docs)
        tokenized_docs = [_tokenize(d) for d in docs]

        doc_freq = Counter()
        for tokens in tokenized_docs:
            for term in set(tokens):
                doc_freq[term] += 1

        self._idf = {
            term: math.log((n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in doc_freq.items()
        }

        self._doc_vectors = []
        self._doc_norms = []
        for tokens in tokenized_docs:
            vec = self._vectorize(tokens)
            norm = math.sqrt(sum(w * w for w in vec.values())) or 1e-9
            self._doc_vectors.append(vec)
            self._doc_norms.append(norm)

        del tokenized_docs
        gc.collect()

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        length = len(tokens)
        return {
            term: (count / length) * self._idf.get(term, 0.0)
            for term, count in tf.items()
        }

    def query(self, text: str) -> List[float]:
        """Podobieństwo cosinusowe zapytania do WSZYSTKICH dokumentów,
        w kolejności indeksowania (lista, nie generator — potrzebujemy
        wielokrotnego zip() z metadanymi w warstwie wyżej)."""
        if not self._doc_vectors:
            return []
        qvec = self._vectorize(_tokenize(text))
        qnorm = math.sqrt(sum(w * w for w in qvec.values())) or 1e-9

        similarities = []
        for dvec, dnorm in zip(self._doc_vectors, self._doc_norms):
            small, big = (qvec, dvec) if len(qvec) < len(dvec) else (dvec, qvec)
            dot = sum(weight * big.get(term, 0.0) for term, weight in small.items())
            similarities.append(dot / (qnorm * dnorm))
        return similarities


# ---------------------------------------------------------------------------
# 4. SILNIK AUDYTU (interfejs publiczny bez zmian — analyze_extracted_json
#    ma dokładnie ten sam kontrakt co poprzednio)
# ---------------------------------------------------------------------------

class ProfileAnalyzerEngine:
    MIN_RELEVANCE_THRESHOLD = 0.15  # przeliczone pod TF-IDF, patrz komentarz wyżej

    WEIGHT_SEMANTIC = 0.45
    WEIGHT_KEYWORD = 0.35
    WEIGHT_HUMAN_VERIFIED = 0.10
    WEIGHT_PRIORITY = 0.10

    def __init__(self, rules_db_path: str = "rules.json"):
        self.rules_db_path = Path(rules_db_path)
        self._index = _TfidfIndex()
        self._rule_metas: List[Dict[str, Any]] = []
        self._loaded = False

        if self.rules_db_path.exists():
            self._build_index()
        else:
            print(f"❌ OSTRZEŻENIE: Brak pliku '{rules_db_path}' w katalogu roboczym!")

    def _build_index(self):
        with open(self.rules_db_path, "r", encoding="utf-8") as f:
            raw_rules = json.load(f)

        rules_list = raw_rules.get("rules", []) if isinstance(raw_rules, dict) else raw_rules
        del raw_rules

        docs: List[str] = []
        self._rule_metas = []

        for rule in rules_list:
            title = rule.get("title", "")
            category = rule.get("category", "")
            keywords_list = rule.get("trigger_keywords", [])
            keywords_str = " ".join(keywords_list)
            priority_score = rule.get("priority_score", 50)
            is_verified = bool(rule.get("verified_by_human", False))

            doc_text = (
                f"{category}: {title}. "
                f"Słowa kluczowe: {keywords_str}. "
                f"Powód: {rule.get('why_it_matters', '')}"
            )

            recommendation = rule.get("quick_fix", "")
            impact_text = rule.get("negative_impact") or rule.get("why_it_matters", "")

            docs.append(doc_text)
            self._rule_metas.append({
                "section": category,
                "rule_name": title,
                "severity": _priority_score_to_severity(priority_score),
                "priority_score": priority_score,
                "impact": impact_text,
                "recommendation": recommendation,
                "verified_by_human": is_verified,
                "trigger_keywords": "|||".join(keywords_list),
            })

        self._index.build(docs)
        self._loaded = True

        print(f"📖 Załadowano z '{self.rules_db_path}': {len(docs)} reguł (indeks TF-IDF, bez PyTorch/ChromaDB).")

        del docs, rules_list
        gc.collect()

    def _search_hybrid(self, query_text: str, candidate_k: int = 5) -> List[Dict[str, Any]]:
        if not self._loaded:
            return []

        similarities = self._index.query(query_text)
        filtered_candidates = []

        for meta, sim in zip(self._rule_metas, similarities):
            keyword_score = _keyword_overlap_score(query_text, meta.get("trigger_keywords", ""))
            is_verified = meta.get("verified_by_human", False)
            p_score = meta.get("priority_score", 50) / 100.0

            relevance = max(sim, keyword_score)
            if relevance < self.MIN_RELEVANCE_THRESHOLD:
                continue

            human_bonus = self.WEIGHT_HUMAN_VERIFIED if is_verified else 0.0
            final_match_rank = (
                (sim * self.WEIGHT_SEMANTIC)
                + (keyword_score * self.WEIGHT_KEYWORD)
                + human_bonus
                + (p_score * self.WEIGHT_PRIORITY)
            )

            cand = dict(meta)
            cand["similarity"] = sim
            cand["keyword_score"] = keyword_score
            cand["matched_by_keywords"] = keyword_score >= 0.5
            cand["final_match_rank"] = final_match_rank
            filtered_candidates.append(cand)

        filtered_candidates.sort(key=lambda x: x["final_match_rank"], reverse=True)
        return filtered_candidates[:candidate_k]

    def analyze_extracted_json(self, profile_data_path: str, target_rules_count: int = 5) -> ProfileAuditReport:
        with open(profile_data_path, "r", encoding="utf-8") as f:
            profile_raw = json.load(f)

        doc_title = profile_raw.get("document_title", Path(profile_data_path).stem)
        extracted_rules = profile_raw.get("extracted_rules", [])
        one_great_thing = profile_raw.get("one_great_thing")

        detected_candidates: List[Dict[str, Any]] = []
        positive_highlights: List[str] = []

        if one_great_thing:
            positive_highlights.append(f"Główny atut: {one_great_thing}")

        for feature in extracted_rules:
            feat_name = str(feature.get("rule_name", ""))
            feat_desc = str(feature.get("full_description", ""))
            feat_look = " ".join([str(x) for x in feature.get("what_to_look_for", [])])

            query_text = f"{feat_name} {feat_desc} {feat_look}".strip()

            if not query_text:
                continue

            matches = self._search_hybrid(query_text, candidate_k=5)

            if not matches:
                continue

            best_match = matches[0]

            item = {
                "section": best_match["section"],
                "rule_name": best_match["rule_name"],
                "severity": best_match["severity"],
                "detected_in": feat_name or "Zdjęcie / Bio",
                "impact_or_reason": best_match["impact"],
                "recommendation": best_match["recommendation"],
                "verified_by_human": best_match["verified_by_human"],
                "matched_by_keywords": best_match.get("matched_by_keywords", False),
                "final_match_rank": best_match["final_match_rank"],
            }
            detected_candidates.append(item)

            if feature.get("positive_impact"):
                positive_highlights.append(f"{feat_name}: {feature['positive_impact']}")

        unique_rules_map: Dict[str, Dict[str, Any]] = {}
        for cand in detected_candidates:
            rule_key = f"{cand['section']}_{cand['rule_name']}"
            if rule_key not in unique_rules_map or cand["final_match_rank"] > unique_rules_map[rule_key]["final_match_rank"]:
                unique_rules_map[rule_key] = cand

        sorted_candidates = sorted(list(unique_rules_map.values()), key=lambda x: x["final_match_rank"], reverse=True)
        selected_rules = sorted_candidates[:target_rules_count]

        final_issues = [
            AuditIssue(
                section=c["section"],
                rule_name=c["rule_name"],
                severity=c["severity"],
                detected_in=c["detected_in"],
                impact_or_reason=c["impact_or_reason"],
                recommendation=c["recommendation"],
                verified_by_human=c["verified_by_human"],
                relevance_score=round(c["final_match_rank"], 2),
                matched_by_keywords=c["matched_by_keywords"],
            ) for c in selected_rules
        ]

        penalty_map = {"CRITICAL": 20, "HIGH": 12, "MEDIUM": 8, "LOW": 4}
        total_penalty = sum(penalty_map.get(issue.severity, 8) for issue in final_issues)
        overall_score = max(0.0, min(100.0, 100.0 - total_penalty))
        critical_count = sum(1 for i in final_issues if i.severity == "CRITICAL")

        report = ProfileAuditReport(
            profile_name=doc_title,
            total_issues=len(final_issues),
            critical_issues_count=critical_count,
            overall_score=overall_score,
            detected_issues=final_issues,
            positive_highlights=positive_highlights
        )

        del profile_raw, extracted_rules, detected_candidates, unique_rules_map, sorted_candidates
        gc.collect()

        return report


# ---------------------------------------------------------------------------
# 5. LENIWY SINGLETON SILNIKA
# ---------------------------------------------------------------------------
# Krytyczne dla RAM/CPU: silnik (i jego indeks TF-IDF) budowany jest RAZ,
# przy pierwszym użyciu — a nie przy KAŻDYM zapytaniu API, jak to było
# poprzednio (main.py tworzył `ProfileAnalyzerEngine(...)` wewnątrz
# endpointu, więc każde żądanie ładowało od nowa model i przebudowywało
# całą bazę wektorową). Dzięki temu serwer startuje natychmiast (health
# check na Renderze nie czeka na ciężką inicjalizację) i nie powtarza tej
# samej pracy w kółko.

_engine_singleton: Optional["ProfileAnalyzerEngine"] = None


def get_engine(rules_db_path: str = "rules.json") -> "ProfileAnalyzerEngine":
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = ProfileAnalyzerEngine(rules_db_path=rules_db_path)
    return _engine_singleton


# ---------------------------------------------------------------------------
# 6. FORMATOWANY WYDRUK RAPORTU — bez zmian
# ---------------------------------------------------------------------------

def print_audit_report(report: ProfileAuditReport):
    print("\n" + "="*65)
    print(f"📊 RAPORT AUDYTU (RAG / HYBRYDOWY): {report.profile_name}")
    print("="*65)
    print(f"📈 Wynik ogólny: {report.overall_score:.1f} / 100")
    print(f"❌ Znalezione błędy: {report.total_issues} (Krytycznych: {report.critical_issues_count})\n")

    if report.detected_issues:
        print("-----------------------------------------------------------")
        print(f"🚨 TOP {len(report.detected_issues)} DOPASOWANYCH REGUŁ / BŁĘDÓW:")
        print("-----------------------------------------------------------")
        for idx, issue in enumerate(report.detected_issues, 1):
            icon = "🔴" if issue.severity == "CRITICAL" else ("🟠" if issue.severity == "HIGH" else "🟡")
            zrodlo = "👤 Człowiek" if issue.verified_by_human else "🤖 AI"
            kw_flag = " 🔑" if issue.matched_by_keywords else ""
            print(f"{idx}. {icon} [{issue.severity}] {issue.section} ({zrodlo} | Wynik trafności: {issue.relevance_score}{kw_flag})")
            print(f"   Wykryta reguła: {issue.rule_name}")
            print(f"   Gdzie wykryto: {issue.detected_in}")
            print(f"   Dlaczego to szkodzi: {issue.impact_or_reason}")
            print(f"   💡 Jak naprawić: {issue.recommendation}\n")

    if report.positive_highlights:
        print("-----------------------------------------------------------")
        print("✅ MOCNE STRONY PROFILU:")
        print("-----------------------------------------------------------")
        for pos in report.positive_highlights:
            print(f"  • {pos}")
    print("="*65 + "\n")


# ---------------------------------------------------------------------------
# 7. EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EXTRACTED_FOLDER = Path("./OCR_result")
    REPORTS_FOLDER = Path("./audit_reports")
    REPORTS_FOLDER.mkdir(exist_ok=True)

    analyzer = get_engine(rules_db_path="rules.json")
    extracted_files = list(EXTRACTED_FOLDER.glob("*.json"))

    if not extracted_files:
        print(f"⚠️ Nie znaleziono plików JSON w katalogu: {EXTRACTED_FOLDER.resolve()}")
    else:
        print(f"🚀 Znaleziono {len(extracted_files)} analiz. Wyciągam do 5 najtrafniejszych reguł z bazy...\n")

        for file in extracted_files:
            audit = analyzer.analyze_extracted_json(str(file), target_rules_count=5)
            print_audit_report(audit)

            out_file = REPORTS_FOLDER / f"{file.stem}_audit_report.json"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(audit.model_dump_json(indent=2))
            print(f"📁 Raport zapisano w: {out_file.name}")
