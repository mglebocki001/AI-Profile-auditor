import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# 1. STRUKTURA RAPORTU AUDYTU (Pydantic Schema)
# ---------------------------------------------------------------------------

class AuditIssue(BaseModel):
    section: str  # kategoria reguły, np. "Ekspozycja Twarzy", "Opis profilu"
    rule_name: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    detected_in: str
    impact_or_reason: str
    recommendation: str
    verified_by_human: bool = False
    relevance_score: float = 0.0  # Końcowa waga dopasowania reguły
    matched_by_keywords: bool = False  # Czy dopasowanie wsparte było twardym trafieniem słów kluczowych

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
    """
    Twardy sygnał leksykalny — niezależny od jakości embeddingu.
    Liczy ile fraz kluczowych reguły dosłownie występuje w tekście zapytania
    (case-insensitive). 2+ trafienia = maksymalny wynik.
    Działa jako "kotwica", która ratuje trafne dopasowania w sytuacji,
    gdy model embeddingowy słabo radzi sobie z polskim tekstem.
    """
    if not trigger_keywords_joined:
        return 0.0

    keywords = [kw.strip().lower() for kw in trigger_keywords_joined.split("|||") if kw.strip()]
    if not keywords:
        return 0.0

    query_lower = query_text.lower()
    hits = sum(1 for kw in keywords if kw in query_lower)

    return min(1.0, hits / 2.0)


# ---------------------------------------------------------------------------
# 3. SILNIK AUDYTU Z HYBRYDOWYM MATCHINGIEM WEKTOROWYM
# ---------------------------------------------------------------------------

class ProfileAnalyzerEngine:
    # Próg minimalnej trafności (semantic LUB keyword) poniżej którego
    # dopasowanie jest odrzucane, niezależnie od bonusów. To eliminuje
    # przypadkowe, "odklejone" dopasowania w raporcie.
    MIN_RELEVANCE_THRESHOLD = 0.42

    # Wagi finalnego rankingu — semantyka dominuje, ale keyword overlap
    # jest teraz osobnym, twardym sygnałem, a nie tylko dodatkiem do similarity.
    WEIGHT_SEMANTIC = 0.60
    WEIGHT_KEYWORD = 0.20
    WEIGHT_HUMAN_VERIFIED = 0.10
    WEIGHT_PRIORITY = 0.10

    def __init__(self, rules_db_path: str = "rules.json"):
        self.rules_db_path = Path(rules_db_path)

        # Inicjalizacja bazy ChromaDB
        self.chroma_client = chromadb.Client()

        # --- (B) MULTILINGUAL EMBEDDING FUNCTION -----------------------
        # Domyślny embedding Chroma (all-MiniLM-L6-v2) jest trenowany
        # głównie na angielskim i dla polskich zdań generuje głównie szum
        # leksykalny zamiast prawdziwego podobieństwa semantycznego.
        # Zamieniamy go na model wielojęzyczny.
        #
        # Wymaga: pip install sentence-transformers
        # Model pobiera się (i cache'uje lokalnie) przy pierwszym użyciu,
        # więc pierwsze uruchomienie wymaga dostępu do internetu.
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-mpnet-base-v2"
        )

        # Tworzenie lub pobieranie kolekcji z resetem pamięci podręcznej
        self.verified_collection = self.chroma_client.get_or_create_collection(
            name="dating_rules_verified",
            embedding_function=self.embedding_function,
        )
        self.unverified_collection = self.chroma_client.get_or_create_collection(
            name="dating_rules_unverified",
            embedding_function=self.embedding_function,
        )

        # Czyścimy stare kolekcje przy tworzeniu instancji, zapobiegając dublowaniu reguł
        if self.verified_collection.count() > 0:
            self.chroma_client.delete_collection("dating_rules_verified")
            self.verified_collection = self.chroma_client.get_or_create_collection(
                name="dating_rules_verified", embedding_function=self.embedding_function
            )

        if self.unverified_collection.count() > 0:
            self.chroma_client.delete_collection("dating_rules_unverified")
            self.unverified_collection = self.chroma_client.get_or_create_collection(
                name="dating_rules_unverified", embedding_function=self.embedding_function
            )

        self._verified_count = 0
        self._unverified_count = 0

        if self.rules_db_path.exists():
            self._build_vector_database()
        else:
            print(f"❌ OSTRZEŻENIE: Brak pliku '{rules_db_path}' w katalogu roboczym!")

    def _build_vector_database(self):
        """Wczytuje rules.json i indeksuje reguły do odpowiednich kolekcji."""
        with open(self.rules_db_path, "r", encoding="utf-8") as f:
            raw_rules = json.load(f)

        rules_list = raw_rules.get("rules", []) if isinstance(raw_rules, dict) else raw_rules

        verified_docs, verified_metas, verified_ids = [], [], []
        unverified_docs, unverified_metas, unverified_ids = [], [], []

        for rule in rules_list:
            rule_id = str(rule.get("id", ""))
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

            metadata = {
                "section": category,
                "rule_name": title,
                "severity": _priority_score_to_severity(priority_score),
                "priority_score": priority_score,
                "impact": impact_text,
                "recommendation": recommendation,
                "verified_by_human": is_verified,
                "positive_impact": rule.get("positive_impact", ""),
                # (D) Zapisujemy oryginalne frazy kluczowe (jako string, bo
                # metadata w Chroma musi być typem prostym), żeby móc policzyć
                # twardy keyword overlap niezależnie od embeddingu.
                "trigger_keywords": "|||".join(keywords_list),
            }

            if is_verified:
                verified_docs.append(doc_text)
                verified_metas.append(metadata)
                verified_ids.append(rule_id or f"verified_rule_{len(verified_ids)}")
            else:
                unverified_docs.append(doc_text)
                unverified_metas.append(metadata)
                unverified_ids.append(rule_id or f"unverified_rule_{len(unverified_ids)}")

        if verified_docs:
            self.verified_collection.add(
                documents=verified_docs,
                metadatas=verified_metas,
                ids=verified_ids
            )
        if unverified_docs:
            self.unverified_collection.add(
                documents=unverified_docs,
                metadatas=unverified_metas,
                ids=unverified_ids
            )

        self._verified_count = len(verified_docs)
        self._unverified_count = len(unverified_docs)

        print(
            f"📖 Załadowano z '{self.rules_db_path}': "
            f"{self._verified_count} reguł zweryfikowanych oraz {self._unverified_count} niezweryfikowanych (AI)."
        )

    def _query_collection_with_scores(self, collection, query_text: str, n_results: int) -> List[Dict[str, Any]]:
        """Pobiera z bazy metadane wraz z obliczonym podobieństwem semantycznym."""
        if n_results <= 0 or collection.count() == 0:
            return []

        safe_n = min(n_results, collection.count())
        results = collection.query(
            query_texts=[query_text],
            n_results=safe_n,
            include=["metadatas", "distances"]
        )

        matched_list = []
        if results and "metadatas" in results and results["metadatas"]:
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [1.0] * len(metas)

            for meta, dist in zip(metas, distances):
                # Konwersja dystansu kosinusowego/L2 na wartość podobieństwa [0.0 - 1.0]
                similarity = max(0.0, 1.0 - (dist / 2.0))
                item = dict(meta)
                item["similarity"] = similarity
                matched_list.append(item)

        return matched_list

    def _search_hybrid(self, query_text: str, candidate_k: int = 5) -> List[Dict[str, Any]]:
        """
        Przeszukuje OBYDWIE kolekcje naraz.
        Bierze pod uwagę:
        1. Zgodność semantyczną (similarity) — wielojęzyczny embedding (B)
        2. Twardy sygnał ze słów kluczowych (D) — ratuje trafne dopasowania,
           gdy embedding zawiedzie
        3. Lekki bonus za weryfikację przez człowieka
        4. Priority score reguły
        Kandydaci poniżej MIN_RELEVANCE_THRESHOLD (semantic LUB keyword)
        są odrzucani (A/C) — to eliminuje "odklejone" dopasowania.
        """
        verified_matches = self._query_collection_with_scores(self.verified_collection, query_text, candidate_k)
        unverified_matches = self._query_collection_with_scores(self.unverified_collection, query_text, candidate_k)

        all_candidates = verified_matches + unverified_matches
        filtered_candidates = []

        for cand in all_candidates:
            sim = cand.get("similarity", 0.0)
            keyword_score = _keyword_overlap_score(query_text, cand.get("trigger_keywords", ""))
            is_verified = cand.get("verified_by_human", False)
            p_score = cand.get("priority_score", 50) / 100.0

            # (C) Twardy próg: odrzucamy kandydatów, którzy nie mają ani
            # sensownego podobieństwa semantycznego, ani trafienia w słowa
            # kluczowe. Bonusy (human/priority) NIE mogą już "wciągnąć"
            # niezwiązanej reguły do wyników — liczą się tylko przy remisach
            # między już-trafnymi kandydatami.
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

            cand["keyword_score"] = keyword_score
            cand["matched_by_keywords"] = keyword_score >= 0.5
            cand["final_match_rank"] = final_match_rank
            filtered_candidates.append(cand)

        # Sortujemy od najlepszego wyniku całkowitego w dół
        filtered_candidates.sort(key=lambda x: x["final_match_rank"], reverse=True)
        return filtered_candidates

    def analyze_extracted_json(self, profile_data_path: str, target_rules_count: int = 5) -> ProfileAuditReport:
        """
        Dla KAŻDEJ wykrytej cechy (feature) z audytu OCR wybiera JEJ NAJLEPSZE
        dopasowanie w bazie reguł (A) — a nie wspólną pulę top-3 ze wszystkich
        cech naraz. Dzięki temu `detected_in` zawsze wskazuje realne źródło
        problemu, a nie przypadkowego "zwycięzcę" wyścigu score'ów między
        niepowiązanymi cechami.

        Dopiero na końcu, jeśli ta sama reguła została realnie i trafnie
        wykryta dla więcej niż jednej cechy, zostawiamy tylko najlepsze
        wystąpienie (ale oba kandydaci już przeszły próg trafności, więc
        to nie jest "podkradanie" wyniku przez niepowiązaną cechę).
        """
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

            # Pobieramy kandydatów już PRZEFILTROWANYCH progiem trafności
            matches = self._search_hybrid(query_text, candidate_k=5)

            if not matches:
                # Brak trafnej reguły dla tej cechy — świadomie pomijamy,
                # zamiast wymuszać dopasowanie na siłę (to był źródłowy błąd).
                continue

            # (A) Bierzemy TYLKO najlepsze dopasowanie DLA TEJ KONKRETNEJ cechy.
            # `detected_in` jest więc zawsze poprawnie przypięty do cechy,
            # z której zapytanie faktycznie pochodziło.
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

        # 1. Deduplikacja po unikalnej nazwie reguły — teraz bezpieczna, bo
        #    wszyscy kandydaci już przeszli próg trafności per-cecha (A+C),
        #    więc wybór "lepszego" wystąpienia nie podmienia detected_in
        #    na coś niepowiązanego.
        unique_rules_map: Dict[str, Dict[str, Any]] = {}
        for cand in detected_candidates:
            rule_key = f"{cand['section']}_{cand['rule_name']}"
            if rule_key not in unique_rules_map or cand["final_match_rank"] > unique_rules_map[rule_key]["final_match_rank"]:
                unique_rules_map[rule_key] = cand

        # 2. Sortowanie globalne wszystkich unikalnych dopasowań
        sorted_candidates = sorted(list(unique_rules_map.values()), key=lambda x: x["final_match_rank"], reverse=True)

        # 3. Przycięcie do żądanych ZASAD (może być mniej niż target_rules_count,
        #    jeśli po odfiltrowaniu szumu nie ma tylu trafnych dopasowań —
        #    to prawidłowe zachowanie, lepsze niż wymuszanie 5 wyników na siłę).
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

        # Wyliczanie punktacji ogólnej
        penalty_map = {"CRITICAL": 20, "HIGH": 12, "MEDIUM": 8, "LOW": 4}
        total_penalty = sum(penalty_map.get(issue.severity, 8) for issue in final_issues)
        overall_score = max(0.0, min(100.0, 100.0 - total_penalty))
        critical_count = sum(1 for i in final_issues if i.severity == "CRITICAL")

        return ProfileAuditReport(
            profile_name=doc_title,
            total_issues=len(final_issues),
            critical_issues_count=critical_count,
            overall_score=overall_score,
            detected_issues=final_issues,
            positive_highlights=positive_highlights
        )


# ---------------------------------------------------------------------------
# 4. FORMATOWANY WYDRUK RAPORTU
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
# 5. EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EXTRACTED_FOLDER = Path("./OCR_result")
    REPORTS_FOLDER = Path("./audit_reports")
    REPORTS_FOLDER.mkdir(exist_ok=True)

    analyzer = ProfileAnalyzerEngine(rules_db_path="rules.json")
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
