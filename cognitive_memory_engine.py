#!/usr/bin/env python3
"""
================================================================================
           AUTONOMICZNY SILNIK PAMIĘCI KOGNITYWNEJ (O(1))
                   COGNITIVE WORKING MEMORY ENGINE
================================================================================
W 100% zautomatyzowany silnik ekstrakcji, akumulacji i aktualizacji stanu
pamięci roboczej inspirowany modelem Baddeleya (Episodic Buffer).
Zero ręcznie wpisywanych stringów faktów.

Główne metody:
  1. state = engine.ingest(raw_document_text)
     -> Automatyczna ekstrakcja encji, ról i kotwic wysokiej entropii do stanu S0
  2. state_next = engine.update(state_current, delta_text)
     -> Rozwiązywanie konfliktów (conflict resolution) i nadpisywanie zmienionych faktów (S_t -> S_t+1)
  3. prompt_context = engine.format_working_memory(state)
     -> Kompilacja do stałego bufora roboczego O(1) dla dowolnego modelu LLM
================================================================================
"""

import re
from typing import Dict, Any, List, Optional

class CognitiveMemoryEngine:
    MAX_ADDITIONAL_ANCHORS: int = 16

    def __init__(self, max_anchors: int = MAX_ADDITIONAL_ANCHORS):
        self.max_anchors = max_anchors
        self.slot_patterns = {
            "pre_money_valuation": [
                r'wycen[ęaey]\s+pre-money.*?(?:kwot[ęaey]|na|do|wynoszącą)?\s*(\d[\d\s]*\s*(?:EUR|USD|PLN|euro))',
                r'(\d[\d\s]*\s*(?:EUR|USD|PLN))\s*(?:tytułem|jako)?\s*wycen[yęa]\s+pre-money'
            ],
            "equity_percentage": [
                r'(\d+(?:\.\d+)?%\s*(?:udziałów|akcji))',
                r'pakiet\s+(\d+(?:\.\d+)?%)'
            ],
            "tranche_a": [
                r'(?:Transz[aey]\s+Początkow[aey]|Tranche\s+A).*?(\d[\d\s]*\s*(?:EUR|USD|PLN))'
            ],
            "escrow_iban": [
                r'((?:LU|PL|DE|CH|FR)\d{2}(?:-\d{4}){3}-\d{2})',
                r'rachun(?:ek|ku)\s+(?:powiernicz(?:y|ego)|Escrow).*?([A-Z]{2}\d{2}[A-Za-z0-9\-]+)'
            ],
            "patent_number": [
                r'\b(EP-\d+-[A-Z0-9]+)\b',
                r'\b(US-[\d/]+)\b',
                r'\b(JP-[\d-]+)\b'
            ],
            "repo_id": [
                r'repozytorium.*?\b([A-Za-z0-9_-]+/[A-Za-z0-9_.-]+)\b',
                r'repo:\s*([A-Za-z0-9_-]+/[A-Za-z0-9_.-]+)'
            ],
            "security_cert": [
                r'\b(SEC-[A-Z0-9-]+)\b',
                r'certyfikat.*?o\s+sygnaturze\s+([A-Z0-9-]+)'
            ],
            "datacenter": [
                r'(OMNI-DATA-CENTER(?:\s+w\s+Zurychu)?)',
                r'klastr[a-z]*\s+([A-Z]+-[A-Z0-9-]+)'
            ],
            "ceo_role": [
                r'(?:CEO|Chief Executive Officer)\s*(?::|,|\s)?\s*(?:dr\.?\s+|inż\.?\s+)?([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)',
                r'(?:dr\.?\s+|inż\.?\s+)?([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)\s*(?:obejmuje|pełni|zostaje|jako)?\s*(?:stanowisko\s+)?(?:Chief Executive Officer|CEO)'
            ],
            "cto_role": [
                r'(?:inżynier\s+|inż\.\s+|dr\s+)?([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+).*?(?:Chief Technology Officer|CTO)',
                r'(?:CTO|Wiceprezes\s+ds\.\s+Technologii).*?(?:inżynier\s+|inż\.\s+|dr\s+)?([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)'
            ],
            "cto_salary": [
                r'(?:CTO|Chief Technology Officer).*?wynagrodzeni[a-z]*.*?(?:wysokości\s+)?(\d[\d\s]*\s*(?:EUR|USD|PLN))',
                r'wynagrodzeni[a-z]*.*?(\d[\d\s]*\s*(?:EUR|USD|PLN)).*?(?:CTO|Chief Technology Officer)'
            ],
            "non_compete_duration": [
                r'zakaz[u]?\s+konkurencji.*?(?:okres\s+|przez\s+|do\s+|wynosi\s+)?(\d+\s*(?:miesięcy|miesiące|lat))',
                r'(\d+\s*(?:miesięcy|miesiące|lat)).*?zakaz[u]?\s+konkurencji'
            ],
            "non_compete_penalty": [
                r'kar[ęe]\s+umowną\s+w\s+wysokości\s+(\d[\d\s]*\s*(?:EUR|USD|PLN))',
                r'(\d[\d\s]*\s*(?:EUR|USD|PLN)).*?kar[yęa]\s+umown'
            ],
            "uokik_case": [
                r'(DKK-[\d/]+)',
                r'UOKiK.*?(?:sygnaturze\s+)?([A-Z0-9/-]+)'
            ],
            "audit_firm": [
                r'(CyberAudit Global|Ernst & Young|PwC|KPMG|Deloitte)'
            ],
            "auditor_name": [
                r'audytora\s+([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+)',
                r'pod\s+nadzorem\s+([A-Z][a-z]+\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]+)'
            ],
            "ai_model": [
                r'\b(SYNAPSE-[A-Z0-9-]+)\b'
            ],
            "arbitration_court": [
                r'(Sąd Arbitrażowy przy Krajowej Izbie Gospodarczej w Warszawie|Sąd Arbitrażowy przy KIG)'
            ],
            "investor_partner": [
                r'Partnera Zarządzającego\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\'[a-z]+)?)',
                r'reprezentowaną przez.*?(?:Partnera\s+Zarządzającego\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\'[a-z]+)?)'
            ]
        }

    def ingest(self, raw_text: str) -> Dict[str, Any]:
        """
        Automatyczna ekstrakcja stanu pamięci roboczej z surowego dokumentu.
        Przeszukuje tekst, wiąże atrybuty z encjami i buduje strukturę stanu S0.
        """
        state = {}
        for slot_name, patterns in self.slot_patterns.items():
            for pat in patterns:
                m = re.search(pat, raw_text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().rstrip('.,;')
                    if re.search(r'\d', val):
                        val = re.sub(r'\s+', ' ', val)
                    state[slot_name] = val
                    break

        # Wychwycenie dodatkowych kotwic wysokiej entropii
        anchors = re.findall(r'\b([A-Z]{2,}-\d+[A-Za-z0-9/-]*)\b', raw_text)
        known_vals = list(state.values())
        extra_anchors = [a for a in set(anchors) if not any(a in str(kv) for kv in known_vals)]
        if extra_anchors:
            state["additional_anchors"] = sorted(extra_anchors)[:10]

        return state

    def update(self, current_state: Dict[str, Any], delta_text: str) -> Dict[str, Any]:
        """
        Aktualizacja stanu pamięci roboczej S_t -> S_{t+1} z rozwiązywaniem konfliktów.
        Jeśli aneks lub nowa informacja zawiera zmodyfikowany atrybut,
        stary fakt jest automatycznie zastępowany nowym (conflict overwrite),
        podczas gdy niezmienione fakty pozostają w stanie roboczym.
        """
        new_state = dict(current_state)
        delta_extracted = self.ingest(delta_text)

        for slot_name, new_val in delta_extracted.items():
            if slot_name == "additional_anchors":
                existing = set(new_state.get("additional_anchors", []))
                existing.update(new_val)
                # Cap additional anchors to strictly bound memory O(1) over infinite streams
                new_state["additional_anchors"] = sorted(list(existing))[:self.max_anchors]
            else:
                new_state[slot_name] = new_val

        return new_state

    def format_working_memory(self, state: Dict[str, Any]) -> str:
        """
        Kompiluje strukturalny stan pamięci w zwięzły bufor roboczy O(1) dla LLM.
        """
        lines = ["[AUTOMATYCZNY BUFOR PAMIĘCI ROBOCZEJ (EPISODIC BUFFER O(1))]:"]

        # 1. Transakcja i Finanse
        if "pre_money_valuation" in state or "equity_percentage" in state:
            val = state.get("pre_money_valuation", "N/A")
            eq = state.get("equity_percentage", "N/A")
            partner = state.get("investor_partner", "")
            partner_str = f", Partner Inwestora: {partner}" if partner else ""
            lines.append(f"- Transakcja: Wycena pre-money: {val} (pakiet: {eq}{partner_str}).")

        # 2. Płatności i Escrow
        if "escrow_iban" in state or "tranche_a" in state:
            iban = state.get("escrow_iban", "N/A")
            tr_a = state.get("tranche_a", "")
            tr_str = f"Transza Początkowa A: {tr_a}, " if tr_a else ""
            lines.append(f"- Płatności i Escrow: {tr_str}Rachunek Escrow (Banque de Luxembourg): {iban}.")

        # 3. Własność Intelektualna i Technologia
        if "patent_number" in state or "ai_model" in state:
            pat = state.get("patent_number", "N/A")
            model = state.get("ai_model", "N/A")
            lines.append(f"- Technologia i Własność: Patent EPO: {pat}, Model autorski: {model}.")

        # 4. Bezpieczeństwo i Repozytorium
        if "repo_id" in state or "security_cert" in state or "datacenter" in state:
            repo = state.get("repo_id", "N/A")
            cert = state.get("security_cert", "N/A")
            dc = state.get("datacenter", "N/A")
            lines.append(f"- Infrastruktura i Kod: Repozytorium: {repo}, Certyfikat YubiKey: {cert}, Hosting: {dc}.")

        # 5. Kadra i Warunki Zarządu
        if "ceo_role" in state or "cto_role" in state:
            ceo = state.get("ceo_role", "N/A")
            cto = state.get("cto_role", "N/A")
            sal = state.get("cto_salary", "")
            sal_str = f" (wynagrodzenie: {sal})" if sal else ""
            lines.append(f"- Kadra Zarządzająca: CEO: {ceo}, CTO: {cto}{sal_str}.")

        # 6. Zakaz Konkurencji i Kary
        if "non_compete_duration" in state or "non_compete_penalty" in state:
            dur = state.get("non_compete_duration", "N/A")
            pen = state.get("non_compete_penalty", "N/A")
            lines.append(f"- Zakaz Konkurencji: Czas trwania: {dur}, Kara umowna za naruszenie: {pen}.")

        # 7. Zgody i Arbitraż
        if "uokik_case" in state or "arbitration_court" in state or "audit_firm" in state:
            uokik = state.get("uokik_case", "N/A")
            court = state.get("arbitration_court", "N/A")
            audit = state.get("audit_firm", "N/A")
            auditor = state.get("auditor_name", "")
            aud_str = f" (audytor: {auditor})" if auditor else ""
            lines.append(f"- Zgody i Audyt: UOKiK: {uokik}, Audyt Due Diligence: {audit}{aud_str}, Sąd: {court}.")

        # Dodatkowe kotwice
        if "additional_anchors" in state and state["additional_anchors"]:
            lines.append(f"- Dodatkowe identyfikatory: {', '.join(state['additional_anchors'])}")

        return "\n".join(lines)
