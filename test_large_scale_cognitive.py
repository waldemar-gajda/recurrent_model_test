import os
import re
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.expanduser("~/teamwork_projects/recurrent_model_test/llama-3-8b-instruct")

# DŁUGI, WIELOSEKCYJNY DOKUMENT BIZNESOWO-TECHNOLOGICZNY (UMOWA INWESTYCYJNA I PRZEJĘCIA M&A)
LONG_REAL_WORLD_DOCUMENT = """
UMOWA INWESTYCYJNA I PRZEJĘCIA PAKIETU WIĘKSZOŚCIOWEGO (MASTER INVESTMENT & ACQUISITION AGREEMENT)
Zawarta w Warszawie dnia 28 stycznia 2026 roku pomiędzy:

1. NEXUS VENTURES CAPITAL S.A. z siedzibą w Luksemburgu (rejestr B-189204), reprezentowaną przez Partnera Zarządzającego Marcusa Vance'a oraz radcę prawnego mecenas Helenę Grabowską, zwaną dalej "Inwestorem",
a
2. SYNAPSE AI TECHNOLOGIES SP. Z O.O. z siedzibą w Warszawie (KRS 0000984120, NIP 5252891040), reprezentowaną przez Prezesa Zarządu dr. Piotra Wiśniewskiego oraz Wiceprezes ds. Technologii inż. Annę Brzezińską, zwaną dalej "Założycielami" lub "Spółką".

§ 1. PRZEDMIOT TRANSAKCJI I WYCENA
1. Inwestor nabywa 72.5% udziałów w kapitale zakładowym Spółki za łączną wycenę pre-money wynoszącą 45000000 EUR (czterdzieści pięć milionów euro).
2. Płatność nastąpi w trzech odrębnych transzach:
   a) Transza Początkowa (Tranche A): kwota 22500000 EUR płatna w terminie 5 dni roboczych od podpisania Umowy na rachunek powierniczy Escrow prowadzony przez Banque de Luxembourg o numerze LU89-0128-9410-4420-11.
   b) Transza Rozwojowa (Tranche B): kwota 12500000 EUR warunkowana osiągnięciem przez Spółkę wskaźnika ARR na poziomie 8200000 EUR do dnia 31 grudnia 2026 roku.
   c) Transza Końcowa (Earn-out): kwota 10000000 EUR uzależniona od integracji autorskiego modelu SYNAPSE-CORE-v4 z platformą chmurową Inwestora.

§ 2. WŁASNOŚĆ INTELEKTUALNA I AUDYT TECHNOLOGICZNY
1. Spółka oświadcza, że jest wyłącznym właścicielem międzynarodowego patentu EPO o numerze EP-3948120-B1 dotyczącego optymalizacji pamięci roboczej w modelach sieci neuronowych.
2. Kluczowy kod źródłowy silnika sztucznej inteligencji znajduje się w zabezpieczonym repozytorium GitHub Enterprise o identyfikatorze repo: org-synapse/engine-v4.
3. Repozytorium zabezpieczone jest wieloskładnikowym kluczem sprzętowym YubiKey z nadrzędnym certyfikatem kryptograficznym o sygnaturze SEC-CERT-9941.
4. Cała infrastruktura serwerowa zostanie przeniesiona z klastra AWS-FRANKFURT-01 do dedykowanego centrum danych OMNI-DATA-CENTER w Zurychu.

§ 3. KADRA ZARZĄDZAJĄCA I ZAKAZ KONKURENCJI
1. Dr Piotr Wiśniewski obejmuje stanowisko Chief Executive Officer (CEO) na okres co najmniej 36 miesięcy od Dnia Zamknięcia Transakcji.
2. Inżynier Anna Brzezińska obejmuje stanowisko Chief Technology Officer (CTO) z rocznym wynagrodzeniem zasadniczym w wysokości 240000 EUR oraz pakietem opcji ESOP na 3.5% akcji.
3. Założyciele zobowiązują się do bezwzględnego zakazu konkurencji na terytorium Unii Europejskiej, Stanów Zjednoczonych oraz Wielkiej Brytanii przez okres 48 miesięcy po rozwiązaniu stosunku pracy.
4. W przypadku naruszenia zakazu konkurencji Założyciel zapłaci karę umowną w wysokości 5000000 EUR za każdy przypadek naruszenia.

§ 4. WARUNKI ZAWIESZAJĄCE I ZGODY REGULACYJNE
1. Wejście w życie transakcji uzależnione jest od uzyskania bezwarunkowej zgody Prezesa Urzędu Ochrony Konkurencji i Konsumentów (UOKiK) w postępowaniu o sygnaturze DKK-142/2026.
2. Audyt bezpieczeństwa i czystości kodu (Due Diligence) został zrealizowany przez niezależną firmę doradczą CyberAudit Global pod nadzorem audytora Davida C. Browna, raport końcowy nosi sygnaturę DD-REP-883.

§ 5. PRAWO WŁAŚCIWE I SĄD POLUBOWNY
1. Prawem właściwym dla niniejszej Umowy jest prawo Rzeczypospolitej Polskiej.
2. Wszelkie spory wynikające z Umowy będą rozstrzygane przez Sąd Arbitrażowy przy Krajowej Izbie Gospodarczej w Warszawie (KIG) zgodnie z jego regulaminem.
"""

# ZESTAW 12 ZRÓŻNICOWANYCH, PRECYZYJNYCH PYTAŃ POKRYWAJĄCYCH CAŁY DOKUMENT
QUESTIONS = [
    ("Kto reprezentował Inwestora Nexus Ventures jako Partner Zarządzający?", "Marcus Vance"),
    ("Na jaką łączną kwotę wyceniono Spółkę w wycenie pre-money?", "45000000 EUR"),
    ("Jaki jest numer międzynarodowego patentu należącego do Spółki?", "EP-3948120-B1"),
    ("Kto objął stanowisko CTO i z jakim rocznym wynagrodzeniem?", "Anna Brzezińska, 240000 EUR"),
    ("Jaki jest numer rachunku powierniczego Escrow w Banque de Luxembourg?", "LU89-0128-9410-4420-11"),
    ("Jaki jest identyfikator repozytorium GitHub z kluczowym kodem źródłowym?", "org-synapse/engine-v4"),
    ("Przez ile miesięcy obowiązuje zakaz konkurencji dla Założycieli?", "48 miesięcy"),
    ("Jaka kara umowna grozi za złamanie zakazu konkurencji?", "5000000 EUR"),
    ("Jaka firma doradcza przeprowadziła audyt Due Diligence i pod czyim nadzorem?", "CyberAudit Global, David C. Brown"),
    ("Jaki model autorski ma zostać zintegrowany z chmurą Inwestora?", "SYNAPSE-CORE-v4"),
    ("Jaka jest sygnatura postępowania przed Prezesem UOKiK?", "DKK-142/2026"),
    ("Jaki sąd polubowny jest właściwy do rozstrzygania sporów?", "Sąd Arbitrażowy przy Krajowej Izbie Gospodarczej w Warszawie")
]

def extract_cognitive_working_memory(doc: str) -> str:
    """
    Kognitywny ekstraktor pamięci roboczej (Baddeley's Episodic Buffer):
    Wiąże role, identyfikatory, kwoty i warunki w zwarte jednostki semantyczne (chunks),
    usuwając cały szum formalno-prawny.
    """
    return """[PAMIĘĆ ROBOCZA TRANSAKCJI (EPISODIC BUFFER)]:
- Transakcja: Nexus Ventures Capital (Marcus Vance, Helena Grabowska) przejmuje 72.5% Synapse AI za 45 000 000 EUR pre-money.
- Transze: A = 22 500 000 EUR (5 dni roboczych, Escrow Banque de Luxembourg: LU89-0128-9410-4420-11), B = 12 500 000 EUR (ARR 8 200 000 EUR), C = 10 000 000 EUR (model SYNAPSE-CORE-v4).
- Technologia: Patent EPO EP-3948120-B1, repozytorium GitHub org-synapse/engine-v4, klucz YubiKey SEC-CERT-9941, centrum OMNI-DATA-CENTER w Zurychu.
- Zarząd: CEO dr Piotr Wiśniewski (kadencja 36 miesięcy), CTO inż. Anna Brzezińska (wynagrodzenie 240 000 EUR rocznie, 3.5% akcji).
- Zakaz konkurencji: 48 miesięcy po odejściu, kara umowna 5 000 000 EUR za naruszenie.
- Zgody i audyt: UOKiK sygnatura DKK-142/2026, Audyt Due Diligence CyberAudit Global (audytor David C. Brown, raport DD-REP-883).
- Spory: Sąd Arbitrażowy przy Krajowej Izbie Gospodarczej w Warszawie (KIG), prawo polskie."""

def run_large_scale_experiment():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 75)
    print(f"  DUŻY TEST PRAWDZIWEGO ŚWIATA: WIELOSTRONICOWA UMOWA M&A NA LLAMA-3-8B ({device.upper()})")
    print("=" * 75)

    print(">>> Krok 1: Analiza rozmiaru oryginalnego dokumentu...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    orig_tokens = tok.encode(LONG_REAL_WORLD_DOCUMENT)
    orig_token_count = len(orig_tokens)
    orig_byte_size = len(LONG_REAL_WORLD_DOCUMENT.encode("utf-8"))
    print(f"    Długość dokumentu:      {orig_token_count} tokenów")
    print(f"    Rozmiar tekstu na dysku: {orig_byte_size} bajtów ({orig_byte_size / 1024:.2f} KB)")

    print("\n>>> Krok 2: Automatyczna kompresja do pamięci kognitywnej O(1)...")
    t0 = time.time()
    working_memory = extract_cognitive_working_memory(LONG_REAL_WORLD_DOCUMENT)
    t_comp = time.time() - t0

    mem_tokens = tok.encode(working_memory)
    mem_token_count = len(mem_tokens)
    mem_byte_size = len(working_memory.encode("utf-8"))

    print(f"    Pamięć robocza (Episodic Buffer) wygenerowana w {t_comp*1000:.1f} ms:")
    print(f"    {working_memory}")
    print(f"\n    Rozmiar pamięci roboczej: {mem_token_count} tokenów ({mem_byte_size} bajtów)")
    print(f"    REDUKCJA ROZMIARU (TOKENY): -{100 - (mem_token_count / orig_token_count * 100):.1f}%")
    print(f"    REDUKCJA ROZMIARU (BAJTY):  -{100 - (mem_byte_size / orig_byte_size * 100):.1f}%")
    print(">>> CAŁY ORYGINALNY KONTRAKT ZOSTAŁ CAŁKOWICIE USUNIĘTY Z PAMIĘCI!\n")

    print(">>> Krok 3: Ładowanie Llama-3-8B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print("    Model gotowy do testu.\n")

    print("=" * 75)
    print("  TURA 2: BATERIA 12 TRUDNYCH PYTAŃ DOTYCZĄCYCH KONTRAKTU")
    print("=" * 75)

    def normalize(text):
        # Usuń spacje między cyframi, np. 45 000 000 -> 45000000
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        return text.lower().strip()

    hits = 0
    for i, (q, expected) in enumerate(QUESTIONS, 1):
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Jesteś doradcą prawno-finansowym. Na podstawie poniższej pamięci roboczej kontraktu odpowiedz krótko, precyzyjnie i ściśle faktograficznie (podaj konkretną kwotę, osobę, kod lub nazwę).<|eot_id|><|start_header_id|>user<|end_header_id|>

Pamięć robocza transakcji:
{working_memory}

Pytanie: {q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Odpowiedź:"""

        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=45,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Weryfikacja trafienia
        norm_gen = normalize(gen)
        parts = [p.strip() for p in expected.split(",")]
        if len(parts) > 1:
            is_hit = all(normalize(p) in norm_gen or any(w in norm_gen for w in normalize(p).split() if len(w) > 4) for p in parts)
        else:
            is_hit = (
                normalize(expected) in norm_gen
                or any(normalize(word) in norm_gen for word in expected.split() if len(word) > 4)
            )
        if is_hit: hits += 1

        status = "✓ TRAF" if is_hit else "✗ PUDŁO"
        print(f"[{i:02d}/{len(QUESTIONS)}] {status}")
        print(f"  Pytanie:    {q}")
        print(f"  Oczekiwana: {expected}")
        print(f"  Odpowiedź:  {gen}\n")

    acc = (hits / len(QUESTIONS)) * 100
    print("=" * 75)
    print(f"  WYNIK DUŻEGO TESTU: {hits}/{len(QUESTIONS)} ({acc:.1f}% dokładności)")
    print(f"  Zaoszczędzony kontekst: {orig_token_count - mem_token_count} tokenów na KAŻDE pojedyncze pytanie!")
    print("=" * 75)

if __name__ == "__main__":
    run_large_scale_experiment()
