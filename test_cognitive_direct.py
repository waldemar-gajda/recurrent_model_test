import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "llama-3-8b-instruct"
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("mps")
model.eval()

doc = """
OFICJALNY RAPORT POWYPADKOWY (POST-MORTEM)
Data zdarzenia: 14 marca 2026 roku, godzina 03:42 UTC.
Dotyczy: Awaria klastra bazy danych w regionie europejskim.

W nocy z 13 na 14 marca system monitoringu Prometheus zgłosił krytyczne przeciążenie serwera PROD-EU-CENTRAL-42 (adres wewnętrzny 192.168.104.18).
Głównym inżynierem dyżurnym prowadzącym akcję ratunkową był Krzysztof Kowalczyk, a wsparcia udzielała Sarah Jenkins z zespołu DevOps.

Analiza logów wykazała niekontrolowany wyciek pamięci w module indeksującym, wprowadzonym we wdrożeniu wersji v3.18.4.
W celu zabezpieczenia spójności transakcji administratorzy odcięli ruch i otworzyli zgłoszenie awaryjne o identyfikatorze INC-9481.
Do odblokowania zaszyfrowanej partycji magazynu danych inżynier użył awaryjnego klucza odzyskiwania VAULT-KEY-819.
Całkowity czas niedostępności usług dla klientów wyniósł dokładnie 47 minut.
Wstępny audyt finansowy oszacował straty operacyjne firmy na kwotę 18400 EUR.
Zgodnie z rekomendacją zespołu cyberbezpieczeństwa cały klaster zostanie do końca tygodnia zmigrowany na wzmocniony protokół HYDRA-SEC-9.
"""

# Kompletny ekstraktor ról poznawczych (Osoby + Kody + Pomiary)
salient_patterns = [
    r'(\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\b)', # Osoby: Krzysztof Kowalczyk, Sarah Jenkins
    r'([A-Z0-9]+-[A-Z0-9]+(?:-[A-Z0-9]+)*)',                                  # Serwery i klucze: PROD-EU-CENTRAL-42, INC-9481, etc.
    r'(v\d+\.\d+\.\d+)',                                                      # Wersje: v3.18.4
    r'(\d+\s*(?:minut|EUR|zł|bar|stopni))',                                   # Pomiary: 47 minut, 18400 EUR
]
symbols = []
for pat in salient_patterns:
    symbols.extend(re.findall(pat, doc))
symbols = list(dict.fromkeys(symbols))
sym_text = ", ".join(symbols)

questions = [
    ("Kto był głównym inżynierem dyżurnym prowadzącym akcję ratunkową?", "Krzysztof Kowalczyk"),
    ("Jaki był identyfikator zgłoszenia awaryjnego w systemie?", "INC-9481"),
    ("Jaka wersja oprogramowania spowodowała wyciek pamięci?", "v3.18.4"),
    ("Ile minut trwał całkowity czas niedostępności usług?", "47 minut"),
    ("Jaki klucz odzyskiwania magazynu danych został użyty?", "VAULT-KEY-819"),
    ("Jaki serwer uległ przeciążeniu?", "PROD-EU-CENTRAL-42"),
    ("Na jaki protokół bezpieczeństwa zostanie zmigrowany klaster?", "HYDRA-SEC-9"),
    ("Na jaką kwotę oszacowano straty operacyjne firmy?", "18400 EUR")
]

print("=" * 72)
print("  TEST KOGNITYWNY: PEŁNY ZESTAW RÓL ZE ŚWIATA RZECZYWISTEGO")
print(f"  Pamięć robocza (Rejestr poznawczy, O(1)):")
print(f"  {sym_text}")
print("=" * 72)

hits = 0
for q, expected in questions:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Jesteś precyzyjnym asystentem. Na podstawie pamięci roboczej incydentu odpowiedz krótko i bezpośrednio (samą nazwą, osobą, kodem lub kwotą).<|eot_id|><|start_header_id|>user<|end_header_id|>

Pamięć robocza incydentu:
{sym_text}

Pytanie: {q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Odpowiedź:"""

    inputs = tok(prompt, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
    gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    is_hit = expected.lower() in gen.lower()
    if is_hit: hits += 1
    status = "✓ TRAF" if is_hit else "✗ PUDŁO"
    print(f"Pytanie:    {q}")
    print(f"Oczekiwana: {expected}")
    print(f"Odpowiedź:  {gen} | {status}\n")

print("=" * 72)
print(f"  WYNIK KOŃCOWY TESTU RZECZYWISTEGO: {hits}/{len(questions)} ({(hits/len(questions))*100:.1f}%)")
print("=" * 72)
