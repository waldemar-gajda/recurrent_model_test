import random
from typing import List, Dict, Tuple

TEMPLATES = [
    # English templates
    {
        "passage": "Confidential Mission Briefing: The access code for vault {code} has been assigned to Operative {name}. All communications must reference protocol {protocol}.",
        "qa": [
            ("What is the access code for vault?", "{code}"),
            ("Which protocol must all communications reference?", "{protocol}"),
            ("Who was assigned the vault access code?", "Operative {name}")
        ]
    },
    {
        "passage": "Scientific Log Entry #{code}: The newly synthesized compound {compound} reached stability at {temp} degrees Celsius under atmospheric pressure of {pressure} bar.",
        "qa": [
            ("At what temperature did compound {compound} reach stability?", "{temp} degrees Celsius"),
            ("What was the log entry number?", "#{code}"),
            ("What was the atmospheric pressure?", "{pressure} bar")
        ]
    },
    {
        "passage": "System Telemetry: Server cluster {server} experienced a critical memory event. The fallback encryption key is {key} and recovery agent is {name}.",
        "qa": [
            ("What is the fallback encryption key for server {server}?", "{key}"),
            ("Who is the recovery agent?", "{name}")
        ]
    },
    # Polish templates
    {
        "passage": "Raport operacyjny: Tajny kod dostępu do sektora {code} został przekazany agentowi o pseudonimie {name}. Hasło awaryjne to {key}.",
        "qa": [
            ("Jaki jest tajny kod dostępu do sektora?", "{code}"),
            ("Jakie jest hasło awaryjne?", "{key}"),
            ("Komu przekazano kod dostępu?", "Agentowi {name}")
        ]
    },
    {
        "passage": "Dziennik laboratoryjny: Próbka biologiczna {sample} została zabezpieczona w komorze chłodniczej {chamber} w temperaturze {temp} stopni Celsjusza.",
        "qa": [
            ("W której komorze zabezpieczono próbkę {sample}?", "W komorze {chamber}"),
            ("W jakiej temperaturze zabezpieczono próbkę?", "{temp} stopni Celsjusza")
        ]
    },
    {
        "passage": "Zarządzenie bezpieczeństwa IT: Identyfikator sesji administratora to {code}. Wszystkie zapytania do bazy danych muszą zawierać sygnaturę {key}.",
        "qa": [
            ("Jaki jest identyfikator sesji administratora?", "{code}"),
            ("Jaką sygnaturę muszą zawierać zapytania do bazy?", "{key}")
        ]
    }
]

NAMES = ["Jan Kowalski", "Marcus Vance", "Elena Rostova", "Adam Nowak", "Sarah Connor", "Viktor Stone", "Piotr Wiśniewski"]
COMPOUNDS = ["Xenon-9", "Auranil-B", "Krypton-7", "Spectra-X", "Chronos-4"]
PROTOCOLS = ["Omega-Red", "Starlight-4", "Cerberus-X", "Hydra-7", "Pegasus-9"]
CHAMBERS = ["A-102", "B-404", "Gamma-7", "Delta-9", "Echo-12"]

def generate_memory_sample(seed: int = None) -> Tuple[str, str, str]:
    """
    Generates a tuple of (passage, question, target_answer).
    """
    if seed is not None:
        random.seed(seed)
        
    template_data = random.choice(TEMPLATES)
    
    code = f"{random.randint(1000, 9999)}"
    key = f"{random.choice(['ALPHA', 'SIGMA', 'TANGO', 'NEXUS'])}-{random.randint(100, 999)}"
    name = random.choice(NAMES)
    compound = random.choice(COMPOUNDS)
    protocol = random.choice(PROTOCOLS)
    chamber = random.choice(CHAMBERS)
    server = f"US-EAST-{random.randint(1, 99)}"
    temp = f"-{random.randint(10, 80)}" if random.random() > 0.5 else f"{random.randint(20, 150)}"
    pressure = f"{random.randint(1, 20)}.{random.randint(1, 9)}"
    sample = f"BIO-{random.randint(100, 999)}"

    ctx = {
        "code": code,
        "key": key,
        "name": name,
        "compound": compound,
        "protocol": protocol,
        "chamber": chamber,
        "server": server,
        "temp": temp,
        "pressure": pressure,
        "sample": sample,
    }

    passage = template_data["passage"].format(**ctx)
    qa_pair = random.choice(template_data["qa"])
    question = qa_pair[0].format(**ctx)
    answer = qa_pair[1].format(**ctx)

    return passage, question, answer

def generate_multiturn_sample(seed: int = None) -> Tuple[str, str, str, str, str]:
    """
    Generates (passage, q1, a1, q2, a2) where q1 and q2 are distinct questions for the same passage.
    """
    if seed is not None:
        random.seed(seed)
        
    template_data = random.choice(TEMPLATES)
    
    code = f"{random.randint(1000, 9999)}"
    key = f"{random.choice(['ALPHA', 'SIGMA', 'TANGO', 'NEXUS'])}-{random.randint(100, 999)}"
    name = random.choice(NAMES)
    compound = random.choice(COMPOUNDS)
    protocol = random.choice(PROTOCOLS)
    chamber = random.choice(CHAMBERS)
    server = f"US-EAST-{random.randint(1, 99)}"
    temp = f"-{random.randint(10, 80)}" if random.random() > 0.5 else f"{random.randint(20, 150)}"
    pressure = f"{random.randint(1, 20)}.{random.randint(1, 9)}"
    sample = f"BIO-{random.randint(100, 999)}"

    ctx = {
        "code": code,
        "key": key,
        "name": name,
        "compound": compound,
        "protocol": protocol,
        "chamber": chamber,
        "server": server,
        "temp": temp,
        "pressure": pressure,
        "sample": sample,
    }

    passage = template_data["passage"].format(**ctx)
    qa_list = template_data["qa"]
    if len(qa_list) >= 2:
        qa1, qa2 = random.sample(qa_list, 2)
    else:
        qa1, qa2 = qa_list[0], qa_list[0]

    return (
        passage,
        qa1[0].format(**ctx),
        qa1[1].format(**ctx),
        qa2[0].format(**ctx),
        qa2[1].format(**ctx)
    )

def generate_dataset(num_samples: int = 100, base_seed: int = 42) -> List[Tuple[str, str, str]]:
    samples = []
    for i in range(num_samples):
        samples.append(generate_memory_sample(seed=base_seed + i))
    return samples

def generate_multiturn_dataset(num_samples: int = 100, base_seed: int = 42) -> List[Tuple[str, str, str, str, str]]:
    samples = []
    for i in range(num_samples):
        samples.append(generate_multiturn_sample(seed=base_seed + i))
    return samples

if __name__ == "__main__":
    print("Przykładowe wygenerowane próbki pamięciowe wieloturowe:")
    dataset = generate_multiturn_dataset(2)
    for idx, (p, q1, a1, q2, a2) in enumerate(dataset, 1):
        print(f"\n[Próbka {idx}]")
        print(f"Passage: {p}")
        print(f"Tura 2 - Pytanie: {q1} | Odpowiedź: {a1}")
        print(f"Tura 3 - Pytanie: {q2} | Odpowiedź: {a2}")
