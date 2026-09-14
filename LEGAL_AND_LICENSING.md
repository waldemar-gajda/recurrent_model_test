# Intellectual Property Strategy & Open-Source Licensing

**Project:** Baddeley Cognitive Working Memory Architectures for Large Language Models  
**Author:** Waldemar Gajda (Independent Researcher)  

---

## 1. Defensive Open-Source Strategy: The AGPLv3 Imperative

A paramount risk facing innovative long-context architectures is **proprietary cloud enclosure**: commercial hyperscalers (e.g., closed-source AI service providers) absorbing open-source algorithmic breakthroughs into proprietary cloud APIs without contributing improvements back to the scientific community. Standard permissive licenses (MIT, Apache 2.0, BSD) permit cloud providers to host modified versions behind proprietary endpoints, circumventing copyleft triggers.

To protect the Baddeley Cognitive Working Memory architecture from corporate appropriation, the core codebase is released under the **GNU Affero General Public License version 3 (AGPLv3)**.

### Strategic Advantages of AGPLv3:
1. **Network Copyleft Trigger (Section 13):** AGPLv3 explicitly mandates that any entity offering the software as a service over a computer network (Software-as-a-Service, Model-as-a-Service) must make the complete, corresponding source code of the modified software available to all network users free of charge.
2. **Cloud Enclosure Defense:** Hyperscalers cannot encapsulate the recurrent cognitive engine within proprietary managed APIs without open-sourcing their entire service infrastructure stack.
3. **Guaranteed Scientific Openness:** All derivative implementations, optimizations, and adaptations developed by third parties must remain open-access for the global research community.

---

## 2. Commercial Dual-Licensing Framework

To support sustainable enterprise deployment, the project pairs AGPLv3 with a **Commercial Dual-Licensing Architecture** (the MySQL / Qt model):
- **Academic & Open-Source Tier (AGPLv3):** Free access for researchers, academic institutions, open-source developers, and non-commercial projects, bound by full copyleft obligations.
- **Commercial Enterprise Tier (Proprietary License):** Enterprise customers seeking to integrate the cognitive memory engine into proprietary, closed-source on-premise applications or commercial cloud offerings purchase a commercial license. The commercial license exempts the enterprise from AGPLv3 copyleft provisions in exchange for licensing fees that fund ongoing core development.

---

## 3. Patent Strategy & Prior Art Publication

1. **Defensive Prior-Art Publication:** Publishing this comprehensive preprint on arXiv and submitting to peer-reviewed conferences establishes indisputable, timestamped prior art covering:
   - The quadripartite mapping of Baddeley working memory components onto autoregressive LLM state spaces.
   - The dual-stream coupling of continuous scene slots with discrete phonological registers.
   - The Gated Recurrent Bridge with negative initialization ($b_{init} = -4.0$) for non-destructive LLM adaptation.
   - The two-tier evolutionary deliberation workspace (Draft $\to$ Verify).
2. **Defensive Patent Portfolio:** Filing targeted provisional patent applications covering specific hardware-level implementations (e.g., the exact cross-attention slot memory bank caching mechanism and sparse entmax pooling operators) provides a defensive shield against predatory patent assertion entities.
