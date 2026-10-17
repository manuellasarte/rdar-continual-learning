# 🧠 RDAR · Rectificación Dispersa Anclada y Regenerativa

**Aprendizaje continuo para LLMs con backbone congelado, memoria dispersa y anclaje simbólico.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/code/manuellasarte/rdar-continual-learning)

> Implementación de referencia de **RDAR**, una mecánica de *lifelong learning* que combina la regla del perceptrón de Rosenblatt (1958) con hallazgos recientes sobre pérdida de plasticidad (Dohare et al., *Nature* 2024) y capas de memoria dispersa (Meta, 2024).

---

## 🎯 Resultado principal

| Tarea | Sin RDAR | Con RDAR |
|---|---|---|
| Retención de conocimiento sintético (4 preguntas) | **0 %** | **100 %** |

El baseline falla incluso alucinando respuestas plausibles (`Cortex-A53`, `Rockchip`) cuando se le pregunta por hechos inventados (`RDAR-7X`, `Nexo-Alpha`). RDAR los recupera al 100 % tras enseñarlos una sola vez.

> 📓 **[Ver notebook completo en Kaggle →](https://www.kaggle.com/code/manuellasarte/rdar-continual-learning)**

---

## 🧩 Qué es RDAR

RDAR extiende la regla de rectificación del perceptrón con tres factores adicionales que, combinados, resuelven los dos problemas históricos del aprendizaje continuo en redes profundas: **olvido catastrófico** y **pérdida de plasticidad**.

$$
w_{\text{nuevo}} = w_{\text{actual}} + \eta \cdot \text{sign}(E) \cdot |E| \cdot x \cdot M_{\text{sparse}} \cdot A_{\text{ancla}}
$$

| Componente | Rol | Implementación |
|---|---|---|
| **Backbone congelado** | Solo aporta el lenguaje; no se modifica. | Qwen2.5-7B-Instruct en 4-bit (`bitsandbytes`) |
| **Memoria dispersa** | Aprende pares `(clave → valor)` con activación top-k. | `SparseMemoryLayer` con pool de 4 000 slots |
| **Anclaje simbólico** | Protege hechos consolidados frente a rectificación. | `SymbolicAnchor` (grafo dirigido `networkx`) |
| **Regeneración** | Reinicia slots de baja utilidad periódicamente. | `_regenerate_one` + `regenerate_batch` |

### Los tres pilares en una frase

1. **Dispersión** (`M_sparse`): sólo se tocan los slots activados por la consulta, no todo el pool.
2. **Anclaje** (`A_ancla`): los slots vinculados a hechos consolidados son prácticamente inmutables.
3. **Regeneración**: sin ella, el sistema pierde plasticidad (Dohare et al., 2024). Con ella, se mantiene indefinidamente.

---

## 📊 Experimentos

El notebook ejecuta cinco experimentos reproducibles:

| # | Experimento | Métrica clave | Resultado |
|---|---|---|---|
| 6.1 | Aprendizaje sintético | Slots anclados creados | 5 hechos / 10 slots |
| 6.2 | Retención con vs. sin RDAR | Accuracy | 0 % → 100 % |
| 6.3 | Rectificación por error | Slots rectificados | 2 (dispersión confirmada) |
| 6.4 | Regeneración de plasticidad | Slots basura eliminados | 20 / 20, anclados intactos |
| 6.5 | Visualización | Bimodalidad de `anchor_weight` | Confirmada |

Resultados completos y discusión en la **Sección 7** del notebook.

---

## 🚀 Cómo reproducir

### Opción A — Kaggle (recomendado)

1. Abre el **[notebook en Kaggle](https://www.kaggle.com/code/manuellasarte/rdar-continual-learning)**.
2. Haz clic en **Copy & Edit** (esquina superior derecha).
3. En **Settings** (panel derecho):
   - `Accelerator` → **GPU T4 x2**
   - `Internet` → **On**
4. Ejecuta con **Run → Run All**.

Tiempo estimado: **15–25 min** la primera vez (descarga el modelo), **~8 min** en sesiones posteriores (si activas *Persistence*).

### Opción B — Local

```bash
git clone https://github.com/manuellasarte/rdar-continual-learning.git
cd rdar-continual-learning
pip install -r requirements.txt
jupyter notebook
```

> ⚠️ Necesitas **≥ 12 GB de VRAM** para cargar el backbone en 4-bit. En CPU funciona pero es muy lento.

---

## 📁 Estructura del repositorio

```
rdar-continual-learning/
├── README.md
├── LICENSE                      # MIT
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── rdar_continual_learning.ipynb
└── docs/
    └── figures/
        ├── memory_distribution.png
        └── retention_comparison.png
```

---

## 🗺️ Roadmap

- [x] **Fase 1 (MVP)** — Backbone congelado + memoria dispersa + anclaje + regeneración. ✅
- [ ] **Fase 2** — RDAR con pesos entrenables (LoRA disperso sobre Qwen2.5-1.5B sin cuantizar).
- [ ] **Fase 3** — Anclaje simbólico con resolución de contradicciones y extracción automática.
- [ ] **Fase 4** — Benchmark multi-tarea (100+ tareas) y comparación contra LoRA, EWC, replay.
- [ ] **Fase 5** — Interfaz Gradio + caso de uso real (asistente personal que aprende).

---

## 📚 Referencias clave

- Rosenblatt, F. (1958). *The Perceptron*. **Psychological Review**, 65(6), 386–408. [DOI](https://doi.org/10.1037/h0042519)
- Dohare, S. et al. (2024). *Loss of plasticity in deep continual learning*. **Nature**, 632, 768–774. [DOI](https://doi.org/10.1038/s41586-024-07711-7)
- Berges, V.-P. et al. (2024). *Memory Layers at Scale*. Meta AI. [arXiv:2412.09764](https://arxiv.org/abs/2412.09764)
- Garcez, A. d'Avila & Lamb, L. C. (2023). *Neurosymbolic AI: The 3rd Wave*. **AI Review**, 56. [arXiv:2012.05876](https://arxiv.org/abs/2012.05876)
- Dettmers, T. et al. (2023). *QLoRA*. **NeurIPS 2023**. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

Bibliografía completa en la **Sección 9** del notebook.

---

## 📄 Licencia

Este proyecto está bajo licencia **MIT** — ver [LICENSE](LICENSE).

---

## ✍️ Autor

**Manuel Lasarte León** ([@k4r4m310](https://www.linkedin.com/in/manuellasarte/))

Si el proyecto te resulta útil o interesante, una ⭐ en el repo o un comentario en el notebook de Kaggle es bienvenido.

---

## 📌 Cita

```bibtex
@misc{lasarte2026rdar,
  author       = {Lasarte Leon, Manuel},
  title        = {{RDAR: Rectificacion Dispersa Anclada y Regenerativa --
                   Aprendizaje continuo en LLMs con backbone congelado}},
  year         = {2026},
  howpublished = {GitHub \& Kaggle},
  url          = {https://github.com/manuellasarte/rdar-continual-learning}
}
```
