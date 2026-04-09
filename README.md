# 🏥 MedGuard: AI Inference & Evaluation Engine

> **Note:** This repository contains the **Generative AI & LLM Inference subsystem** developed by [Faisal Jammali]. It is a core component of the broader **MedGuard Capstone Project**, built in collaboration with Saud (Data Integration), Komail (NLP), and Marai (System Architect). 
> For the full system architecture, visit the [Main MedGuard Repository](https://github.com/faisaljammali3-code/MedGuard-AI-Capstone/tree/main).

---

## 🧠 Overview
This repository hosts the **Generative AI Pipeline** responsible for the final medical reasoning and report generation in the MedGuard system. 

It takes cleaned medical entities (extracted via NLP) and known interactions (from OpenFDA), and orchestrates them through **Google's MedGemma-1.5 (4B)** to generate safe, actionable, and structured clinical reports for healthcare providers.

## 🚀 Key Technical Contributions (My Role)
As the Generative AI Engineer on the team, my primary focus was bridging the gap between an experimental LLM and a production-ready medical system:

* **Inference Pipeline Architecture:** Designed an Object-Oriented, MLOps-ready Python service handling data ingestion, risk stratification, and LLM orchestration.
* **LLM Optimization & Prompt Engineering:** Developed strict prompt templates to force MedGemma into outputting structured data (Mechanism, Action, Note) and drastically minimize hallucinations.
* **Clinical Guardrails (Safety First):** Implemented deterministic rule-based overrides. For example, if the LLM suggests "Monitor" for a mathematically proven *Severe* interaction, the guardrail escalates the action to "AVOID".
* **Technical Workaround (Dummy Image Injection):** *Challenge:* MedGemma-1.5 requires multimodal (Image+Text) inputs, but our system is text-only (Electronic Medical Records). *Solution:* Engineered a dynamic dummy image tensor injection at inference time, successfully unlocking the model's medical reasoning capabilities for pure text data.
* **Automated Output Evaluation:** Built a scoring system evaluating the LLM's completeness, safety, and actionability, achieving a pipeline quality score of **0.97 (97%)**.

## 🛠️ Tech Stack
* **AI/LLM:** PyTorch, Transformers, HuggingFace Hub
* **Model:** Google `MedGemma-1.5-4b-it` (running in `bfloat16` for VRAM optimization)
* **Software Engineering:** OOP, Python `logging` for Observability, `dataclasses`
* **Data Processing:** Pandas, NumPy

## ⚙️ Pipeline Execution

The system is designed as a standalone Python service ready to be integrated into an API (e.g., FastAPI):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Securely export HuggingFace Token
export HF_TOKEN="your_hf_token_here"

# 3. Run the evaluation pipeline
python src/inference_pipeline.py