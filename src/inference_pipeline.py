"""
Clinical Decision Support System Pipeline
Author: Faisal Jammali
Description: 
This module orchestrates an AI-powered pipeline to detect and analyze 
Drug-Drug (DDI) and Drug-Condition (DCI) interactions using MedGemma-1.5.
Incorporates clinical guardrails, LLM evaluation, and structured report generation.
"""

import os
import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login

# =========================================================
# 1. Logging Configuration (Observability)
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MedGemmaPipeline")

# =========================================================
# 2. Configuration & Secrets Management
# =========================================================
def _get_hf_token() -> str:
    """Securely fetch HuggingFace token from Kaggle Secrets or Env variables."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        logger.info("Successfully loaded HF_TOKEN from Kaggle Secrets.")
        return token
    except Exception as e:
        logger.warning(f"Kaggle Secrets not available, falling back to OS env vars. ({e})")
        return os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

class PipelineConfig:
    HF_TOKEN = _get_hf_token()
    MODEL_ID = "google/medgemma-1.5-4b-it"
    KAGGLE_WORKING_DIR = "/kaggle/working"
    PARQUET_FILE = "/kaggle/input/conflicts-1000/conflicts_1000.parquet" # تأكد أن هذا المسار صحيح لديك
    
    NOISE_WORDS = {
        "breath", "shortness", "chest", "pain", "eyes", "discomfort",
        "diarrhea", "nausea", "vomiting", "fever", "extended release",
        "solution", "ophth", "tartrate", "liquid", "every", "week",
        "long term", "dosage", "medications", "provider", "prescribed",
        "mouth", "investigation", "prophylaxis", "steroid", "mild",
        "anxiety", "sodium", "abdominal", "topamax", "plavix", "ranitidine"
    }

    SEVERE_INTERACTION_PAIRS = {
        frozenset(['simvastatin', 'erythromycin']),
        frozenset(['warfarin', 'ibuprofen']),
        frozenset(['warfarin', 'aspirin']),
        frozenset(['fluoxetine', 'tramadol']),
        frozenset(['sertraline', 'tramadol']),
        frozenset(['metoprolol', 'verapamil']),
        frozenset(['digoxin', 'amiodarone']),
        frozenset(['bupropion', 'sertraline']),
        frozenset(['escitalopram', 'metoprolol']),
        frozenset(['lorazepam', 'metoprolol']),
        frozenset(['citalopram', 'tramadol']),
        frozenset(['apixaban', 'aspirin']),
        frozenset(['clopidogrel', 'aspirin']),
    }

# =========================================================
# 3. Data Models (Structured Data Handling)
# =========================================================
@dataclass
class InteractionCase:
    rx_id: str
    subject_id: str
    target_drug: str
    interacting_drugs: List[str]
    severity: str
    mechanism: str
    patient_context: Dict

@dataclass
class ClinicalReport:
    rx_id: str
    mechanism: str
    clinical_action: str
    doctor_note: str
    confidence_score: float
    raw_llm_output: str

# =========================================================
# 4. Core Pipeline Modules
# =========================================================
class DataIngestionEngine:
    """Handles data extraction and initial preprocessing."""
    
    @staticmethod
    def _is_valid_drug(word: str) -> bool:
        cleaned_word = word.lower().strip()
        return cleaned_word not in PipelineConfig.NOISE_WORDS and len(cleaned_word) > 3

    @staticmethod
    def _calculate_severity(mechanism_type: str, drug: str, interacting_drugs: List[str]) -> str:
        drug_lower = drug.lower()
        for other_drug in interacting_drugs:
            pair = frozenset([drug_lower, other_drug.lower()])
            if pair in PipelineConfig.SEVERE_INTERACTION_PAIRS:
                return 'Severe'
        return 'Major' if mechanism_type == 'DDI' else 'Moderate'

    def process_parquet(self, filepath: str, sample_size: Optional[int] = None) -> List[InteractionCase]:
        logger.info(f"Loading data from {filepath}")
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
            
        if sample_size:
            df = df.head(sample_size)
            
        cases =[]
        for _, row in df.iterrows():
            sid = str(row.get('subject_id', 'Unknown'))
            
            # Process DDI
            for hit in row.get('ddi_hits', []):
                drug = hit['drug']
                real_drugs =[w for w in hit.get('with', []) if self._is_valid_drug(w)]
                if real_drugs:
                    cases.append(InteractionCase(
                        rx_id=f"{sid}_DDI_{drug}",
                        subject_id=sid,
                        target_drug=drug,
                        interacting_drugs=real_drugs,
                        severity=self._calculate_severity('DDI', drug, real_drugs),
                        mechanism="DDI",
                        patient_context={}
                    ))
                    
            # Process DCI
            for hit in row.get('dci_hits', []):
                drug = hit['drug']
                dx_list = list(hit.get('dx',[]))
                if dx_list:
                    cases.append(InteractionCase(
                        rx_id=f"{sid}_DCI_{drug}",
                        subject_id=sid,
                        target_drug=drug,
                        interacting_drugs=dx_list,
                        severity=self._calculate_severity('DCI', drug, dx_list),
                        mechanism="DCI",
                        patient_context={}
                    ))
                    
        logger.info(f"Extracted {len(cases)} valid interaction cases for LLM processing.")
        return cases


class RiskStratificationModule:
    """Analyzes clinical priority based on severity scoring."""
    
    SEVERITY_WEIGHTS = {'Severe': 1.0, 'Major': 0.8, 'Moderate': 0.5, 'Minor': 0.2, 'Unknown': 0.4}
    
    def evaluate_priority(self, case: InteractionCase) -> Dict:
        score = self.SEVERITY_WEIGHTS.get(case.severity, 0.4)
        priority = "HIGH" if score >= 0.8 else "MEDIUM" if score >= 0.5 else "LOW"
        return {'priority': priority, 'severity_score': score}


class MedGemmaInferenceService:
    """Handles LLM interactions, prompt engineering, and response parsing."""
    
    PROMPT_TEMPLATE = """You are an expert clinical pharmacist. Analyze this interaction based on medical evidence.

EXAMPLE 1 (DDI):
Drug: Simvastatin
Interacts with: Erythromycin
Type: Drug-Drug Interaction

MECHANISM: Erythromycin potently inhibits CYP3A4, INCREASING Simvastatin plasma levels 10-20 fold. Risk: rhabdomyolysis.
ACTION: [AVOID]
NOTE: Hold Simvastatin during Erythromycin therapy. Monitor CK if myalgia develops.

ANALYZE:
Drug: {target_drug}
Interacts with: {interacting_drugs}
Type: {interaction_type}

FORMAT (STRICT):
MECHANISM: [1-2 sentences]
ACTION:[ONE: [AVOID], [HOLD DRUG], [ADJUST DOSE], [MONITOR],[CONSULT PRESCRIBER]]
NOTE: [Specific clinical guidance]"""

    def __init__(self):
        self._initialize_model()

    def _initialize_model(self):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU acceleration is required for MedGemma inference.")
            
        logger.info("Initializing MedGemma-1.5-4b model...")
        login(token=PipelineConfig.HF_TOKEN)
        
        self.processor = AutoProcessor.from_pretrained(PipelineConfig.MODEL_ID, token=PipelineConfig.HF_TOKEN)
        self.model = AutoModelForImageTextToText.from_pretrained(
            PipelineConfig.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=PipelineConfig.HF_TOKEN,
            low_cpu_mem_usage=True
        ).eval()
        
        self.dummy_image = Image.fromarray(np.ones((896, 896, 3), dtype=np.uint8) * 255)
        logger.info(f"Model loaded successfully. GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def generate_clinical_report(self, case: InteractionCase, context: Dict) -> ClinicalReport:
        prompt = self.PROMPT_TEMPLATE.format(
            target_drug=case.target_drug,
            interacting_drugs=", ".join(case.interacting_drugs),
            interaction_type="Drug-Drug Interaction (DDI)" if case.mechanism == "DDI" else "Drug-Condition Interaction (DCI)"
        )
        
        raw_output = self._run_inference(prompt)
        parsed_report = self._parse_llm_output(case.rx_id, raw_output, context)
        return self._apply_clinical_guardrails(case, parsed_report)

    def _run_inference(self, prompt: str) -> str:
        try:
            messages =[{"role": "user", "content":[{"type": "image", "image": self.dummy_image}, {"type": "text", "text": prompt}]}]
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=350,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.85,
                    repetition_penalty=1.1,
                    pad_token_id=1
                )
            
            text = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return f"[LLM_ERROR: {str(e)}]"
        finally:
            torch.cuda.empty_cache()

    def _parse_llm_output(self, rx_id: str, text: str, context: Dict) -> ClinicalReport:
        """Parses unstructured LLM output into a structured report format."""
        text = re.sub(r'<unused\d+>|\*\*+', '', text)
        lines = text.split('\n')
        
        mechanism, action, note = "", "[CONSULT PRESCRIBER]", "Consult clinical pharmacist."
        
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if 'MECHANISM:' in line_upper:
                mechanism = line.split(':', 1)[1].strip() if ':' in line else line
            elif 'ACTION:' in line_upper:
                action_match = re.search(r'\[.*?\]', line)
                if action_match: action = action_match.group(0)
            elif 'NOTE:' in line_upper or 'DOCTOR NOTE:' in line_upper:
                note = line.split(':', 1)[1].strip() if ':' in line else line
                
        if not mechanism or len(mechanism) < 20:
            mechanism = text[:300].replace('\n', ' ')
            
        return ClinicalReport(
            rx_id=rx_id, mechanism=mechanism.strip()[:500],
            clinical_action=action.strip(), doctor_note=note.strip()[:300],
            confidence_score=context['severity_score'], raw_llm_output=text
        )

    def _apply_clinical_guardrails(self, case: InteractionCase, report: ClinicalReport) -> ClinicalReport:
        """Applies deterministic rules to override potential LLM hallucinations."""
        if case.severity in['Severe', 'Major'] and '[MONITOR]' in report.clinical_action and 'AVOID' not in report.clinical_action:
            report.clinical_action = '[AVOID] or [CONSULT PRESCRIBER]'
            logger.debug(f"Guardrail triggered for RX_ID {report.rx_id}: Action escalated due to high severity.")
        return report


class OutputEvaluator:
    """Evaluates the quality and completeness of generated reports."""
    
    def evaluate(self, report: ClinicalReport) -> Dict:
        checks = {
            'has_mechanism': len(report.mechanism) > 30,
            'valid_action': any(tag in report.clinical_action for tag in['MONITOR', 'AVOID', 'ADJUST', 'HOLD', 'CONSULT']),
            'has_note': len(report.doctor_note) > 20,
            'no_error_flag': 'LLM_ERROR' not in report.raw_llm_output
        }
        
        score = sum(checks.values()) / len(checks)
        return {'quality_score': score, 'passed': score >= 0.75}


# =========================================================
# 5. Main Orchestrator (Pipeline Entry Point)
# =========================================================
class ClinicalPipelineOrchestrator:
    def __init__(self):
        os.makedirs(PipelineConfig.KAGGLE_WORKING_DIR, exist_ok=True)
        self.data_engine = DataIngestionEngine()
        self.risk_module = RiskStratificationModule()
        self.llm_service = MedGemmaInferenceService()
        self.evaluator = OutputEvaluator()

    def run_pipeline(self, output_filename: str, sample_size: Optional[int] = None):
        logger.info("Starting MedGemma Clinical Pipeline...")
        
        cases = self.data_engine.process_parquet(PipelineConfig.PARQUET_FILE, sample_size)
        if not cases:
            logger.warning("No data found to process. Exiting.")
            return

        results =[]
        for idx, case in enumerate(cases):
            if idx > 0 and idx % 10 == 0:
                logger.info(f"Processed {idx}/{len(cases)} cases...")
                
            context = self.risk_module.evaluate_priority(case)
            report = self.llm_service.generate_clinical_report(case, context)
            evaluation = self.evaluator.evaluate(report)
            
            results.append({
                "RX_ID": report.rx_id,
                "SUBJECT_ID": case.subject_id,
                "TARGET_DRUG": case.target_drug,
                "INTERACTS_WITH": ";".join(case.interacting_drugs),
                "TYPE": case.mechanism,
                "SEVERITY": case.severity,
                "PRIORITY": context['priority'],
                "MECHANISM": report.mechanism,
                "ACTION": report.clinical_action,
                "NOTE": report.doctor_note,
                "QUALITY_SCORE": evaluation['quality_score']
            })

        self._export_results(results, output_filename)
        self._generate_summary(results)

    def _export_results(self, results: List[Dict], filename: str):
        filepath = os.path.join(PipelineConfig.KAGGLE_WORKING_DIR, filename)
        df = pd.DataFrame(results)
        df.to_csv(filepath, sep='\t', index=False)
        logger.info(f"Pipeline complete. Results exported to {filepath}")

    def _generate_summary(self, results: List[Dict]):
        df = pd.DataFrame(results)
        avg_quality = df['QUALITY_SCORE'].mean()
        high_priority = len(df[df['PRIORITY'] == 'HIGH'])
        
        logger.info("=== PIPELINE EXECUTION SUMMARY ===")
        logger.info(f"Total Processed: {len(results)}")
        logger.info(f"High Priority Cases: {high_priority}")
        logger.info(f"Average Pipeline Quality Score: {avg_quality:.2f}")
        logger.info("==================================")


# =========================================================
# Execution Block
# =========================================================
if __name__ == "__main__":
    # Test Run (Sample of 20)
    orchestrator = ClinicalPipelineOrchestrator()
    orchestrator.run_pipeline(output_filename="reports_TEST_20.tsv", sample_size=20)