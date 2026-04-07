import concurrent.futures
import json
import logging
import os
import re
import shutil
import threading
import warnings
from typing import List, Optional

logger = logging.getLogger(__name__)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from backend.graph.skill_graph import DOMAIN_KEYWORDS

# Phase 2B: Import NER extractor for enhanced skill extraction
try:
    from .ner_extractor import get_ner_extractor, extract_skills_ner
    _ner_available = True
except ImportError:
    _ner_available = False
    logger.warning("[NER] Not available - will use fallback extractors")

FORCE_REGEX      = os.environ.get("FORCE_REGEX", "0") == "1"
MIN_FREE_MB      = int(os.environ.get("MIN_FREE_MB", "600"))
SLM_LOAD_TIMEOUT = int(os.environ.get("SLM_LOAD_TIMEOUT", "90"))

# ── Noise words to NEVER treat as skills ────────────────────────────────
# These are common JD boilerplate that pollute extraction
JD_NOISE = {
    'responsibilities','requirements','skills','experience','ability','knowledge',
    'understanding','familiarity','proven','excellent','outstanding','strong',
    'deep','good','solid','working','hands-on','practical','theoretical',
    'study','select','develop','design','build','create','implement','extend',
    'train','retrain','test','run','perform','keep','join','thrive',
    'similar','appropriate','existing','cutting-edge','sophisticated','robust',
    'bachelor','master','bsc','phd','degree','field','role','team','plus',
    'year','years','month','months','candidate','applicant','engineer','developer',
    'analytical','problem','solving','communication','interpersonal',
    'minimum','required','preferred','optional','nice','must','should',
    'r','c','b','a',  # single letters that aren't real skills
}

# ── Canonical skill aliases — normalize variants to one form ────────────
SKILL_ALIASES = {
    # ML/Data Science
    # NOTE: 'ml', 'ai', 'cv', 'dl', 'rl' intentionally removed from global aliases —
    # they are dangerously ambiguous abbreviations in non-tech domains
    # (ml = millilitres, cv = curriculum vitae, ai = AI company names, etc.)
    # These expansions are handled by domain-specific pattern matching instead.
    'sklearn': 'scikit-learn',
    'sk-learn': 'scikit-learn',
    'sci-kit': 'scikit-learn',
    'tf': 'tensorflow',
    'torch': 'pytorch',
    'sklearn': 'scikit-learn',
    'sk-learn': 'scikit-learn',
    'sci-kit': 'scikit-learn',
    'tf': 'tensorflow',
    'torch': 'pytorch',
    
    # Programming Languages
    'js': 'javascript',
    'ts': 'typescript',
    'node': 'node.js',
    'nodejs': 'node.js',
    
    # Cloud & DevOps
    'k8s': 'kubernetes',
    'kube': 'kubernetes',
    'pg': 'postgresql',
    'postgres': 'postgresql',
    'mongo': 'mongodb',
    'es': 'elasticsearch',
    'gcp': 'google cloud platform',
    'aws': 'aws',
    'azure': 'microsoft azure',
    
    # Time Series & ML Methods
    'arima': 'time series forecasting',
    'sarima': 'time series forecasting',
    'lstm': 'deep learning',
    'cnn': 'computer vision',
    'bert': 'natural language processing',
    'transformers': 'transformers',
    'langchain': 'langchain',
    
    # Tools & Version Control
    'git': 'git',
    'github': 'git',
    'gitlab': 'git',
    'jupyter': 'jupyter notebook',
    'ipython': 'jupyter notebook',
    
    # ─── DESIGN ALIASES (Phase 2B: New Design Domain Support) ────────────
    'xd': 'adobe xd',
    'adobexd': 'adobe xd',
    'adobe xd': 'adobe xd',
    'sketch app': 'sketch',
    'invision': 'invision',
    'invsion': 'invision',
    'proto.io': 'proto.io',
    'protoio': 'proto.io',
    'ps': 'photoshop',
    'adobe photoshop': 'photoshop',
    'pscc': 'photoshop',
    'ai': 'illustrator',
    'adobe illustrator': 'illustrator',
    'affinity': 'affinity designer',
    'canva': 'canva',
    'gimp': 'gimp',
    'wireframe': 'wireframing',
    'wireframes': 'wireframing',
    'prototype': 'prototyping',
    'prototypes': 'prototyping',
    'user flow': 'user flows',
    'ux design': 'ui/ux design',
    'ui/ux': 'ui/ux design',
    'interaction design': 'interaction design',
    'ixd': 'interaction design',
    'ui design': 'ui design',
    'user experience': 'ux research',
    'user research': 'ux research',
    'design systems': 'design systems',
    'design system': 'design system',
    'component library': 'design systems',
    'design tokens': 'design systems',
    'visual design': 'visual design',
    'graphic design': 'graphic design',
    'typography': 'typography',
    'color theory': 'color theory',
    'motion graphics': 'motion design',
    'animation': 'motion design',
    'a11y': 'accessibility',
    'wcag': 'accessibility',
    'ada': 'accessibility',
    
    # ─── CONSTRUCTION ALIASES ────────────────────────────────────
    'boq': 'bill of quantities',
    'quantity surveyor': 'quantity surveying',
    'qty surveyor': 'quantity surveying',
    'takeoff': 'takeoff',
    'construction management': 'construction management',
    'pmp': 'pmp',
    
    # ─── GENERIC PHRASE NORMALIZATION ────────────────────────────
    'analytical and critical thinking skills': 'analytical thinking',
    'analytical and critical thinking': 'analytical thinking',
    'critical thinking skills': 'critical thinking',
    'construction estimating': 'cost estimation',
    'finance experience': 'finance',
    'networking abilities': 'networking',
    "sound knowledge of construction": "construction knowledge",
    "valid driver's license": "driver's license",
}

SKILL_PROMPT = (
    "You are a precise skill extraction engine for a recruiting system.\n"
    "Extract ONLY genuine technical and professional skills from the text.\n"
    "Rules:\n"
    "- Include: programming languages, frameworks, libraries, tools, methodologies, domains\n"
    "- Exclude: generic verbs (develop, design, build), adjectives (strong, excellent),\n"
    "  single letters unless a real language (e.g. R is ok, but not as a standalone adjective),\n"
    "  job titles, degree names, and phrases longer than 4 words\n"
    "- Normalize skill names to lowercase canonical forms\n"
    "Return ONLY a JSON array of lowercase strings. No explanation, no examples.\n"
    "If no real skills found return []."
)

_pipeline = None
_nlp      = None
_ner      = None  # Phase 2B: NER extractor instance
_backend  = None
_model_id = None

def extract_skills(text, domain: str = "generic"):
    """Extract skills using primary backend + NER augmentation + ALWAYS regex fallback.

    The optional `domain` parameter allows domain-specific filtering/boosting.
    Pipeline:
    1. Primary backend (SLM, SpaCy, or Regex)
    2. Phase 2B: NER-based extraction (enhanced skill recognition)
    3. Regex fallback (catches anything missed)
    4. Domain filtering (prioritize relevant skills)
    """
    _ensure_loaded()
    raw = []
    
    # Primary extraction (SLM, SpaCy, or Regex)
    if _backend == "slm":
        raw = _slm_extract(text)
    elif _backend == "spacy":
        raw = _spacy_extract(text)
    else:
        raw = _regex_extract(text, domain)

    # ─── PHASE 2B: Augment with NER-based extraction ────────────────────
    if _ner_available:
        try:
            ner_skills = extract_skills_ner(text, domain)
            raw = list(set(raw) | set(ner_skills))
            logger.debug(f"[NER] Extracted {len(ner_skills)} skills for domain: {domain}")
        except Exception as e:
            logger.warning(f"[NER] Extraction error: {e}")

    # Always augment with regex to catch anything primary method missed
    if _backend != "regex":  # If not already using regex
        regex_skills = _regex_extract(text, domain)
        raw = list(set(raw) | set(regex_skills))

    cleaned = _clean_skills(raw)
    # Filter relevancy by domain (keeps list capped and domain-aware)
    filtered = filter_relevant_skills(cleaned, domain=domain)
    return filtered

def _clean_skills(skills):
    """Post-process: normalize aliases, remove noise, deduplicate."""
    cleaned = []
    seen = set()
    for s in skills:
        s = s.lower().strip()
        # Skip empty, too short, too long
        if len(s) < 2 or len(s) > 50: continue
        # Skip pure noise words
        if s in JD_NOISE: continue
        # Skip strings that look like sentences (contain verb-like patterns)
        words = s.split()
        if len(words) > 5: continue
        # Skip entries that are mostly stopwords
        noise_count = sum(1 for w in words if w in JD_NOISE)
        if noise_count > len(words) * 0.5 and len(words) > 1: continue
        # Apply alias normalization
        s = SKILL_ALIASES.get(s, s)
        # Deduplicate
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned

def get_model_info():
    if _backend is None: return {"backend": "loading", "model": "initialising..."}
    return {"backend": _backend, "model": _model_id or _backend}

def normalize_skills(skills):
    """Normalize skill list: lowercase, deduplicate, remove empty."""
    return list(set(s.lower().strip() for s in skills if s and s.strip()))

def filter_relevant_skills(skills, domain='generic'):
    """Filter out irrelevant noise skills based on domain.

    For non-generic domains, skills that are *exclusively* found in the
    primary skill list of a *different* domain are dropped outright.
    This prevents tech skills (python, pytorch …) bleeding into, e.g., a
    construction role and appearing as false "matched_skills".
    """
    skills_norm = [s for s in normalize_skills(skills) if 2 <= len(s) <= 50 and s not in JD_NOISE]
    if not domain or domain == 'generic' or domain not in DOMAIN_KEYWORDS:
        return skills_norm[:30]

    # ── Build cross-domain exclusion set ────────────────────────────────
    # Skills that belong ONLY to other domains' primary lists — not the
    # current domain and not shared across multiple domains — are noise here.
    try:
        from backend.extraction.skill_ontology import DOMAIN_SKILL_MAPPING
        current_primary   = set(DOMAIN_SKILL_MAPPING.get(domain, {}).get("primary",   []))
        current_secondary = set(DOMAIN_SKILL_MAPPING.get(domain, {}).get("secondary", []))
        current_allowed   = current_primary | current_secondary

        # Collect every skill that appears as primary in *another* domain
        foreign_primary: set = set()
        for d, cfg in DOMAIN_SKILL_MAPPING.items():
            if d != domain:
                foreign_primary.update(cfg.get("primary", []))

        # A skill is "foreign-only" when it's in another domain's primary
        # list AND not in our own domain's allowed set at all
        foreign_only = foreign_primary - current_allowed
    except Exception:
        foreign_only = set()

    skills_norm = [s for s in skills_norm if s not in foreign_only]

    # Prefer skills that appear in domain dependencies or match domain keywords
    domain_cfg = DOMAIN_KEYWORDS.get(domain, {})
    domain_deps = set(domain_cfg.get('deps', {}).keys())
    domain_keywords = set(domain_cfg.get('keywords', []))

    prioritized = []
    secondary = []
    for s in skills_norm:
        if s in domain_deps:
            prioritized.append(s)
        else:
            # check if any domain keyword is substring of skill
            if any(kw in s for kw in domain_keywords):
                prioritized.append(s)
            else:
                secondary.append(s)

    result = prioritized + secondary
    return result[:30]

def preload_async():
    t = threading.Thread(target=_ensure_loaded, daemon=True, name="extractor-preload")
    t.start()
    logger.info("[Extractor] Background preload started.")

def _ensure_loaded():
    global _backend, _model_id, _pipeline, _nlp, _ner
    if _backend is not None: return
    
    # Phase 2B: Initialize NER extractor if available
    if _ner_available and _ner is None:
        try:
            _ner = get_ner_extractor()
            logger.info("[NER] Initialized")
        except Exception as e:
            logger.warning(f"[NER] Failed to initialize: {e}")
    
    if FORCE_REGEX:
        _backend, _model_id = "regex", "keyword-list"; return
    free_mb = _free_mb_on_cache_drive()
    logger.info("[Extractor] Free disk: %.0f MB", free_mb)
    if free_mb >= MIN_FREE_MB:
        pipe, mid = _try_load_slm_with_timeout(SLM_LOAD_TIMEOUT)
        if pipe:
            _pipeline, _backend, _model_id = pipe, "slm", mid; return
    else:
        logger.warning("[Extractor] Only %.0f MB free - skipping SLM.", free_mb)
    nlp = _try_load_spacy()
    if nlp:
        _nlp, _backend, _model_id = nlp, "spacy", "en_core_web_sm"; return
    logger.warning("[Extractor] Using regex fallback.")
    _backend, _model_id = "regex", "keyword-list"

def _free_mb_on_cache_drive():
    cache_dir = os.environ.get("HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    try:
        os.makedirs(cache_dir, exist_ok=True)
        return shutil.disk_usage(cache_dir).free / (1024*1024)
    except Exception: return 0.0

def _try_load_slm_with_timeout(timeout_secs):
    logger.info("[SLM] Starting load (timeout: %ds) ...", timeout_secs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_try_load_slm)
        try:
            return future.result(timeout=timeout_secs)
        except concurrent.futures.TimeoutError:
            logger.warning("[SLM] Timed out after %ds - falling back.", timeout_secs)
            return None, None
        except Exception as e:
            logger.warning("[SLM] Load error: %s", e)
            return None, None

def _try_load_slm():
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        mid = "Qwen/Qwen2-0.5B-Instruct"
        free_mb = _free_mb_on_cache_drive()
        if free_mb < MIN_FREE_MB:
            logger.warning("[SLM] Disk too full (%.0f MB) - skipping.", free_mb)
            return None, None
        logger.info("[SLM] Loading %s ...", mid)
        with warnings.catch_warnings():
            warnings.filterwarnings("error",
                message=".*[Nn]ot enough free disk space.*", category=UserWarning)
            pipe = hf_pipeline("text-generation", model=mid,
                dtype=torch.float16 if device==0 else torch.float32,
                device=device if device==-1 else None,
                device_map="auto" if device==0 else None,
                trust_remote_code=True)
        logger.info("[SLM] Ready - %s", mid)
        return pipe, mid
    except UserWarning as e:
        logger.warning("[SLM] Disk warning: %s", e)
    except (RuntimeError, OSError) as e:
        msg = str(e).lower()
        if any(k in msg for k in ("not enough space", "os error 112", "disk")):
            logger.warning("[SLM] Disk full - stopping.")
        else:
            logger.warning("[SLM] Error: %s", e)
    except ImportError:
        logger.warning("[SLM] transformers/torch not installed.")
    except Exception as e:
        logger.warning("[SLM] Could not load: %s", e)
    return None, None

def _slm_extract(text):
    snippet = text[:3000]
    try:
        messages = [{"role": "system", "content": SKILL_PROMPT},
                    {"role": "user",   "content": "Text:\n" + snippet}]
        prompt = _pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "<|system|>\n" + SKILL_PROMPT + "<|end|>\n<|user|>\nText:\n" + snippet + "<|end|>\n<|assistant|>\n"
    try:
        out   = _pipeline(prompt, max_new_tokens=300, temperature=0.01,
                          do_sample=False, pad_token_id=_pipeline.tokenizer.eos_token_id)
        reply = out[0]["generated_text"]
        reply = reply[len(prompt):].strip() if reply.startswith(prompt) \
                else reply.split("<|assistant|>")[-1].strip()
        reply = re.sub(r"```[a-z]*", "", reply).strip("`").strip()
        m = re.search(r"\[.*?\]", reply, re.DOTALL)
        if m:
            skills = json.loads(m.group())
            if isinstance(skills, list):
                return [str(s).lower().strip() for s in skills if s]
        # Last-resort: extract quoted strings from reply — but ONLY from
        # the assistant portion to avoid harvesting skill names embedded in
        # the echoed system prompt (e.g. the old example strings).
        reply_only = reply.split("<|assistant|>")[-1].strip()
        return re.findall(r'"([^"]+)"', reply_only)
    except Exception as e:
        logger.warning("[SLM] Inference error (%s) - fallback.", e)
        return _spacy_extract(text) if _nlp else _regex_extract(text, domain)

def _try_load_spacy():
    try:
        import spacy
        try:    return spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
    except Exception as e:
        logger.warning("[spaCy] Failed: %s", e)
    return None

# spaCy context patterns — only extract skills after these signal phrases
_SKILL_CTX = re.compile(
    r"\b(?:experience (?:with|in)|proficient (?:in|with)|knowledge of|"
    r"skilled in|expertise in|worked with|using|built with|"
    r"developed (?:in|with)|familiarity with|background in|"
    r"proficiency in|competency in)\s+([\w\s\.\+\#\/\-]{2,40})",
    re.IGNORECASE)

_STOPWORDS = {
    "the","a","an","and","or","in","of","for","to","with","is",
    "are","was","be","has","have","our","your","we","you","i","it",
    "this","that","company","team","role","position","candidate",
    "will","can","must","should","may","might","shall","would","could",
    "not","also","such","as","by","on","at","from","into","through"
}

def _spacy_extract(text):
    """
    Extract skills using spaCy NER + noun phrase patterns + contextual keywords.
    Now includes phrase-based extraction for multi-word skills.
    """
    doc = _nlp(text)
    found = set()
    
    # ===== Named Entity Recognition =====
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
            t = ent.text.strip().lower()
            # Only add if it looks like a real tool/framework name
            if 2 < len(t) < 40 and t not in _STOPWORDS and t not in JD_NOISE:
                if not re.search(r'^[a-z]$', t):  # skip single letters
                    found.add(t)
    
    # ===== Contextual Skill Phrases (after keywords like "proficient in") =====
    for m in _SKILL_CTX.finditer(text):
        for part in re.split(r'[,/&]', m.group(1)):
            part = part.strip().lower()
            if 2 < len(part) < 40 and part not in JD_NOISE:
                found.add(part)
    
    # ===== Noun Phrase Extraction =====
    # Captures multi-word skills like "analytical thinking", "cost estimation", etc.
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        # Keep phrase if: reasonable length, not stopwords, not noise
        if 3 < len(phrase) < 50 and phrase not in JD_NOISE:
            # Heuristic: skill phrases usually contain at least one non-stopword
            words = phrase.split()
            non_stop = [w for w in words if w not in _STOPWORDS]
            if len(non_stop) >= 1:  # At least one meaningful word
                found.add(phrase)
    
    # ===== Augment with regex extraction =====
    found.update(_regex_extract(text))
    
    return list(found)

# ── Comprehensive regex skill list ──────────────────────────────────────
# Covers the resume above + typical ML/DS JDs properly
_KNOWN_SKILLS = [
    # Core languages
    "python","java","javascript","typescript","c++","c#","go","rust","ruby",
    "kotlin","swift","scala","matlab","bash","shell","php","r language",
    # ML/AI core
    "machine learning","deep learning","reinforcement learning",
    "natural language processing","computer vision","mlops","data science",
    "feature engineering","statistical analysis","hypothesis testing",
    "time series forecasting","regression","classification","clustering",
    "model evaluation","cross validation","predictive modeling",
    # Frameworks & libraries
    "pytorch","tensorflow","keras","scikit-learn","pandas","numpy","scipy",
    "matplotlib","seaborn","plotly","xgboost","lightgbm","huggingface",
    "transformers","langchain","spacy","nltk","opencv","pillow",
    "arima","sarima","statsmodels",
    # Web & API
    "react","angular","vue","next.js","fastapi","django","flask","express",
    "node.js","rest api","graphql","grpc",
    # Data & DB
    "sql","postgresql","mysql","sqlite","mongodb","redis","elasticsearch",
    "bigquery","spark","hadoop","kafka","airflow",
    # Cloud & DevOps
    "docker","kubernetes","aws","azure","gcp","terraform","ansible",
    "github actions","gitlab ci","jenkins","ci/cd","linux","nginx",
    # Tools
    "git","jupyter notebook","streamlit","gradio","mlflow","wandb",
    "tableau","power bi","excel","looker",
    # Soft / methodology
    "agile","scrum","system design","microservices","tdd",
    "q-learning","game theory","geospatial analysis","iot",
    
    # ─── DESIGN SKILLS (Phase 2B: New Design Domain Support) ──────────────
    "figma","adobe xd","sketch","invision","proto.io","framer",
    "photoshop","illustrator","indesign","gimp","canva","affinity designer",
    "wireframing","prototyping","user flows","interaction design",
    "ui design","ux design","ux research","design systems",
    "visual design","graphic design","motion design","accessibility",
    "html","css","bootstrap","tailwind css","typography","color theory",
    "wire-framing","user experience design","information architecture",
    
    # Healthcare
    "patient care","clinical diagnosis","surgical technique","medication management","ehr",
    "hipaa compliance","cardiology","neurology","oncology","pediatrics","laboratory",
    "hl7","fhir","dicom","emr","clinical informatics","medical coding","telemedicine",
    # Finance
    "portfolio management","derivatives","equity analysis","fixed income","forex","risk management",
    "gaap","financial modeling","audit","credit risk","operational risk","financial reporting",
    "sap","oracle","quickbooks","xero",
    # Manufacturing & Operations
    "supply chain","lean manufacturing","six sigma","production planning","quality control",
    "iso standards","scada","erp systems","preventive maintenance","root cause analysis",
    "cad","solidworks","autocad","catia",
    # Sales & Marketing
    "lead generation","customer relationship management","digital marketing","content strategy",
    "market research","salesforce","hubspot","google analytics","conversion optimization",
    # Logistics & Supply Chain
    "warehouse management","transportation","order fulfillment","demand forecasting","customs regulations",
    "tms","wms","reverse logistics","procurement","vendor management",
    # HR
    "recruiting","onboarding","employee engagement","payroll","compensation analysis","hris","ats",
    # Education
    "curriculum design","student assessment","e-learning","training program development","lms",
    # Construction
    "quantity surveying","bill of quantities","cost estimation","construction management",
    "takeoff","site supervision","health and safety","osha compliance","contract administration",
    "building codes","structural analysis","blueprint reading","site inspection","cost control",
    "construction estimating","revit","bim","building information modeling",
    # Legal
    "contract law","litigation","intellectual property","legal research","regulatory compliance","case law analysis",
    
    # Generic Soft Skills
    "analytical thinking", "critical thinking", "communication", "leadership",
    "problem-solving", "time management", "project coordination", "driver's license",
    "construction knowledge", "finance", "networking",
]

def _regex_extract(text, domain: str = "generic"):
    t = text.lower()

    # For a known non-generic domain, pre-filter _KNOWN_SKILLS to exclude
    # skills that are primary to a *different* domain only.
    # This prevents python/pytorch/etc. appearing in construction extractions.
    candidate_skills = _KNOWN_SKILLS
    if domain and domain != "generic":
        try:
            from backend.extraction.skill_ontology import DOMAIN_SKILL_MAPPING
            current_allowed = (
                set(DOMAIN_SKILL_MAPPING.get(domain, {}).get("primary",   [])) |
                set(DOMAIN_SKILL_MAPPING.get(domain, {}).get("secondary", []))
            )
            foreign_primary: set = set()
            for d, cfg in DOMAIN_SKILL_MAPPING.items():
                if d != domain:
                    foreign_primary.update(cfg.get("primary", []))
            foreign_only = foreign_primary - current_allowed
            candidate_skills = [s for s in _KNOWN_SKILLS if s not in foreign_only]
        except Exception:
            pass  # fall back to full list if import fails

    found = [s for s in candidate_skills if re.search(r"\b" + re.escape(s) + r"\b", t)]
    # Also check for 'R' as a language specifically (must be standalone word, uppercase in original)
    if re.search(r'\bR\b', text) and 'r language' not in found:
        # Only add R if it appears in a clear programming context
        if re.search(r'\b(?:python|java|r)\b', t) and re.search(r'\bR(?:\s*,|\s+and|\s+or)\s*(?:Python|Java|SQL)', text):
            found.append('r language')
    return found