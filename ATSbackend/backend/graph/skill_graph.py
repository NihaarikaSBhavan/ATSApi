import time
import re

import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
from typing import List, Dict, Tuple

from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from backend.embeddings.embedder import embed_skills, cosine_similarity_pair

# Tech/Software Development domain
TECH_DEPS = {
    "deep learning": ["machine learning","python","numpy"],
    "pytorch":       ["python","numpy","deep learning"],
    "tensorflow":    ["python","numpy","deep learning"],
    "keras":         ["python","tensorflow","deep learning"],
    "nlp":           ["python","machine learning"],
    "computer vision":["python","deep learning","numpy"],
    "mlops":         ["docker","kubernetes","machine learning","ci/cd"],
    "kubernetes":    ["docker","linux"],
    "fastapi":       ["python","rest api"],
    "django":        ["python","sql"],
    "react":         ["javascript","html","css"],
    "next.js":       ["react","javascript"],
    "scikit-learn":  ["python","numpy","pandas","machine learning"],
    "graphql":       ["rest api"],
    "terraform":     ["aws","linux"],
    "node.js":       ["javascript","rest api"],
    "spring boot":   ["java","sql","rest api"],
    "java":          ["object oriented programming","sql"],
    "rust":          ["systems programming","memory management"],
    "go":            ["concurrent programming","systems programming"],
    "aws":           ["cloud computing","devops","monitoring"],
    "azure":         ["cloud computing","devops"],
    "gcp":           ["cloud computing","bigquery"],
    "docker":        ["containerization","linux"],
    "ci/cd":         ["automation","git"],
    "jenkins":       ["ci/cd","devops"],
    "git":           ["version control"],
}

# Healthcare domain
HEALTHCARE_DEPS = {
    "clinical diagnosis": ["anatomy","physiology","patient assessment"],
    "surgical technique": ["anatomy","sterile technique","surgical instruments"],
    "patient care": ["communication","empathy","clinical knowledge"],
    "wound care": ["infection control","sterile technique","medications"],
    "medication management": ["pharmacology","patient safety","compliance"],
    "iv therapy": ["phlebotomy","infection control","patient care"],
    "cardiology": ["anatomy","heart physiology","ekg interpretation"],
    "neurology": ["neuroanatomy","neuropharmacology","diagnostic imaging"],
    "oncology": ["cancer biology","chemotherapy","radiation therapy"],
    "pediatrics": ["child development","pediatric medications","patient psychology"],
    "psychiatry": ["psychology","behavioral assessment","mental health medications"],
    "ehr": ["medical terminology","data entry","hipaa compliance"],
    "hipaa compliance": ["data security","patient privacy","regulatory knowledge"],
    "drug interactions": ["pharmacology","medication safety"],
    "pathology": ["medical laboratory science","microscopy","tissue analysis"],
    "blood analysis": ["laboratory science","hematology","quality control"],
    "immunology": ["immunology science","laboratory techniques"],
    "histology": ["tissue preparation","microscopy","staining techniques"],
    "medical imaging": ["dicom standards","imaging interpretation","radiation safety"],
    "ekg interpretation": ["cardiology","electrophysiology"],
}

# Finance & Banking domain
FINANCE_DEPS = {
    "portfolio management": ["investment analysis","risk management","asset allocation"],
    "derivatives": ["advanced mathematics","financial modeling","risk assessment"],
    "equity analysis": ["financial accounting","fundamental analysis","valuation"],
    "fixed income": ["bond analysis","credit analysis","yield analysis"],
    "forex": ["currency markets","economic analysis","technical analysis"],
    "algorithmic trading": ["programming","advanced mathematics","trading strategies"],
    "risk management": ["statistical analysis","compliance","audit"],
    "credit risk": ["financial analysis","default probability modeling"],
    "operational risk": ["compliance","audit","process management"],
    "gaap": ["accounting standards","financial reporting"],
    "financial reporting": ["accounting","sec compliance","audit"],
    "auditing": ["compliance","risk assessment","financial analysis"],
    "tax accounting": ["tax law","financial accounting"],
    "retail banking": ["customer service","regulatory knowledge","sales"],
    "commercial lending": ["credit analysis","financial modeling","negotiation"],
    "financial modeling": ["advanced excel","programming","accounting"],
    "bloomberg terminal": ["financial data analysis","trading tools"],
    "sql": ["data analysis","database management"],
    "python": ["financial analysis","data science","machine learning"],
}

# Manufacturing & Operations domain
MANUFACTURING_DEPS = {
    "supply chain": ["procurement","demand forecasting","inventory management"],
    "production planning": ["scheduling","resource allocation","lean manufacturing"],
    "lean manufacturing": ["process optimization","waste reduction","six sigma"],
    "six sigma": ["statistical analysis","process improvement"],
    "preventive maintenance": ["equipment knowledge","scheduling","safety"],
    "mechanical engineering": ["equipment design","capm","material science"],
    "cad": ["computer design","technical drawing","engineering"],
    "process engineering": ["systems design","optimization","safety"],
    "materials science": ["chemistry","mechanical properties"],
    "automation": ["robotics","programming","electrical systems"],
    "iso standards": ["compliance","quality management"],
    "statistical process control": ["statistics","process monitoring"],
    "root cause analysis": ["problem solving","data analysis"],
    "scada": ["industrial automation","electronics","programming"],
    "iot sensors": ["electronics","data collection","networking"],
    "erp systems": ["data management","process integration"],
    "warehouse management": ["inventory systems","logistics","safety"],
    "osha compliance": ["safety regulations","training"],
}

# Sales & Marketing domain
SALES_MARKETING_DEPS = {
    "account management": ["sales","customer relationship management","negotiation"],
    "lead generation": ["sales","marketing","email marketing"],
    "customer relationship management": ["sales","data management"],
    "digital marketing": ["marketing","social media","content strategy"],
    "brand management": ["marketing","strategic thinking","communication"],
    "content strategy": ["writing","marketing","audience analysis"],
    "conversion optimization": ["analytics","ux knowledge","sales"],
    "a/b testing": ["statistics","analytics"],
    "customer segmentation": ["analytics","marketing","data analysis"],
    "marketing automation": ["email marketing","crm","analytics"],
    "salesforce": ["crm","database management","reporting"],
    "hubspot": ["crm","marketing automation","sales"],
    "google analytics": ["analytics","data interpretation","web knowledge"],
    "social media marketing": ["content creation","audience engagement","analytics"],
    "email marketing": ["copywriting","analytics","list management"],
    "market research": ["analytics","consumer psychology","data collection"],
    "negotiation": ["communication","strategic thinking"],
    "customer psychology": ["communication","decision analysis"],
}

# Logistics & Supply Chain domain
LOGISTICS_DEPS = {
    "warehouse management": ["inventory management","logistics","safety"],
    "transportation": ["route optimization","logistics","compliance"],
    "route optimization": ["analytics","geography","mathematics"],
    "order fulfillment": ["warehouse management","customer service"],
    "last-mile delivery": ["logistics","customer service","route optimization"],
    "procurement": ["vendor management","negotiation","compliance"],
    "inventory management": ["analytics","forecasting","supply chain"],
    "demand forecasting": ["analytics","statistics","supply chain data"],
    "supply chain management": ["procurement","inventory management","logistics"],
    "customs regulations": ["international trade","compliance","documentation"],
    "international trade": ["customs knowledge","compliance","networking"],
    "reverse logistics": ["logistics","inventory management"],
    "wms": ["warehouse management","database systems"],
    "tms": ["transportation management","analytics"],
    "erp systems": ["data management","business processes"],
    "vendor management": ["negotiation","communication","compliance"],
}

# Human Resources domain
HR_DEPS = {
    "recruiting": ["talent sourcing","interviewing","job analysis"],
    "sourcing": ["networking","research","job knowledge"],
    "interviewing": ["communication","people assessment","job knowledge"],
    "performance management": ["employee relations","goal setting","feedback"],
    "employee relations": ["communication","conflict resolution","compliance"],
    "onboarding": ["training","documentation","communication"],
    "employee engagement": ["communication","cultural understanding"],
    "payroll": ["accounting","compliance","data management"],
    "benefits administration": ["compliance","communication"],
    "compensation analysis": ["analytics","market research","statistical analysis"],
    "training and development": ["instructional design","communication","subject matter expertise"],
    "instructional design": ["content creation","learning objectives","assessment"],
    "employment law": ["compliance","legal knowledge"],
    "eeoc compliance": ["compliance","employment law"],
    "labor relations": ["communication","conflict resolution","compliance"],
    "hris": ["database systems","data management"],
    "ats": ["recruitment technology","data management"],
    "lms": ["learning management","data management"],
}

# Education domain
EDUCATION_DEPS = {
    "curriculum design": ["subject matter expertise","learning objectives","assessment"],
    "classroom management": ["communication","student psychology","patience"],
    "student assessment": ["evaluation methods","educational measurement"],
    "e-learning": ["instructional design","technology","content creation"],
    "training program development": ["instructional design","subject matter expertise"],
    "virtual classrooms": ["online teaching","technology","communication"],
    "lms": ["learning management systems","course design"],
    "student counseling": ["psychology","communication","empathy"],
    "mentoring": ["communication","subject matter expertise","patience"],
    "special education": ["diagnostic assessment","individualized education planning"],
    "accreditation": ["compliance","documentation","quality assurance"],
    "enrollment management": ["student recruitment","data analysis"],
    "school administration": ["compliance","budget management","leadership"],
}

# Construction domain
CONSTRUCTION_DEPS = {
    "project management": ["scheduling","resource allocation","cost management"],
    "cost estimation": ["construction and material knowledge","analytics","finance"],
    "quantity surveying": ["cost estimation","bill of quantities","contract administration"],
    "bill of quantities": ["construction and material knowledge","quantity surveying","cost control"],
    "cost control": ["budget management","risk assessment","finance"],
    "scheduling": ["project management","resource planning","critical path"],
    "safety": ["osha compliance","safety protocols","hazard identification"],
    "blueprint reading": ["technical drawing","spatial reasoning"],
    "cad": ["computer design","technical drawing"],
    "structural analysis": ["engineering","mathematics"],
    "building codes": ["construction knowledge","compliance","osha compliance"],
    "concrete": ["materials knowledge","construction techniques","ISO codebook knowledge"],
    "steel work": ["welding","construction techniques","safety"],
    "electrical systems": ["electricity knowledge","building codes"],
    "hvac": ["mechanical systems","building codes"],
    "quality control": ["inspection","standards compliance"],
    "site supervision": ["construction knowledge","communication","safety"],
    "subcontracting": ["negotiation","communication","vendor management"],
    "osha compliance": ["safety regulations","training"],
    "risk assessment": ["safety","compliance","project management"],
    "construction management": ["project management","scheduling","cost control"],
    "takeoff": ["quantity surveying","blueprint reading","estimation"],
}


# Legal domain
LEGAL_DEPS = {
    "corporate law": ["contract law","business law","legal writing"],
    "litigation": ["case law","discovery process","court procedures"],
    "contract law": ["legal writing","business knowledge","negotiation"],
    "intellectual property": ["patent law","trademark law","legal research"],
    "legal research": ["case law analysis","legal databases","writing"],
    "westlaw": ["legal research","legal writing"],
    "lexisnexis": ["legal research","legal writing"],
    "case law analysis": ["legal reasoning","research skills"],
    "regulatory compliance": ["government regulations","documentation","auditing"],
    "discovery process": ["litigation","documentation","legal writing"],
    "deposition": ["litigation","communication","legal knowledge"],
    "trial preparation": ["litigation","organization","case strategy"],
    "contract management": ["legal knowledge","business processes"],
    "legal writing": ["communication","legal knowledge"],
}

# Domain keywords mapping for auto-detection
DOMAIN_KEYWORDS = {
    "tech": {
        "keywords": ["python", "javascript", "java", "react", "angular", "node.js", "docker", 
                     "kubernetes", "devops", "ci/cd", "aws", "cloud", "backend", "frontend", 
                     "full-stack", "machine learning", "ai", "deep learning", "nlp", "sql",
                     "html", "css", "git", "linux", "database", "api", "rest", "graphql"],
        "titles": ["software engineer", "developer", "devops", "data scientist", "ml engineer"],
        "deps": TECH_DEPS
    },
    "healthcare": {
        "keywords": ["clinical", "patient care", "nursing", "physician", "medication", "pharmacy",
                     "surgery", "diagnosis", "anatomy", "physiology", "ehr", "hipaa", "cardiology",
                     "pathology", "laboratory", "imaging", "anesthesia", "pediatrics"],
        "titles": ["nurse", "doctor", "physician", "pharmacist", "surgeon", "clinician", "medical"],
        "deps": HEALTHCARE_DEPS
    },
    "finance": {
        "keywords": ["portfolio", "derivatives", "equity", "fixed income", "forex", "trading",
                     "risk management", "financial", "banking", "investment", "credit", "audit",
                     "gaap", "financial modeling", "bloomberg", "stock market"],
        "titles": ["analyst", "trader", "banker", "accountant", "auditor", "financial advisor"],
        "deps": FINANCE_DEPS
    },
    "manufacturing": {
        "keywords": ["production", "supply chain", "lean", "six sigma", "maintenance", "qc",
                     "mechanical", "cad", "automation", "robotics", "manufacturing", "plant",
                     "process", "iso", "statistical", "scada", "erp"],
        "titles": ["engineer", "manager", "supervisor", "technician", "operator"],
        "deps": MANUFACTURING_DEPS
    },
    "sales_marketing": {
        "keywords": ["sales", "marketing", "account management", "lead generation", "crm",
                     "digital marketing", "brand", "content", "conversion", "analytics",
                     "salesforce", "hubspot", "email", "social media", "advertising"],
        "titles": ["sales", "marketing", "account executive", "campaign manager", "analyst"],
        "deps": SALES_MARKETING_DEPS
    },
    "logistics": {
        "keywords": ["supply chain", "warehouse", "transportation", "logistics", "inventory",
                     "procurement", "distribution", "fulfillment", "route optimization", "customs"],
        "titles": ["logistics", "supply chain", "warehouse", "delivery"],
        "deps": LOGISTICS_DEPS
    },
    "hr": {
        "keywords": ["recruiting", "talent", "hris", "compensation", "payroll", "training",
                     "employee", "benefits", "ats", "onboarding", "performance"],
        "titles": ["recruiter", "hr", "talent", "compensation", "benefits"],
        "deps": HR_DEPS
    },
    "education": {
        "keywords": ["teaching", "curriculum", "student", "classroom", "assessment", "training",
                     "lms", "educational", "course", "instruction", "pedagogy"],
        "titles": ["teacher", "instructor", "educator", "trainer", "professor"],
        "deps": EDUCATION_DEPS
    },
    "construction": {
        "keywords": ["construction", "project management", "building", "safety", "blueprint",
                     "structural", "cad", "electrical", "hvac", "site supervision",
                     "quantity surveying", "bill of quantities", "cost estimation", "takeoff", "construction management"],
        "titles": ["engineer", "manager", "supervisor", "contractor", "quantity surveyor"],
        "deps": CONSTRUCTION_DEPS
    },
    "legal": {
        "keywords": ["legal", "contract", "litigation", "law", "compliance", "attorney",
                     "intellectual property", "discovery", "court", "regulatory"],
        "titles": ["attorney", "lawyer", "counsel", "legal", "paralegal"],
        "deps": LEGAL_DEPS
    }
}

# Merge all domain dependencies for generic fallback and cross-domain reasoning
ALL_DEPS = {}
for deps in [TECH_DEPS, HEALTHCARE_DEPS, FINANCE_DEPS, MANUFACTURING_DEPS,
             SALES_MARKETING_DEPS, LOGISTICS_DEPS, HR_DEPS, EDUCATION_DEPS,
             CONSTRUCTION_DEPS, LEGAL_DEPS]:
    for k,v in deps.items():
        if k not in ALL_DEPS:
            ALL_DEPS[k] = v
        else:
            # merge unique entries for overlap keys
            existing = set(ALL_DEPS[k])
            for u in v:
                if u not in existing:
                    ALL_DEPS[k].append(u)
                    existing.add(u)

# Default fallback: use generic combined dependencies
SKILL_DEPS = ALL_DEPS


def detect_domain(text: str) -> str:
    """Auto-detect domain from job description or resume text.
    
    Tier 2 implementation: Analyzes text for domain-specific keywords and job titles.
    Returns the detected domain or 'generic' as fallback.
    
    Args:
        text: Job description or resume text
        
    Returns:
        Domain name (e.g., 'tech', 'healthcare', 'finance', etc.)
    """
    if not text:
        return "generic"
    
    text_lower = text.lower()
    domain_scores = defaultdict(float)

    # Score each domain based on keyword and title matches (word boundary to avoid false positives)
    for domain, config in DOMAIN_KEYWORDS.items():
        for keyword in config["keywords"]:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            if re.search(pattern, text_lower):
                domain_scores[domain] += 2.0
        for title in config["titles"]:
            pattern = r"\b" + re.escape(title.lower()) + r"\b"
            if re.search(pattern, text_lower):
                domain_scores[domain] += 5.0

    if not domain_scores:
        return "generic"
    
    # Use robust tie-breaking: prefer non-tech when equal or ambiguous
    max_score = max(domain_scores.values())
    top_domains = [d for d,s in domain_scores.items() if s == max_score]
    if len(top_domains) == 1:
        best_domain = top_domains[0]
    else:
        # drop tech if there is a tie with other domain(s)
        non_tech = [d for d in top_domains if d != "tech"]
        best_domain = non_tech[0] if non_tech else "tech"

    # Threshold: low score indicates no clear domain -> generic
    if max_score < 3.0:
        return "generic"

    return best_domain


class DynamicSkillGraph:
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, decay=0.95, domain: str = None):
        """Initialize skill graph with optional domain specialization.

        Tier 1 implementation: Accepts explicit domain parameter for production use.
        Tier 2 implementation: If domain not provided or unknown, uses 'generic' as default.

        Args:
            alpha: Weight for co-occurrence similarity (default 0.5)
            beta: Weight for semantic similarity (default 0.3)
            gamma: Weight for dependency/relationship strength (default 0.2)
            decay: Time decay factor for old co-occurrences (default 0.95)
            domain: Domain name (e.g., 'tech', 'healthcare', 'finance').
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay = decay

        if domain and domain in DOMAIN_KEYWORDS:
            self.domain = domain
            self.skill_deps = DOMAIN_KEYWORDS[domain]["deps"]
        elif domain and domain in ["generic", "all"]:
            self.domain = "generic"
            self.skill_deps = ALL_DEPS
        else:
            self.domain = "generic"
            self.skill_deps = ALL_DEPS
        
        self.graph = nx.Graph()
        self.freq = defaultdict(int)
        self.co_raw = defaultdict(float)
        self.timestamps = {}
        self._embeddings = {}
        self._built = False


    def build(self, corpus):
        self._apply_time_decay()
        for skills in corpus: self._ingest_doc(skills)
        self._recompute_edges()
        self._built = True

    def ingest(self, skills):
        self._ingest_doc(skills)
        self._recompute_edges()

    def _ingest_doc(self, skills):
        unique = list(set(s.lower().strip() for s in skills if s))
        for s in unique: self.freq[s] += 1
        for a,b in combinations(unique,2):
            key = tuple(sorted([a,b]))
            self.co_raw[key] += 1.0
            self.timestamps[key] = time.time()

    def _apply_time_decay(self):
        for key in list(self.co_raw.keys()):
            self.co_raw[key] *= self.decay
            if self.co_raw[key] < 0.01:
                del self.co_raw[key]; self.timestamps.pop(key, None)

    def _recompute_edges(self):
        if not self.co_raw: return
        new_skills = [s for s in self.freq if s not in self._embeddings]
        if new_skills:
            vecs = embed_skills(new_skills)
            for i,s in enumerate(new_skills): self._embeddings[s] = vecs[i]
        for (a,b), co_count in self.co_raw.items():
            if a not in self._embeddings or b not in self._embeddings: continue
            norm_co = co_count / max(1, min(self.freq[a], self.freq[b]))
            sem_sim = cosine_similarity_pair(self._embeddings[a], self._embeddings[b])
            dep_sim = self._dependency_score(a,b)
            w = self.alpha*norm_co + self.beta*sem_sim + self.gamma*dep_sim
            self.graph.add_edge(a, b, weight=w, co=norm_co, sem=sem_sim, dep=dep_sim)

    def _dependency_score(self, a, b):
        """Calculate dependency strength between two skills for current domain.

        Uses domain-specific skill dependencies first, then falls back to all-domain dependencies.
        """
        deps_a = self.skill_deps.get(a, [])
        deps_b = self.skill_deps.get(b, [])
        global_a = ALL_DEPS.get(a, [])
        global_b = ALL_DEPS.get(b, [])

        combined_a = set(deps_a or []) | set(global_a or [])
        combined_b = set(deps_b or []) | set(global_b or [])

        if b in combined_a or a in combined_b:
            return 1.0
        if combined_a & combined_b:
            return 0.5
        return 0.0

    def _fuzzy_match(self, skill, resume_set):
        try:
            from rapidfuzz import fuzz
            best = max((fuzz.token_sort_ratio(skill,r) for r in resume_set), default=0)
            return best / 100.0
        except ImportError: return 0.0

    def soft_match_score(self, resume_skills, jd_skills):
        if not jd_skills: return 0.0
        resume_set = set(s.lower() for s in resume_skills)
        total = 0.0
        for js in jd_skills:
            js = js.lower()
            if js in resume_set: total += 1.0; continue
            fz = self._fuzzy_match(js, resume_set)
            if fz >= 0.72: total += fz * 0.9; continue
            if self._built and self.graph.has_node(js):
                bg = max((self.graph[js][r]['weight'] for r in resume_set if self.graph.has_edge(js,r)), default=0.0)
                if bg > 0: total += bg; continue
            if js in self._embeddings:
                bs = max((cosine_similarity_pair(self._embeddings[js], self._embeddings[r]) for r in resume_set if r in self._embeddings), default=0.0)
                total += bs * 0.85
        return total / len(jd_skills)

    def infer_skills(self, resume_skills, top_k=5, scored: bool = False):
        """Infer related skills from the graph.

        If `scored` is True, return a list of (skill, score) tuples where score is
        normalized to [0,1]. Otherwise return list of skill names (legacy).
        """
        if not self._built: return [] if not scored else []
        rs = set(s.lower() for s in resume_skills)
        scores = defaultdict(float)
        for skill in rs:
            if not self.graph.has_node(skill): continue
            for nb, data in self.graph[skill].items():
                if nb not in rs: scores[nb] += data.get('weight', 0.0)

        # Normalize scores to [0,1]
        if not scores:
            return [] if not scored else []
        max_score = max(scores.values())
        norm = {k: (v / max_score) if max_score > 0 else 0.0 for k, v in scores.items()}

        ranked = sorted(norm.items(), key=lambda x: x[1], reverse=True)[:top_k]
        if scored:
            return ranked
        return [s for s,_ in ranked]

    def skill_gap_analysis(self, resume_skills, jd_skills):
        if not self._built: return []
        rs = set(s.lower() for s in resume_skills)
        missing = [s.lower() for s in jd_skills if s.lower() not in rs]
        results = []
        for ms in missing:
            min_dist = float('inf'); bridge = []
            if self.graph.has_node(ms):
                for r in rs:
                    if not self.graph.has_node(r): continue
                    try:
                        path = nx.shortest_path(self.graph, r, ms)
                        if len(path)-1 < min_dist: min_dist=len(path)-1; bridge=path[1:-1]
                    except nx.NetworkXNoPath: pass
            imp = 1.0/(min_dist+1) if min_dist < float('inf') else 0.1
            results.append({'skill':ms, 'distance':int(min_dist) if min_dist<float('inf') else -1,
                            'bridging_skills':bridge, 'importance':round(imp,3)})
        results.sort(key=lambda x:x['importance'], reverse=True)
        return results

    def top_related(self, skill, top_k=5):
        if not self.graph.has_node(skill): return []
        return sorted(self.graph[skill].items(), key=lambda x:x[1]['weight'], reverse=True)[:top_k]

    def adjacency_summary(self):
        return {node: [(n,round(d['weight'],3)) for n,d in
                sorted(self.graph[node].items(), key=lambda x:x[1]['weight'],reverse=True)[:3]]
                for node in self.graph.nodes}

    def graph_match_score(self, resume, jd):
        return self.soft_match_score(resume, jd)

    def get_domain_info(self) -> Dict:
        """Get current domain configuration for debugging/logging.
        
        Returns:
            Dictionary with domain name and number of dependencies configured
        """
        return {
            "domain": self.domain,
            "num_skill_dependencies": len(self.skill_deps),
            "available_domains": list(DOMAIN_KEYWORDS.keys())
        }

SkillGraph = DynamicSkillGraph
