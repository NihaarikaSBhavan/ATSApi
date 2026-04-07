"""
Phase 2B: Comprehensive Skill Ontology & Domain-Specific Skill Mappings
Includes design, tech, healthcare, finance, manufacturing, and all other domains.
"""

# Master skill ontology with canonical names and synonyms
SKILL_ONTOLOGY = {
    # ─── DESIGN DOMAIN ───────────────────────────────────────────
    "figma": ["figma", "fig", "figmax"],
    "adobe xd": ["adobe xd", "xd", "adobexd"],
    "sketch": ["sketch", "sketchapp"],
    "invision": ["invision", "invsion"],
    "framer": ["framer"],
    "proto.io": ["protoio", "proto.io", "protopie"],
    
    "photoshop": ["photoshop", "ps", "adobe photoshop", "pscc"],
    "illustrator": ["illustrator", "ai", "adobe illustrator"],
    "indesign": ["indesign", "adobe indesign", "id"],
    "canva": ["canva"],
    "affinity designer": ["affinity designer", "affinity"],
    "gimp": ["gimp"],
    
    "wireframing": ["wireframing", "wireframe", "wireframes", "wire-framing"],
    "prototyping": ["prototyping", "prototype", "prototypes", "interactive prototype"],
    "user flows": ["user flows", "user flow", "flow diagram", "flow diagrams"],
    "interaction design": ["interaction design", "interaction designer", "ixd", "ux design"],
    "ui design": ["ui design", "ui designer", "user interface design"],
    "ux design": ["ux design", "user experience design", "user experience"],
    "ux research": ["ux research", "user research", "research", "user testing"],
    "design systems": ["design systems", "design system", "component library", "design tokens"],
    "visual design": ["visual design", "visual designer"],
    "graphic design": ["graphic design", "graphic designer", "graphics"],
    "typography": ["typography", "typographic"],
    "color theory": ["color theory", "color science"],
    "motion design": ["motion design", "motion graphics", "animation"],
    "accessibility": ["accessibility", "a11y", "wcag", "ada compliance"],
    
    "html": ["html", "html5", "html 5"],
    "css": ["css", "css3", "css 3", "styling"],
    "javascript": ["javascript", "js", "ecmascript", "es6", "es2015"],
    "bootstrap": ["bootstrap", "bootstrap framework"],
    "tailwind css": ["tailwind css", "tailwindcss", "tailwind"],
    
    # ─── TECH DOMAIN ─────────────────────────────────────────────
    "python": ["python", "python3", "py"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "java": ["java"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "csharp": ["csharp", "c#", "c sharp"],
    "php": ["php"],
    "go": ["go", "golang"],
    "rust": ["rust"],
    "c++": ["c++", "cpp"],
    "c": ["c (language)", "c programming"],
    "ruby": ["ruby"],
    "r": ["r (programming)", "r language"],
    
    "react": ["react", "reactjs", "react.js"],
    "angular": ["angular", "angularjs"],
    "vue": ["vue", "vuejs", "vue.js"],
    "next.js": ["next.js", "nextjs", "next"],
    "ember": ["ember", "emberjs"],
    "svelte": ["svelte"],
    
    "django": ["django", "django framework"],
    "flask": ["flask"],
    "fastapi": ["fastapi", "fast api"],
    "spring": ["spring", "spring boot"],
    "node.js": ["node.js", "nodejs", "node", "npm"],
    "express": ["express", "express.js"],
    "gin": ["gin (framework)"],
    "rails": ["rails", "ruby on rails"],
    "laravel": ["laravel"],
    
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "scikit-learn": ["scikit-learn", "sklearn", "sk-learn"],
    "keras": ["keras"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm", "light gbm"],
    "pandas": ["pandas", "pandas library"],
    "numpy": ["numpy"],
    "scipy": ["scipy"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly"],
    
    "docker": ["docker", "dockerization", "containerization"],
    "kubernetes": ["kubernetes", "k8s", "kube"],
    "jenkins": ["jenkins"],
    "gitlab": ["gitlab", "gitlab ci"],
    "github": ["github", "github actions"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    
    "aws": ["aws", "amazon web services", "amazon aws"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud platform"],
    "heroku": ["heroku"],
    
    "postgresql": ["postgresql", "postgres", "pg"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "elasticsearch": ["elasticsearch", "es"],
    "dynamodb": ["dynamodb"],
    "cassandra": ["cassandra"],
    
    "rest api": ["rest api", "restful api", "restful", "rest"],
    "graphql": ["graphql"],
    "grpc": ["grpc"],
    
    "machine learning": ["machine learning", "ml", "ml engineering"],
    "deep learning": ["deep learning", "deep neural networks"],
    "computer vision": ["computer vision", "cv", "image recognition"],
    "natural language processing": ["natural language processing", "nlp"],
    "nlp": ["nlp", "natural language"],
    
    # ─── HEALTHCARE DOMAIN ───────────────────────────────────────
    "hl7": ["hl7", "hl-7"],
    "fhir": ["fhir"],
    "dicom": ["dicom"],
    "emr": ["emr", "electronic medical records"],
    "ehr": ["ehr", "electronic health records"],
    "hipaa": ["hipaa"],
    "clinical informatics": ["clinical informatics", "medical informatics"],
    "bioinformatics": ["bioinformatics"],
    "medical coding": ["medical coding", "icd-10", "cpt"],
    "telemedicine": ["telemedicine", "telehealth"],
    
    # ─── FINANCE DOMAIN ──────────────────────────────────────────
    "sap": ["sap"],
    "oracle": ["oracle", "oracle finance"],
    "quickbooks": ["quickbooks", "quickbook"],
    "xero": ["xero"],
    "sage": ["sage"],
    "financial reporting": ["financial reporting", "ifrs", "gaap"],
    "audit": ["audit", "internal audit"],
    "risk management": ["risk management"],
    "derivatives": ["derivatives", "options", "futures"],
    "fixed income": ["fixed income", "bonds"],
    "equity analysis": ["equity analysis", "stocks"],
    
    # ─── MANUFACTURING DOMAIN ────────────────────────────────────
    "cad": ["cad", "computer aided design"],
    "solidworks": ["solidworks"],
    "autocad": ["autocad"],
    "fusion 360": ["fusion 360", "fusion360"],
    "catia": ["catia"],
    "lean": ["lean", "lean manufacturing"],
    "six sigma": ["six sigma"],
    "quality assurance": ["quality assurance", "qa", "qc"],
    "supply chain": ["supply chain", "logistics"],
    "mrp": ["mrp", "material requirements planning"],
    "plc": ["plc", "programmable logic controller"],
    
    # ─── CONSTRUCTION DOMAIN ─────────────────────────────────────
    "revit": ["revit", "autodesk revit"],
    "bim": ["bim", "building information modeling"],
    "project management": ["project management", "pm"],
    "pmp": ["pmp", "project management professional"],
    "quantity surveying": ["quantity surveying", "quantity surveyor", "qs"],
    "bill of quantities": ["bill of quantities", "boq"],
    "cost estimation": ["cost estimation", "estimating", "estimation"],
    "site supervision": ["site supervision", "site management"],
    "osha compliance": ["osha compliance", "safety compliance"],
    
    # ─── SALES & MARKETING ───────────────────────────────────────
    "salesforce": ["salesforce", "sf"],
    "hubspot": ["hubspot"],
    "marketo": ["marketo"],
    "pardot": ["pardot"],
    "google analytics": ["google analytics", "ga"],
    "marketing automation": ["marketing automation"],
    "email marketing": ["email marketing", "email campaigns"],
    "seo": ["seo", "search engine optimization"],
    "sem": ["sem", "search engine marketing"],
    "content marketing": ["content marketing"],
    "brand management": ["brand management"],
    "market research": ["market research"],
    
    # ─── LEGAL DOMAIN ────────────────────────────────────────────
    "lexis nexis": ["lexis nexis", "lexisnexis"],
    "westlaw": ["westlaw"],
    "contract law": ["contract law"],
    "intellectual property": ["intellectual property", "ip law"],
    "corporate law": ["corporate law"],
    "litigation": ["litigation"],
    "compliance": ["compliance", "regulatory compliance"],
    "legal research": ["legal research"],
    "contract drafting": ["contract drafting"],
    
    # ─── LOGISTICS & SUPPLY CHAIN ────────────────────────────────
    "sap logistics": ["sap logistics"],
    "jda": ["jda"],
    "manhattan associates": ["manhattan associates"],
    "warehouse management": ["warehouse management", "wms"],
    "inventory management": ["inventory management"],
    "route optimization": ["route optimization"],
    "freight management": ["freight management"],
    "customs compliance": ["customs compliance"],
    
    # ─── EDUCATION ───────────────────────────────────────────────
    "learning management system": ["learning management system", "lms", "canvas", "blackboard"],
    "curriculum development": ["curriculum development", "curriculum design"],
    "instructional design": ["instructional design", "id", "instructional designer"],
    "e-learning": ["e-learning", "elearning", "online education"],
    "assessment design": ["assessment design", "testing", "exam design"],
    "moodle": ["moodle"],
    "schoology": ["schoology"],
    "student engagement": ["student engagement", "student motivation"],
    "pedagogical design": ["pedagogical design", "pedagogy"],
    
    # ─── HR DOMAIN ───────────────────────────────────────────────
    "recruitment": ["recruitment", "recruiting", "talent acquisition"],
    "employee relations": ["employee relations", "employee management"],
    "payroll": ["payroll", "compensation management"],
    "performance management": ["performance management", "performance review"],
    "training development": ["training development", "employee training"],
    "organizational development": ["organizational development", "org development"],
    "hris": ["hris", "human resources information system"],
    "ats": ["ats", "applicant tracking system"],
    "compensation analysis": ["compensation analysis", "compensation planning"],
    "benefits administration": ["benefits administration", "benefits planning"],
    "employee engagement": ["employee engagement"],
    "diversity and inclusion": ["diversity and inclusion", "d&i"],
    "labor relations": ["labor relations", "industrial relations"],
    "workforce planning": ["workforce planning"],
    
    # ─── SALES & MARKETING - EXPANDED ────────────────────────────
    "sales management": ["sales management", "sales strategy"],
    "account management": ["account management", "account executive"],
    "customer relationship management": ["customer relationship management", "crm"],
    "digital marketing": ["digital marketing"],
    "lead generation": ["lead generation", "lead management"],
    "conversion optimization": ["conversion optimization", "conversion rate optimization"],
    "brand management": ["brand management", "brand strategy"],
    "market research": ["market research", "market analysis"],
    "social media marketing": ["social media marketing", "social media management"],
    "paid advertising": ["paid advertising", "ppc", "pay per click"],
    "affiliate marketing": ["affiliate marketing"],
    "seo": ["seo", "search engine optimization"],
    "sem": ["sem", "search engine marketing"],
    "email marketing": ["email marketing", "email campaigns"],
    "content strategy": ["content strategy", "content creation"],
    "marketing automation": ["marketing automation"],
    "market intelligence": ["market intelligence"],
    
    # ─── LEGAL DOMAIN - EXPANDED ──────────────────────────────────
    "contract law": ["contract law", "contracts"],
    "intellectual property": ["intellectual property", "ip law", "patents"],
    "corporate law": ["corporate law", "business law"],
    "litigation": ["litigation", "litigator"],
    "compliance": ["compliance", "regulatory compliance"],
    "legal research": ["legal research"],
    "contract drafting": ["contract drafting"],
    "intellectual property law": ["intellectual property law", "trademark law", "copyright"],
    "employment law": ["employment law"],
    "regulatory law": ["regulatory law", "administrative law"],
    "discovery": ["discovery", "e-discovery"],
    "legal writing": ["legal writing", "legal writing skills"],
    "bankruptcy law": ["bankruptcy law"],
    "immigration law": ["immigration law"],
    "family law": ["family law"],
    "real estate law": ["real estate law"],
    
    # ─── LOGISTICS & SUPPLY CHAIN - EXPANDED ─────────────────────
    "warehouse management": ["warehouse management", "wms"],
    "inventory management": ["inventory management"],
    "supply chain management": ["supply chain management", "supply chain"],
    "route optimization": ["route optimization"],
    "freight management": ["freight management"],
    "customs compliance": ["customs compliance", "customs clearance"],
    "transportation management": ["transportation management", "tms"],
    "demand forecasting": ["demand forecasting", "demand planning"],
    "procurement": ["procurement", "purchasing"],
    "vendor management": ["vendor management", "supplier management"],
    "logistics planning": ["logistics planning"],
    "reverse logistics": ["reverse logistics"],
    "order fulfillment": ["order fulfillment"],
    "inventory control": ["inventory control"],
    
    # ─── MANUFACTURING - EXPANDED ─────────────────────────────────
    "cad": ["cad", "computer aided design"],
    "solidworks": ["solidworks"],
    "autocad": ["autocad"],
    "fusion 360": ["fusion 360", "fusion360"],
    "catia": ["catia"],
    "creo": ["creo"],
    "lean manufacturing": ["lean manufacturing", "lean"],
    "six sigma": ["six sigma", "lean six sigma"],
    "quality assurance": ["quality assurance", "qa"],
    "quality control": ["quality control", "qc"],
    "supply chain management": ["supply chain management"],
    "mrp": ["mrp", "material requirements planning"],
    "erp": ["erp", "enterprise resource planning"],
    "plc": ["plc", "programmable logic controller"],
    "scada": ["scada", "supervisory control"],
    "production planning": ["production planning"],
    "preventive maintenance": ["preventive maintenance"],
    "root cause analysis": ["root cause analysis", "rca"],
    "iso standards": ["iso standards", "iso 9001"],
    "statistical process control": ["statistical process control", "spc"],
    
    # ─── FINANCE - EXPANDED ───────────────────────────────────────
    "financial reporting": ["financial reporting", "financial statements"],
    "audit": ["audit", "internal audit"],
    "risk management": ["risk management"],
    "derivatives": ["derivatives"],
    "fixed income": ["fixed income", "bonds"],
    "equity analysis": ["equity analysis", "stock analysis"],
    "portfolio management": ["portfolio management"],
    "forex": ["forex", "foreign exchange"],
    "corporate finance": ["corporate finance"],
    "investment banking": ["investment banking"],
    "financial modeling": ["financial modeling"],
    "valuation": ["valuation", "company valuation"],
    "gaap": ["gaap"],
    "ifrs": ["ifrs"],
    "credit analysis": ["credit analysis", "credit risk"],
    "operational risk": ["operational risk"],
    "treasury": ["treasury", "treasury management"],
    "trading": ["trading", "securities trading"],
    "financial analysis": ["financial analysis"],
    
    # ─── HEALTHCARE - EXPANDED ────────────────────────────────────
    "hl7": ["hl7", "hl-7"],
    "fhir": ["fhir"],
    "dicom": ["dicom"],
    "emr": ["emr", "electronic medical records"],
    "ehr": ["ehr", "electronic health records"],
    "hipaa": ["hipaa"],
    "clinical informatics": ["clinical informatics"],
    "medical informatics": ["medical informatics"],
    "bioinformatics": ["bioinformatics"],
    "medical coding": ["medical coding", "icd-10", "cpt"],
    "telemedicine": ["telemedicine", "telehealth"],
    "patient care": ["patient care"],
    "clinical diagnosis": ["clinical diagnosis"],
    "pharmaceutical": ["pharmaceutical", "pharmacology"],
    "nursing": ["nursing", "registered nurse", "rn"],
    "laboratory": ["laboratory", "lab work"],
    "radiology": ["radiology", "medical imaging"],
    "cardiology": ["cardiology"],
    "neurology": ["neurology"],
    "pediatrics": ["pediatrics"],
    "oncology": ["oncology"],
    "surgical technique": ["surgical technique", "surgery"],
    "medication management": ["medication management"],
    
    # ─── GENERAL SOFT SKILLS / PROFESSIONAL ──────────────────────
    "agile": ["agile", "agile methodology"],
    "scrum": ["scrum", "scrum master"],
    "kanban": ["kanban"],
    "git": ["git", "version control", "github", "gitlab", "bitbucket"],
    "communication": ["communication skills", "communication"],
    "leadership": ["leadership"],
    "problem-solving": ["problem-solving", "problem solving"],
    "time management": ["time management"],
    "project coordination": ["project coordination"],
    "teamwork": ["teamwork", "team collaboration"],
    "critical thinking": ["critical thinking"],
    "analytical thinking": ["analytical thinking"],
    "decision making": ["decision making"],
    "presentation skills": ["presentation skills", "public speaking"],
    "negotiation": ["negotiation"],
    "conflict resolution": ["conflict resolution"],
}

# Domain-specific skill mappings (which skills are relevant to each domain)
DOMAIN_SKILL_MAPPING = {
    "design": {
        "primary": [
            "figma", "adobe xd", "sketch", "photoshop", "illustrator", "indesign",
            "wireframing", "prototyping", "user flows", "interaction design",
            "ui design", "ux design", "ux research", "design systems",
            "visual design", "graphic design", "motion design", "accessibility",
            "html", "css", "javascript", "bootstrap", "tailwind css",
        ],
        "secondary": ["canva", "affinity designer", "typography", "color theory"]
    },
    "tech": {
        "primary": [
            "python", "javascript", "java", "typescript", "react", "django",
            "nodejs", "docker", "kubernetes", "aws", "postgresql", "git",
            "machine learning", "tensorflow", "pytorch", "scikit-learn",
            "html", "css", "bootstrap", "tailwind css",
            "figma", "adobe xd", "sketch", "wireframing", "prototyping",
            "ui design", "ux design", "interaction design", "user flows",
        ],
        "secondary": ["php", "ruby", "go", "rust", "angular", "vue", "spring",
        "photoshop", "illustrator", "visual design", "graphic design", "motion design"]
    },
    "healthcare": {
        "primary": [
            "hl7", "fhir", "dicom", "emr", "ehr", "hipaa", "clinical informatics",
            "medical coding", "telemedicine", "patient care", "clinical diagnosis",
            "laboratory", "nursing", "radiology", "medication management"
        ],
        "secondary": ["bioinformatics", "pharmaceutical", "surgical technique"]
    },
    "finance": {
        "primary": [
            "sap", "oracle", "financial reporting", "audit", "risk management",
            "derivatives", "fixed income", "equity analysis", "financial modeling",
            "portfolio management", "corporate finance", "gaap", "ifrs"
        ],
        "secondary": ["quickbooks", "xero", "investment banking", "valuation"]
    },
    "manufacturing": {
        "primary": [
            "cad", "lean", "six sigma", "quality assurance", "quality control",
            "supply chain", "mrp", "erp", "plc", "scada", "production planning",
            "preventive maintenance", "root cause analysis", "iso standards"
        ],
        "secondary": ["solidworks", "autocad", "fusion 360", "catia", "statistical process control"]
    },
    "construction": {
        "primary": [
            "revit", "bim", "project management", "quantity surveying",
            "cost estimation", "site supervision", "osha compliance",
            "building codes", "blueprint reading", "site inspection"
        ],
        "secondary": ["pmp", "structural analysis", "contract administration"]
    },
    "sales_marketing": {
        "primary": [
            "salesforce", "hubspot", "marketing automation", "google analytics",
            "seo", "content marketing", "email marketing", "lead generation",
            "digital marketing", "social media marketing", "sales management",
            "account management"
        ],
        "secondary": ["marketo", "pardot", "sem", "brand management", "market research"]
    },
    "legal": {
        "primary": [
            "contract law", "intellectual property", "corporate law",
            "litigation", "compliance", "legal research", "contract drafting",
            "employment law", "regulatory law", "intellectual property law"
        ],
        "secondary": ["lexis nexis", "westlaw", "bankruptcy law", "immigration law", "e-discovery"]
    },
    "logistics": {
        "primary": [
            "warehouse management", "inventory management", "supply chain management",
            "route optimization", "freight management", "transportation management",
            "demand forecasting", "procurement", "vendor management"
        ],
        "secondary": ["sap logistics", "jda", "customs compliance", "order fulfillment"]
    },
    "hr": {
        "primary": [
            "recruitment", "employee relations", "payroll", "performance management",
            "training development", "organizational development", "hris", "ats",
            "compensation analysis", "employee engagement"
        ],
        "secondary": ["benefits administration", "diversity and inclusion", "labor relations", "workforce planning"]
    },
    "education": {
        "primary": [
            "curriculum development", "instructional design", "e-learning",
            "assessment design", "learning management system", "student engagement",
            "pedagogical design"
        ],
        "secondary": ["moodle", "schoology"]
    },
}

# Domain-specific regex patterns for improved extraction
DOMAIN_PATTERNS = {
    "design": [
        r"(?:ui|ux)(?:\s|-)?design(?:er)?",
        r"figma|adobe\s?xd|sketch|invision|proto\.io",
        r"photoshop|illustrator|indesign|gimp",
        r"wireframe?(?:ing)?|prototyp(?:e|ing)|user\s?flows?",
        r"interaction\s?design|visual\s?design|graphic\s?design",
        r"design\s?systems?|component\s?librar(?:y|ies)",
        r"accessibility|wcag|a11y",
        r"motion\s?design|motion\s?graphics|animation",
    ],
    "tech": [
        r"python|javascript|java|typescript|kotlin",
        r"react|angular|vue|next\.?js|node\.?js",
        r"docker|kubernetes|(?:k8s)",
        r"aws|azure|gcp|google\s?cloud",
        r"machine\s?learning|deep\s?learning|ai|artificial\s?intelligence",
        r"tensorflow|pytorch|scikit.?learn",
    ],
    "healthcare": [
        r"hl7|fhir|dicom|emr|ehr",
        r"hipaa|clinical\s?informatics|medical\s?informatics",
        r"medical\s?coding|icd.?10|cpt",
        r"telemedicine|telehealth|ehr|emr",
        r"patient\s?care|clinical\s?diagnosis|nursing",
        r"radiology|cardiology|neurology|oncology",
    ],
    "finance": [
        r"sap|oracle|quickbooks|xero",
        r"financial\s?reporting|ifrs|gaap",
        r"audit|risk\s?management",
        r"derivatives|fixed\s?income|equity\s?analysis",
        r"portfolio\s?management|corporate\s?finance",
        r"financial\s?modeling|valuation|trading",
    ],
    "construction": [
        r"revit|bim|building\s?information\s?modeling",
        r"quantity\s?surveying|bill\s?of\s?quantities|boq",
        r"cost\s?estimation|site\s?supervision|site\s?management",
        r"osha\s?compliance|safety\s?compliance",
        r"project\s?management|blueprint|structural",
    ],
    "manufacturing": [
        r"solidworks|autocad|cad|fusion\s?360",
        r"lean\s?manufacturing|six\s?sigma",
        r"quality\s?(?:assurance|control)|qa|qc",
        r"mrp|erp|plc|scada|production\s?planning",
        r"preventive\s?maintenance|root\s?cause\s?analysis",
        r"iso\s?standards|statistical\s?process\s?control",
    ],
    "sales_marketing": [
        r"salesforce|hubspot|(?:crm|customer\s?relationship)",
        r"marketing\s?automation|email\s?marketing|social\s?media",
        r"lead\s?generation|seo|sem|ppc",
        r"digital\s?marketing|content\s?marketing|brand\s?management",
        r"sales\s?management|account\s?management|sales\s?strategy",
    ],
    "logistics": [
        r"warehouse\s?management|wms|inventory\s?management",
        r"supply\s?chain|logistics|transportation",
        r"route\s?optimization|freight\s?management",
        r"demand\s?forecasting|procurement|vendor\s?management",
        r"tms|customs|order\s?fulfillment",
    ],
    "legal": [
        r"contract\s?law|intellectual\s?property|corporate\s?law",
        r"litigation|compliance|regulatory|employment\s?law",
        r"legal\s?research|contract\s?drafting|discovery",
        r"trademark|copyright|patent|bankruptcy|immigration",
    ],
    "hr": [
        r"recruitment|recruiting|talent\s?acquisition",
        r"employee\s?relations|payroll|performance\s?management",
        r"training\s?development|organizational\s?development",
        r"hris|ats|compensation\s?analysis|benefits",
        r"employee\s?engagement|diversity\s?and\s?inclusion",
    ],
    "education": [
        r"curriculum\s?(?:development|design)|instructional\s?design",
        r"e-learning|elearning|online\s?education",
        r"assessment\s?design|learning\s?management|lms",
        r"pedagogical|student\s?engagement|moodle|blackboard",
    ],
}

def normalize_skill_name(skill_text: str) -> str:
    """
    Normalize a skill string using the ontology.
    Returns the canonical form if found, otherwise lowercase.
    """
    skill_lower = skill_text.lower().strip()
    
    # Direct lookup
    for canonical, synonyms in SKILL_ONTOLOGY.items():
        if skill_lower in synonyms:
            return canonical
    
    # Fallback: return lowercase
    return skill_lower

def get_domain_skills(domain: str) -> set:
    """Get all skill names relevant to a domain."""
    if domain not in DOMAIN_SKILL_MAPPING:
        return set()
    
    mapping = DOMAIN_SKILL_MAPPING[domain]
    return set(mapping.get("primary", []) + mapping.get("secondary", []))

def is_skill_relevant_to_domain(skill: str, domain: str) -> bool:
    """Check if a skill is relevant to a domain."""
    domain_skills = get_domain_skills(domain)
    skill_normalized = normalize_skill_name(skill)
    return skill_normalized in domain_skills
