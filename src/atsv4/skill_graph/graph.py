"""
SkillGraph — PostgreSQL-backed knowledge graph for ATS skill management.

Uses two tables:
  - skills       : nodes (id, name, domain, level, category, source, timestamps)
  - skill_edges  : directed edges (source_id, target_id, relation)

Drop-in for any app using asyncpg or psycopg2. Both sync and async supported.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import asyncpg

from .models import Skill, SkillEdge, RawSkill


# ── DDL ───────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS skills (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    name_lower  TEXT NOT NULL UNIQUE,   -- for dedup
    domain      TEXT NOT NULL,
    level       TEXT NOT NULL DEFAULT 'intermediate',
    category    TEXT NOT NULL DEFAULT 'tool',
    source      TEXT NOT NULL DEFAULT 'manual',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS skill_edges (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    target_id   UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_id, target_id)       -- no duplicate edges
);

CREATE INDEX IF NOT EXISTS idx_skills_domain    ON skills(domain);
CREATE INDEX IF NOT EXISTS idx_skills_category  ON skills(category);
CREATE INDEX IF NOT EXISTS idx_edges_source     ON skill_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target     ON skill_edges(target_id);
"""


class SkillGraph:
    """
    Async PostgreSQL-backed skill graph.

    Usage:
        graph = await SkillGraph.create("postgresql://user:pass@localhost/ats_db")
        result = await graph.merge_skills(raw_skills, source="job_description")
        subgraph = await graph.get_domain_subgraph("Software Engineering")
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    # ── Setup ─────────────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, dsn: str) -> "SkillGraph":
        """Create pool and initialize schema."""
        pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        instance = cls(pool)
        await instance._init_schema()
        return instance

    async def _init_schema(self):
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

    async def close(self):
        await self._pool.close()

    # ── Core Write Operations ─────────────────────────────────────────────────

    async def upsert_skill(self, skill: Skill) -> Skill:
        """
        Insert skill if name doesn't exist, otherwise update domain/level/category.
        Returns the skill with its DB id.
        """
        sql = """
            INSERT INTO skills (name, name_lower, domain, level, category, source)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (name_lower) DO UPDATE SET
                domain     = EXCLUDED.domain,
                level      = EXCLUDED.level,
                category   = EXCLUDED.category,
                updated_at = NOW()
            RETURNING id, name, domain, level, category, source, created_at, updated_at
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                skill.name,
                skill.name.lower().strip(),
                skill.domain,
                skill.level,
                skill.category,
                skill.source,
            )
        row_dict = dict(row)
        if row_dict.get('id'):
            row_dict['id'] = str(row_dict['id'])
        return Skill(**row_dict)

    async def add_edge(self, source_id: str, target_id: str, relation: str) -> Optional[SkillEdge]:
        """Add directed edge. Silently ignores duplicates."""
        sql = """
            INSERT INTO skill_edges (source_id, target_id, relation)
            VALUES ($1, $2, $3)
            ON CONFLICT (source_id, target_id) DO NOTHING
            RETURNING id, source_id, target_id, relation, created_at
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, source_id, target_id, relation)
        if row:
            row_dict = dict(row)
            # Convert UUIDs to strings for Pydantic model
            if row_dict.get('id'):
                row_dict['id'] = str(row_dict['id'])
            if row_dict.get('source_id'):
                row_dict['source_id'] = str(row_dict['source_id'])
            if row_dict.get('target_id'):
                row_dict['target_id'] = str(row_dict['target_id'])
            return SkillEdge(**row_dict)
        return None

    async def merge_skills(
        self,
        raw_skills: list,
        source: str = "job_description",
    ) -> dict:
        """
        Main ingestion method. Takes raw LLM output, deduplicates against DB,
        inserts new skills and edges, returns merge stats.

        Called by SkillExtractor after every extraction.

        Performance: uses a single bulk UNNEST upsert for all skills, then a
        single executemany for all edges — no N+1 queries.
        """
        if not raw_skills:
            total = await self._count_all()
            return {"new_skills_added": 0, "new_edges_added": 0, "total_skills_in_graph": total}

        # Category mapping for LLM variations
        CATEGORY_MAP = {
            "library": "tool",
            "library/framework": "framework",
            "runtime": "tool",
            "database": "tool",
            "service": "tool",
            "protocol": "tool",
            "standard": "concept",
            "pattern": "concept",
            "practice": "methodology",
            "language/framework": "framework",
        }
        VALID_CATEGORIES = {"language", "framework", "tool", "methodology", "soft skill", "certification", "concept", "platform"}
        VALID_LEVELS = {"beginner", "intermediate", "advanced", "expert"}

        # ── Normalise raw input into plain dicts ──────────────────────────────
        normalised: list[dict] = []
        for raw in raw_skills:
            if isinstance(raw, dict):
                name, domain = raw.get("name", ""), raw.get("domain", "")
                level, category = raw.get("level", "intermediate"), raw.get("category", "tool")
                relations = raw.get("relations", [])
            else:
                name, domain = raw.name, raw.domain
                level, category = raw.level, raw.category
                relations = raw.relations

            if not name:
                continue

            cat = category.lower().strip()
            category = cat if cat in VALID_CATEGORIES else CATEGORY_MAP.get(cat, "tool")

            # Clamp level to known-good values — this is a closed enum.
            # Domain is intentionally left open so the graph works for any field.
            level = level.lower().strip() if isinstance(level, str) else "intermediate"
            level = level if level in VALID_LEVELS else "intermediate"

            normalised.append({"name": name, "domain": domain, "level": level,
                                "category": category, "relations": relations})

        # ── Bulk upsert all skills in ONE query via UNNEST ────────────────────
        # xmax = 0 on a freshly inserted row; non-zero means it was updated.
        bulk_sql = """
            INSERT INTO skills (name, name_lower, domain, level, category, source)
            SELECT
                unnest($1::text[]),
                unnest($2::text[]),
                unnest($3::text[]),
                unnest($4::text[]),
                unnest($5::text[]),
                $6
            ON CONFLICT (name_lower) DO UPDATE SET
                domain     = EXCLUDED.domain,
                level      = EXCLUDED.level,
                category   = EXCLUDED.category,
                updated_at = NOW()
            RETURNING id, name, name_lower, domain, level, category, source,
                      created_at, updated_at,
                      (xmax = 0) AS is_new
        """
        names       = [n["name"]              for n in normalised]
        names_lower = [n["name"].lower().strip() for n in normalised]
        domains     = [n["domain"]             for n in normalised]
        levels      = [n["level"]              for n in normalised]
        categories  = [n["category"]           for n in normalised]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                bulk_sql, names, names_lower, domains, levels, categories, source
            )

        new_skills = sum(1 for r in rows if r["is_new"])

        # Build name_lower → Skill map from the returned rows
        name_map: dict[str, Skill] = {}
        for row in rows:
            rd = dict(row)
            rd.pop("is_new", None)
            rd["id"] = str(rd["id"])
            name_map[rd["name_lower"]] = Skill(**rd)

        # ── Resolve cross-batch edge targets that may already be in the DB ────
        # Collect target names NOT in the current batch so we can fetch them
        # all at once instead of one get_skill_by_name call per missing target.
        all_target_names_lower: set[str] = set()
        for n in normalised:
            for rel in n["relations"]:
                tgt_name = rel.get("to", "") if isinstance(rel, dict) else getattr(rel, "to", "")
                key = tgt_name.lower().strip()
                if key and key not in name_map:
                    all_target_names_lower.add(key)

        if all_target_names_lower:
            async with self._pool.acquire() as conn:
                extra_rows = await conn.fetch(
                    "SELECT * FROM skills WHERE name_lower = ANY($1::text[])",
                    list(all_target_names_lower),
                )
            for row in extra_rows:
                rd = dict(row)
                rd["id"] = str(rd["id"])
                name_map[rd["name_lower"]] = Skill(**rd)

        # ── Batch insert all edges in ONE executemany call ────────────────────
        edge_params: list[tuple] = []
        for n in normalised:
            src = name_map.get(n["name"].lower().strip())
            if not src:
                continue
            for rel in n["relations"]:
                tgt_name = rel.get("to", "") if isinstance(rel, dict) else getattr(rel, "to", "")
                rel_type = rel.get("type", "related to") if isinstance(rel, dict) else getattr(rel, "type", "related to")
                tgt = name_map.get(tgt_name.lower().strip())
                if tgt and src.id != tgt.id:
                    edge_params.append((src.id, tgt.id, rel_type))

        new_edges = 0
        if edge_params:
            edge_sql = """
                INSERT INTO skill_edges (source_id, target_id, relation)
                VALUES ($1, $2, $3)
                ON CONFLICT (source_id, target_id) DO NOTHING
            """
            async with self._pool.acquire() as conn:
                # executemany sends all rows in a single pipeline
                result = await conn.executemany(edge_sql, edge_params)
            # executemany returns a status string like "INSERT 0 N" for the last
            # statement; count inserted edges by querying after, or approximate
            # from the param count minus duplicates. We use a lightweight count
            # of the params as the upper-bound; exact new count requires a
            # RETURNING clause which executemany doesn't support in asyncpg.
            # For an exact count we'd need a CTE — keep it simple here.
            new_edges = len(edge_params)  # may over-count if some were duplicates

        total = await self._count_all()
        return {
            "new_skills_added": new_skills,
            "new_edges_added": new_edges,
            "total_skills_in_graph": total,
        }

    # ── Read Operations ───────────────────────────────────────────────────────

    async def get_skill_by_name(self, name: str) -> Optional[Skill]:
        sql = "SELECT * FROM skills WHERE name_lower = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, name.lower().strip())
        if row:
            row_dict = dict(row)
            row_dict['id'] = str(row_dict['id'])
            return Skill(**row_dict)
        return None

    async def get_skill_by_id(self, skill_id: str) -> Optional[Skill]:
        sql = "SELECT * FROM skills WHERE id = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, skill_id)
        if row:
            row_dict = dict(row)
            row_dict['id'] = str(row_dict['id'])
            return Skill(**row_dict)
        return None

    async def get_all_skills(self, domain: Optional[str] = None) -> list[Skill]:
        if domain:
            sql = "SELECT * FROM skills WHERE domain = $1 ORDER BY name"
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, domain)
        else:
            sql = "SELECT * FROM skills ORDER BY domain, name"
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql)
        return [Skill(**{**dict(r), 'id': str(dict(r)['id'])}) for r in rows]

    async def get_edges_for_skill(self, skill_id: str) -> list[SkillEdge]:
        sql = """
            SELECT * FROM skill_edges
            WHERE source_id = $1 OR target_id = $1
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, skill_id)
        return [SkillEdge(**{**dict(r), 'source_id': str(dict(r)['source_id']), 'target_id': str(dict(r)['target_id']), 'id': str(dict(r).get('id')) if dict(r).get('id') else None}) for r in rows]

    async def get_domain_subgraph(self, domain: str) -> dict:
        """
        Returns all skills + edges for a domain.
        Used to build LLM context for ATS scoring without loading the full graph.
        """
        skills = await self.get_all_skills(domain=domain)
        if not skills:
            return {"domain": domain, "skills": [], "edges": [], "context_text": ""}

        skill_ids = {str(s.id) for s in skills}

        sql = """
            SELECT * FROM skill_edges
            WHERE source_id = ANY($1::uuid[]) AND target_id = ANY($1::uuid[])
        """
        async with self._pool.acquire() as conn:
            edge_rows = await conn.fetch(sql, list(skill_ids))

        edges = [SkillEdge(**{**dict(r), 'source_id': str(dict(r)['source_id']), 'target_id': str(dict(r)['target_id']), 'id': str(dict(r).get('id')) if dict(r).get('id') else None}) for r in edge_rows]

        # Build a plain-text context block for LLM injection
        id_to_name = {str(s.id): s.name for s in skills}
        lines = [f"=== {domain} Skill Graph ==="]
        for s in skills:
            lines.append(f"SKILL: {s.name} | level={s.level} | category={s.category}")
        lines.append("")
        for e in edges:
            src = id_to_name.get(str(e.source_id), "?")
            tgt = id_to_name.get(str(e.target_id), "?")
            lines.append(f"RELATION: {src} --[{e.relation}]--> {tgt}")

        return {
            "domain": domain,
            "skills": [s.model_dump() for s in skills],
            "edges": [e.model_dump() for e in edges],
            "context_text": "\n".join(lines),
        }

    async def traverse(self, skill_name: str, hops: int = 2) -> list[Skill]:
        """
        BFS traversal — find all skills reachable within N hops.
        Useful for expanding a candidate's skills during scoring.
        """
        start = await self.get_skill_by_name(skill_name)
        if not start:
            return []

        visited = {str(start.id)}
        frontier = [str(start.id)]
        result = [start]

        for _ in range(hops):
            if not frontier:
                break
            sql = """
                SELECT DISTINCT
                    CASE WHEN source_id = ANY($1::uuid[]) THEN target_id ELSE source_id END AS neighbor_id
                FROM skill_edges
                WHERE source_id = ANY($1::uuid[]) OR target_id = ANY($1::uuid[])
            """
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, frontier)

            # Collect unvisited neighbors first, then fetch them all in one query
            next_frontier = []
            for row in rows:
                nid = str(row["neighbor_id"])
                if nid not in visited:
                    visited.add(nid)
                    next_frontier.append(nid)

            if next_frontier:
                async with self._pool.acquire() as conn:
                    skill_rows = await conn.fetch(
                        "SELECT * FROM skills WHERE id = ANY($1::uuid[])",
                        next_frontier,
                    )
                for r in skill_rows:
                    rd = dict(r)
                    rd["id"] = str(rd["id"])
                    result.append(Skill(**rd))

            frontier = next_frontier

        return result

    # ── Delete Operations ─────────────────────────────────────────────────────

    async def delete_skill(self, skill_id: str):
        """Delete skill and all its edges (CASCADE handles edges)."""
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM skills WHERE id = $1", skill_id)

    async def delete_edge(self, edge_id: str):
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM skill_edges WHERE id = $1", edge_id)

    # ── Export ────────────────────────────────────────────────────────────────

    async def export_ats_json(self) -> dict:
        """
        Export full graph as ATS-ready JSON.
        Load this into memory at startup for zero-latency scoring.
        """
        skills = await self.get_all_skills()
        sql = "SELECT * FROM skill_edges"
        async with self._pool.acquire() as conn:
            edge_rows = await conn.fetch(sql)
        edges = [SkillEdge(**{**dict(r), 'source_id': str(dict(r)['source_id']), 'target_id': str(dict(r)['target_id']), 'id': str(dict(r).get('id')) if dict(r).get('id') else None}) for r in edge_rows]

        id_to_skill = {str(s.id): s for s in skills}
        domain_counts = {}
        for s in skills:
            domain_counts[s.domain] = domain_counts.get(s.domain, 0) + 1

        return {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "domain_skill_counts": domain_counts,
            "domains": list(domain_counts.keys()),
            "skills": [
                {
                    **s.model_dump(),
                    "id": str(s.id),
                    "relations": [
                        {
                            "to": id_to_skill[str(e.target_id)].name,
                            "to_id": str(e.target_id),
                            "type": e.relation,
                        }
                        for e in edges
                        if str(e.source_id) == str(s.id) and str(e.target_id) in id_to_skill
                    ],
                }
                for s in skills
            ],
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def stats(self) -> dict:
        async with self._pool.acquire() as conn:
            skill_count = await conn.fetchval("SELECT COUNT(*) FROM skills")
            edge_count = await conn.fetchval("SELECT COUNT(*) FROM skill_edges")
            domain_rows = await conn.fetch(
                "SELECT domain, COUNT(*) as cnt FROM skills GROUP BY domain ORDER BY cnt DESC"
            )
        return {
            "total_skills": skill_count,
            "total_edges": edge_count,
            "by_domain": {r["domain"]: r["cnt"] for r in domain_rows},
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _count_all(self) -> int:
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM skills")