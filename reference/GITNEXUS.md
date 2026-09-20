# Référence — GitNexus (skills Claude .claude/skills/gitnexus) — MCP code-intelligence multi-repos
# Source : https://github.com/1bit-MONSTER/1bit-MONSTER-scaffold-backup/tree/main/.claude/skills/gitnexus
# Fetched 2026-09-20 — 6 SKILL.md copiés dans reference/1bit/gitnexus/

## Ce que c'est
GitNexus = MCP + CLI (`node .gitnexus/run.cjs analyze`) construisant un **graphe de connaissance**
du code (nodes : File/Folder/Function/Class/Interface/Method/Process/Route/Tool + edges CALLS/
IMPORTS/HAS_METHOD/PROCESS...). 6 skills Claude organisent le workflow.

## Skills (le guide référence les autres)
| Skill | Usage |
|-------|-------|
| gitnexus-guide | Référence outils/schéma (ce doc) |
| gitnexus-exploring | "Comment marche X ?" — architecture |
| gitnexus-impact-analysis | "Qu'est-ce qui casse si je change X ?" — blast radius depth 1/2/3 + confidence |
| gitnexus-debugging | "Pourquoi X échoue ?" — trace de bugs |
| gitnexus-refactoring | rename/extract/split multi-fichiers coordonnés |
| gitnexus-cli | index/status/clean/wiki |

## Outils principaux
- `query` : flots d'exécution groupés par processus liés à un concept
- `context` : vue 360° d'un symbole (refs catégorisées + processus)
- `impact` : blast radius symbole à profondeur 1/2/3 avec confidence
- `trace` : plus court chemin entre deux symboles (CALLS + HAS_METHOD), `maxDepth`/`includeTests`
- `detect_changes` : impact du diff git courant
- `rename` : rename coordonné multi-fichiers avec edits taggés confidence
- `cypher` / `pdg_query` : requêtes graphe brutes + control/data dependence (CDG/REACHING_DEF)
- `explain` : taint findings (source→sink, catégories injection/path/sql/xss)
- **Cross-repo** : `group_list` / `group_sync` → Contract Registry (liens HTTP contractuels entre repos),
  `trace { repo: "@group" }` traverse 1 frontière ContractLink
- `list_repos` paginé (max 200), `route_map`, `shape_check`, `api_impact`, `tool_map`

## Intérêt pour notre workspace (geniex_harness / xdna2)
Le workspace est un monorepo multi-composants : `governor/` (policy/adapter/cost_model/actions),
`profiler_v3/`, `xdna2/` (place_expert, cost_contention_patch, benchmark). GitNexus apporte :
1. **`trace` cross-symbole** : place_expert → cost_contention_patch → cost_model (gouverneur)
   sans chaîner des greps manuels — exactement le besoin de câbler le D2 adapter.
2. **`impact`** : "si je change la fonction `place_expert` (seuils → argmin coût), qu'est-ce
   qui casse dans governor/adapter.py et benchmark_cross_tier ?" — avant commit.
3. **`group` cross-repo** : le jour où adaptive-xdna-runtime / d2-quant-planner sont clonés
   comme repos membres, un `trace { repo: "@d2" }` relie xdna2 ↔ governor ↔ externes.
4. **`detect_changes`** : après nos ajouts dans xdna2/ (qui ne touchent pas les sources),
   confirme que profiler_v3/governor sont intacts (audit "sources non modifiées").

## Note de licence / provenance
- Skills dans un repo "scaffold-backup" (backup du repo principal 1bit-MONSTER).
- Le repo 1bit-MONSTER principal a un `.gitnexusignore` → l'outil est activement utilisé
  pour gérer son gros monorepo (3 107 commits, 100% HF coverage).
- À adapter pour opencode (skills opencode ≠ claude skills) si on veut l'utiliser localement.