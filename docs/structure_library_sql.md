# Structure Library — SQL Retrieval Guide

Connect:
```bash
psql $DATABASE_URL
# or
psql -h localhost -U chatdft -d chatdft
```

---

## Schema

```sql
\d structure_library
```

| Column          | Type      | Description                                      |
|-----------------|-----------|--------------------------------------------------|
| id              | integer   | Primary key                                      |
| session_id      | integer   | Linked ChatDFT session (nullable)                |
| structure_type  | text      | `surface` \| `molecule` \| `adsorption`         |
| label           | text      | Human-readable name, e.g. `Pt(111)-4x4x3`       |
| formula         | text      | Chemical formula, e.g. `Pt48`                    |
| smiles          | text      | SMILES string (molecules only)                   |
| description     | text      | Rich natural-language description (T2S training) |
| ase_code        | text      | Reproducible Python/ASE code                     |
| poscar          | text      | Full VASP POSCAR content                         |
| plot_png_b64    | text      | Base64-encoded PNG image                         |
| meta            | jsonb     | Extra metadata (facet, layers, site_type, …)    |
| created_at      | timestamp | When this entry was created                      |

---

## Common Queries

### 1 — List all structures (summary, no large text columns)
```sql
SELECT id, structure_type, label, formula, session_id, created_at
FROM structure_library
ORDER BY created_at DESC;
```

### 2 — All metal surfaces
```sql
SELECT id, label, formula,
       meta->>'element'      AS element,
       meta->>'surface_type' AS facet,
       meta->>'vacuum'       AS vacuum_ang
FROM structure_library
WHERE structure_type = 'surface'
ORDER BY created_at DESC;
```

### 3 — All gas-phase molecules
```sql
SELECT id, label, formula, smiles, meta->>'n_atoms' AS n_atoms
FROM structure_library
WHERE structure_type = 'molecule'
ORDER BY label;
```

### 4 — All adsorption configurations for a specific molecule (e.g. C4H10)
```sql
SELECT id, label, formula,
       meta->>'site_type' AS site,
       meta->>'rotation'  AS rot_deg,
       meta->>'height'    AS height_ang,
       meta->>'n_atoms'   AS n_atoms
FROM structure_library
WHERE structure_type = 'adsorption'
  AND label ILIKE '%C4H10%'
ORDER BY id;
```

### 5 — All configurations on a specific surface (Pt111)
```sql
SELECT id, label, meta->>'site_type' AS site
FROM structure_library
WHERE structure_type = 'adsorption'
  AND description ILIKE '%Pt(111)%'
ORDER BY id;
```

### 6 — Retrieve POSCAR for a specific entry (e.g. id = 3)
```sql
SELECT label, poscar
FROM structure_library
WHERE id = 3;
```

### 7 — Retrieve the ASE code to reproduce a structure
```sql
SELECT label, ase_code
FROM structure_library
WHERE id = 3;
```

### 8 — Read the full description (T2S training text)
```sql
SELECT label, description
FROM structure_library
WHERE id = 3;
```

### 9 — Text search in descriptions (e.g. "hollow site")
```sql
SELECT id, label, structure_type,
       LEFT(description, 120) AS desc_preview
FROM structure_library
WHERE description ILIKE '%hollow site%'
   OR label       ILIKE '%hollow%';
```

### 10 — All structures from a specific session
```sql
SELECT id, structure_type, label, formula
FROM structure_library
WHERE session_id = 5
ORDER BY structure_type, created_at;
```

### 11 — Export all POSCARs to CSV (for batch download)
```sql
\COPY (
  SELECT id, label, structure_type, poscar
  FROM structure_library
  ORDER BY id
) TO '/tmp/poscar_export.csv' CSV HEADER;
```

### 12 — Export all descriptions + ASE codes to CSV (T2S dataset)
```sql
\COPY (
  SELECT id, structure_type, label, formula, smiles,
         description, ase_code
  FROM structure_library
  ORDER BY structure_type, id
) TO '/tmp/t2s_dataset.csv' CSV HEADER;
```

### 13 — Count entries by type
```sql
SELECT structure_type, COUNT(*) AS n
FROM structure_library
GROUP BY structure_type
ORDER BY n DESC;
```

### 14 — Find duplicate labels (sanity check)
```sql
SELECT label, COUNT(*) AS n
FROM structure_library
GROUP BY label
HAVING COUNT(*) > 1
ORDER BY n DESC;
```

### 15 — Show full entry as JSON (handy for debugging)
```sql
SELECT row_to_json(structure_library) FROM structure_library WHERE id = 1;
```

---

## Meta JSON field queries

The `meta` column is JSONB, so you can filter on nested fields:

```sql
-- All slabs with 4 layers
SELECT label FROM structure_library
WHERE meta->>'nlayers' = '4'
  AND structure_type = 'surface';

-- Adsorption configs with FCC hollow site
SELECT label FROM structure_library
WHERE meta->>'site_type' = 'hollow_fcc';

-- Configs at height > 2.5 Å
SELECT label, meta->>'height' FROM structure_library
WHERE (meta->>'height')::float > 2.5;
```

---

## Quick psql tips

```bash
# turn on expanded output for long text
\x

# show just the POSCAR for id=1
SELECT poscar FROM structure_library WHERE id=1 \gx

# pipe ASE code to a Python file
psql $DATABASE_URL -t -c "SELECT ase_code FROM structure_library WHERE id=2;" > rebuild.py
python rebuild.py
```
