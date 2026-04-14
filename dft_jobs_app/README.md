# DFT Jobs App

A self-contained web app for submitting, monitoring, and visualizing DFT jobs.
Jobs are enqueued to a Redis-backed RQ worker which simulates a 20-step SCF +
relax loop and streams progress over WebSockets to a Next.js UI.

## Architecture

```
 +-------------+        +--------------+        +----------------+
 |  Next.js    |  HTTP  |   FastAPI    |  SQL   |   PostgreSQL   |
 |  (web:3000) +<------>+  (api:8000)  +<------>+    (db:5432)   |
 |             |   WS   |              |        +----------------+
 +------+------+<------>+------+-------+
        ^                      |
        |                      | enqueue / publish
        |                      v
        |               +--------------+
        |   Redis       |  Redis 7     |
        +<--pubsub----->+  (redis:6379)|
                        +------+-------+
                               ^
                               | RQ consume + publish progress
                               |
                        +------+-------+
                        |  RQ worker   |
                        |  (worker)    |
                        +--------------+
```

## How to run

From this directory:

```bash
cd dft_jobs_app
docker-compose up --build
```

Then open http://localhost:3000 and click **New Job**.

Services:

| Service | URL                          |
| ------- | ---------------------------- |
| Web UI  | http://localhost:3000        |
| API     | http://localhost:8000        |
| Docs    | http://localhost:8000/docs   |
| Health  | http://localhost:8000/health |

Stop with `Ctrl+C`, clean state with `docker-compose down -v`.

## Local development (without Docker)

You need PostgreSQL 16 and Redis 7 running locally (`brew install postgresql@16
redis` or equivalent), with a `dftjobs` database owned by user `dft`.

### Backend

```bash
cd dft_jobs_app/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DATABASE_URL=postgresql+asyncpg://dft:dft@localhost:5432/dftjobs
export SYNC_DATABASE_URL=postgresql+psycopg2://dft:dft@localhost:5432/dftjobs
export REDIS_URL=redis://localhost:6379

# API
uvicorn app.main:app --reload --port 8000

# In another shell: worker
rq worker -u $REDIS_URL dft
```

### Frontend

```bash
cd dft_jobs_app/frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 \
NEXT_PUBLIC_WS_URL=ws://localhost:8000 \
npm run dev
```

## API reference

| Method | Path                  | Description                          |
| ------ | --------------------- | ------------------------------------ |
| POST   | `/api/jobs`           | Create a job (`name`, `formula`, `poscar`) and enqueue it |
| GET    | `/api/jobs`           | List jobs, newest first (`page`, `page_size`) |
| GET    | `/api/jobs/{id}`      | Fetch a single job                    |
| DELETE | `/api/jobs/{id}`      | Cancel and delete a job               |
| WS     | `/ws/jobs`            | Global stream of all job updates     |
| WS     | `/ws/jobs/{id}`       | Per-job update stream                 |
| GET    | `/health`             | Liveness check                        |

## Data model

A `Job` contains:

- `id` (uuid), `name`, `formula`, `poscar`
- `status`: `pending` | `running` | `completed` | `failed`
- `created_at`, `updated_at`, `started_at`, `finished_at`
- `energy` (final, eV), `error`
- `convergence`: JSON array of `{ step, energy, force }`
- `structure_xyz`: final geometry (XYZ) rendered in 3Dmol.js

## Worker behaviour

The RQ worker simulates a DFT relaxation: 20 iterations, 1 s each, with a
decaying energy `-10 - (1 - exp(-i/5)) * 5 + noise` and exponentially decaying
force. Each step is persisted and published to Redis pubsub (`job:{id}` and
`jobs:all`). There is a 5% chance the job fails mid-run. On success the worker
writes a mock H2O XYZ as the final structure.
