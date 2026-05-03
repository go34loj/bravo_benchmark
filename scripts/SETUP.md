# Dataset Generation Pipeline Setup

Compact setup for generating `scripts/unified_database.db`.

## 1) Clone and open the scripts folder

```bash
git clone <repo-url>
cd <repo-name>/scripts
```

## 2) Create and activate Conda environment

```bash
conda create -n aqa-data python=3.11 -y
conda activate aqa-data
```

## 3) Install required packages

```bash
conda install -c conda-forge pandas pyyaml -y
```

Built-in modules used by scripts: `sqlite3`, `json`, `argparse`, `pathlib`.

## 4) (Recommended) Prepare data before generation

```bash
python data_manipulation.py
```

This updates `templates_updated.csv` and refreshes scene/template links.

## 5) (Optional) Review config

Default config file: `scripts/config.yaml`.

Typical settings:
- `inputs`: CSV filenames/paths
- `scripts.output_sqlite`: output DB path
- `execution`: enable/disable pipeline stages
- GT auto-run flags:
  - `run_scene_perception_gt: true`
  - `run_compliance_gt: true`

## 6) Generate unified database

```bash
python csv_unifier.py
```

This runs schema-first DB creation (`PRIMARY KEY`/`FOREIGN KEY` + integrity checks), then generates dataset tables.

## 7) Optional explicit GT scripts

Use this only if GT flags are disabled in `config.yaml`:

```bash
python utils/scene_perc_gen_gt.py --db unified_database.db
python utils/compliance_bool_gt_gen.py --db unified_database.db --table-name generated_compliance_questions
```

