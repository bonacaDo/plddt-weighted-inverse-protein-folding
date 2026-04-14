# pLDDT-Weighted ProteinMPNN

Projekt Arnes Hackathon 2026, osredotočen na načrtovanje proteinskih zaporedij z upoštevanjem zanesljivosti struktur.

Ta repozitorij razširi ProteinMPNN z upoštevanjem pLDDT/B-faktorjev pri prenosu sporočil v grafu, tako da model bolj zaupa regijam z visoko zanesljivostjo kot regijam z nizko zanesljivostjo ali neurejenim delom strukture.

## Cilj projekta

ProteinMPNN vse ostanke v vhodni strukturi obravnava enakovredno. Pri AlphaFold strukturah je to pogosto neoptimalno, ker se zanesljivost med regijami močno razlikuje.

Naš pristop v cevovod prenosa sporočil neposredno vključi uteževanje po zanesljivosti:

1. Vdelava zanesljivosti vozlišč: vsak ostanek dobi naučeno vdelavo svoje zanesljivosti.
2. Uteževalna vrata povezav: vsako sporočilo po povezavi se mehko uteži na podlagi zanesljivosti obeh povezanih ostankov.

S tem model zmanjša vpliv šumnih regij brez trdega pragovanja, učenje pa ostane odvedljivo in primerno za prenosno učenje.

## Vsebina repozitorija

- `plddt_weighted/`: implementacija uteževanja po zanesljivosti, učni cevovod, skripte, evalvacija in SLURM opravila.
- `ProteinMPNN/`: originalna izvorna koda in sredstva za referenco ter združljivost.
- `streamlit_demo/`: spletna aplikacija za interaktivni demo načrtovanja zaporedij.
- `demo_cases/`: primeri testnih primerov.
- `outputs/`: generirani primeri in izhodi zagonov.

## Ključne spremembe modela

Glavne arhitekturne nadgradnje so implementirane v `plddt_weighted/training/model_utils.py`:

- Dodan MLP za vdelavo zanesljivosti posameznega ostanka.
- Dodan MLP za vrata povezav (sigmoid) za utežen prenos sporočil.
- Dodana različica kodirne plasti z uteževanjem po zanesljivosti.

Posodobitve pri nalaganju in predobdelavi podatkov vključujejo:

- Ekstrakcijo B-faktorjev iz PDB datotek (uporabljeni kot pLDDT pri AlphaFold strukturah).
- Normalizacijo zanesljivosti pred vhodom v model.
- Posredovanje tenzorjev zanesljivosti skozi poti za treniranje/inferenco.

## Potek podatkov

Uteženi učni cevovod v `plddt_weighted/` je optimiziran za uporabo na HPC:

1. Prenos CATH S40 domen/PDB datotek.
2. Predobdelava struktur v tenzorje `.pt`.
3. Izračun in normalizacija zanesljivosti (B-faktor/pLDDT).
4. Treniranje modela z uteževanjem po zanesljivosti (inicializiranega z utežmi osnovnega ProteinMPNN).
5. Evalvacija proti osnovnemu ProteinMPNN po binih zanesljivosti.

## Hiter začetek

### Možnost A: zagon Streamlit demo aplikacije

Iz korena repozitorija:

```bash
cd streamlit_demo
pip install -r requirements.txt
streamlit run app.py
```

### Možnost B: utežena demo inferenca

Iz korena repozitorija:

```bash
python plddt_weighted/scripts/demo_inference.py \
	--pdb_file plddt_weighted/data/alphafold_eval/AF-P04637-F1.pdb \
	--model_weights plddt_weighted/weights/epoch_last.pt \
	--temperature 0.1 \
	--num_sequences 3
```

Opomba: dejanska razpoložljivost podatkov in checkpointov je odvisna od tega, ali sta bila priprava podatkov in treniranje že izvedena.

## SLURM potek dela (Arnes HPC)

V mapi `plddt_weighted/` oddajte opravila v tem vrstnem redu:

1. Nastavitev okolja

```bash
sbatch slurm/setup_env.slurm
```

2. Priprava podatkov

```bash
sbatch slurm/prepare_data.slurm
```

3. Treniranje

```bash
sbatch slurm/train_pLDDT_results.slurm
```

4. Evalvacija

```bash
sbatch slurm/eval_pLDDT_results.slurm
```

5. Izris grafov

```bash
sbatch slurm/plot_pLDDT_results.slurm
```

## Evalvacija in figure

Skripte za evalvacijo so v `plddt_weighted/evaluation/`.

Uporabni ukazi:

```bash
python plddt_weighted/evaluation/plot_exported_results.py \
	--export_dir plddt_weighted/results/evaluation_pLDDT_results \
	--output_dir plddt_weighted/figures/generated_pLDDT_results
```

```bash
python plddt_weighted/evaluation/plot_results.py --mock --output_dir plddt_weighted/figures/
```

## Zakaj mehka vrata zanesljivosti

Mehko uteževanje povezav smo izbrali pred trdim maskiranjem in preprostim dodajanjem značilk, ker:

- Ohranja odvedljivost in stabilno optimizacijo.
- Omogoča prenosno učenje iz osnovnih ProteinMPNN checkpointov.
- Nauči kontekstno odvisno vedenje zanesljivosti na posamezni povezavi namesto fiksnih pragov.

## Opombe o repozitoriju

- Nekateri veliki podatkovni nabori in generirani artefakti so namenoma izpuščeni iz GitHuba in se ustvarijo/prenesejo s skriptami.
- `plddt_weighted/weights/epoch_last.pt` se uporablja kot priročen lokalni checkpoint za demo in rezervno evalvacijo.

## Zahvale

- Originalni ProteinMPNN: https://github.com/dauparas/ProteinMPNN
- Ta projekt nadgradi to arhitekturo za načrtovanje zaporedij z upoštevanjem zanesljivosti.
