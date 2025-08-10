import os, re, pandas as pd

# Very light parser: read energies from OUTCAR/optimizer log if exist

def parse_job_to_rows(local_dir: str):
    rows = []
    # Example parse from a made-up ase log csv if present
    csvp = os.path.join(local_dir, 'ase_results.csv')
    if os.path.exists(csvp):
        df = pd.read_csv(csvp)
        for _, r in df.iterrows():
            rows.append({
                'step': str(r.get('step', '')),
                'energy': float(r.get('energy', 'nan')),
                'info': {k: r[k] for k in df.columns if k not in ['step','energy']}
            })
        return rows
    # Fallback: OUTCAR last energy
    outcar = os.path.join(local_dir,'OUTCAR')
    if os.path.exists(outcar):
        E = None
        with open(outcar,'r',errors='ignore') as f:
            for line in f:
                if 'free  energy   TOTEN' in line:
                    m = re.search(r'TOTEN\s*=\s*([\-0-9\.]+)', line)
                    if m: E = float(m.group(1))
        if E is not None:
            rows.append({'step':'final','energy':E,'info':{}})
    return rows