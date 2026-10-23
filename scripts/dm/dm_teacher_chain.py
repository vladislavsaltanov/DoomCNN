"""Runs teacher cycle candidates in order: an accepted one joins the base of the next; two rejected in a row stop.

    python scripts/dm/dm_teacher_chain.py --base "" "name|settings|why" ...
"""
import argparse
import subprocess
import sys

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')  # arrows in the rows; the Windows console is cp1251
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='')
    ap.add_argument('candidates', nargs='+')
    a = ap.parse_args()
    base, misses = a.base.split(), 0
    for c in a.candidates:
        name, sets, why = c.split('|')
        cand = base + sets.split()
        out = subprocess.run([sys.executable, 'scripts/dm/dm_teacher_cycle.py', '--name', name, '--base', ' '.join(base),
                              '--cand', ' '.join(cand), '--why', why], capture_output=True, text=True, encoding='utf-8')
        print(out.stdout.strip().split('\n')[-2:], flush=True)
        if 'ACCEPTED' in out.stdout:
            base, misses = cand, 0
        else:
            misses += 1
            if misses == 2:
                print('stop rule: two rejected in a row')
                break
    print('accepted teacher:', ' '.join(base) or '(defaults)')
