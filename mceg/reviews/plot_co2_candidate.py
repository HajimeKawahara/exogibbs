"""Plot archived CO2 thermal and finite-response diagnostics without new solves."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]

def main() -> None:
    """Write a standalone figure from the frozen candidate and equilibrium data."""
    rows=json.loads((ROOT/'tests/data/validation/co2_candidate/20261104/validation.json').read_text())['thermal_diagnostics']
    t=np.array([r['temperature_K'] for r in rows]);ratio=np.array([r['candidate_over_current_K'] for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    axes[0].plot(t,100*(ratio-1),label='NIST 626 / current')
    axes[0].set(xlabel='Temperature [K]',ylabel='Formation K change [%]',title='Fixed energy and atomic reference')
    axes[0].grid(alpha=.25);axes[0].legend()
    data=ROOT/'tests/data/validation/co2_equilibrium_impact/20261104'
    for name in ('original','hydrogen_poor'):
        with np.load(data/(name+'.npz')) as d:
            c=d['species'].tolist().index('C1O2');x=np.exp(d['ln_n']-logsumexp(d['ln_n'],axis=-1,keepdims=True));delta=x[1,:,c]-x[0,:,c]
            if name=='original':
                ts=np.unique(d['temperature_K']);ys=[max(delta[d['temperature_K']==v]) for v in ts]
                axes[1].semilogy(ts,ys,'o--',label='Original grid: maximum at each T')
            else:
                for p in (.1,100.):
                    mask=d['pressure_bar']==p
                    axes[1].semilogy(d['temperature_K'][mask],delta[mask],'o-',label=f'H-poor gas: {p:g} bar')
    axes[1].set(xlabel='Temperature [K]',ylabel='Absolute CO2 mole-fraction increase',title='Response depends on composition')
    axes[1].grid(alpha=.25);axes[1].legend(fontsize=8)
    fig.savefig(ROOT/'documents/_static/co2_candidate_response.png',dpi=180)
    plt.close(fig)

if __name__=='__main__': main()
