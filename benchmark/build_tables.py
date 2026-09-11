from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'results/summary.json').read_text())
labels={'point':'Mean parameters + replay','parameter_only':'Parameter-only, state reset','unpaired':'Unpaired states','pact':'PACT','bayes_replay':'Bayesian parameter replay'}
def fmt(x):
 if abs(x)<1e-12:return r'$<10^{-12}$'
 if abs(x)<.001:
  f,e=f'{x:.2e}'.split('e');return '$'+f+r'\times10^{'+str(int(e))+'}$'
 return f'{x:.3f}'

lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Exact expected prediction scores before the probe. RMSE and energy score average over both equally probable truth assignments. Coverage and width refer to central 90\% discrete-law marginal intervals.}',r'\label{tab:transfer_initial}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Representation & RMSE (N) & ES (N) & Coverage (\%) & Width (N) \\',r'\midrule']
for m,l in labels.items():
 a=s['initial'][m]; lines.append(l+f" & {a['RMSE_N']:.3f} & {a['energy_score_N']:.3f} & {100*a['coverage90']:.1f} & {a['width90_N']:.3f}"+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
(ROOT/'results/table_initial.tex').write_text('\n'.join(lines)+'\n')

lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{Post-probe prediction on 512 virtual records at each noise level. All methods receive the same posterior weights; only the 510 later increments are scored. Values are mean per-record scores. The complete-replay control coincides with PACT in every row.}',r'\label{tab:transfer_conditioned}',r'\begin{tabular}{@{}llrr@{}}',r'\toprule',r'Probe SD (N) & Representation & RMSE (N) & ES (N) \\',r'\midrule']
for sigma in [.05,.15,.3]:
 for i,m in enumerate(list(labels)[:-1]):
  a=s['conditioned'][str(sigma)][m];lines.append((f'{sigma:.2f}' if i==0 else '')+' & '+labels[m]+' & '+fmt(a['RMSE_N'])+' & '+fmt(a['energy_score_N'])+r' \\')
 if sigma!=.3:lines.append(r'\addlinespace')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
(ROOT/'results/table_conditioned.tex').write_text('\n'.join(lines)+'\n')
print('Generated SI tables from summary.json')
