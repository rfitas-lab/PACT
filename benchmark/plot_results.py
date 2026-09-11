"""Vector publication figures generated only from executed benchmark outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
d=np.load(ROOT/'results/trajectories.npz')
s=json.loads((ROOT/'results/summary.json').read_text())
cfg=json.loads((ROOT/'protocol.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
 'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.6,'lines.linewidth':1.4,'savefig.bbox':'tight'})
blue='#25658a'; gold='#ab7e30'; gray='#686b6f'; red='#a84839'
methods=['point','parameter_only','unpaired','pact']
labels=['Mean + replay','Parameters, reset state','Unpaired states','PACT / Bayesian replay']
colors=[gray,red,gold,blue]

def title(ax,letter,text):
 ax.set_title(f'({letter})  {text}',loc='left',fontweight='bold',pad=9)

U=d['U']; last=241; ix=np.flatnonzero(np.isclose(U,.45))[0]
fig,axs=plt.subplots(2,2,figsize=(7.35,5.35),layout='constrained')
ax=axs[0,0];title(ax,'a','Source observations are identical')
ax.plot(d['source_q'],d['source_R'],color=blue,lw=2,label='Realization I')
ax.plot(d['source_q'],d['source_R'],color='black',lw=1,ls='--',label='Realization II')
ax.set(xlabel='Common source shortening q (mm)',ylabel='Total reaction (N)',xlim=(0,.84))
ax.legend(frameon=False,loc='upper left')
ax.text(.44,.44,'Maximum source difference: 0 N',transform=ax.transAxes,fontsize=6.8,ha='center',bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})

ax=axs[0,1];title(ax,'b','Asymmetric target separates them')
tr=d['truth'][:,:last]
ax.fill_between(U[:last],tr.min(axis=0),tr.max(axis=0),color=blue,alpha=.16,label='PACT alternatives')
ax.plot(U[:last],tr[0],color=blue,label='Realization I')
ax.plot(U[:last],tr[1],color='black',ls='--',label='Realization II')
ax.axvline(.45,color='.65',lw=.7,ls=':')
ax.plot([.45,.45],d['probe_values'],linestyle='none',marker='o',ms=3,color=gold)
ax.set(xlabel='Target frame approach U (mm)',ylabel='Target reaction (N)',xlim=(0,1.2))
ax.text(.03,.95,'Gains 1:1 → 1:2\nHistory retained; target reserved',transform=ax.transAxes,va='top',fontsize=7.3)
ax.text(.45,ax.get_ylim()[1]*.05,' probe',fontsize=7,rotation=90,va='bottom')

ax=axs[1,0];title(ax,'c','Unpaired marginals change the target law')
for i,(m,l,c) in enumerate(zip(methods,labels,colors)):
 y=3-i; values=d['prior_'+m][:,ix]; w=d['prior_'+m+'_weights']
 for val,weight in zip(values,w):
  ax.plot([val,val],[y-.19,y+.19],color=c,lw=.9)
  ax.scatter(val,y,s=150*weight,color=c,zorder=5,edgecolors='white',linewidths=.3)
for v in d['probe_values']:
 ax.axvline(v,color=blue,alpha=.22,lw=.8)
ax.set_yticks([3,2,1,0],labels=labels)
ax.set(xlabel='Reaction at U = 0.45 mm (N)',ylim=(-.6,3.6))
ax.tick_params(axis='y',length=0)
ax.spines['left'].set_visible(False)
ax.text(.99,.97,'Dot area = probability',ha='right',va='top',transform=ax.transAxes,fontsize=6.8)

ax=axs[1,1];title(ax,'d','Prediction improves at fixed information')
vals=[s['initial'][m]['energy_score_N'] for m in methods]
ax.barh([3,2,1,0],vals,color=colors,height=.55)
for y,v in zip([3,2,1,0],vals):ax.text(v+.015,y,f'{v:.3f}',va='center',fontsize=7.5)
ax.set_yticks([3,2,1,0],labels=['Mean + replay','State reset','States unpaired','PACT / replay'])
ax.set(xlabel='Expected trajectory energy score (N)',xlim=(0,1.04),ylim=(-.6,3.6))
ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
ax.text(.98,.02,'Lower is better · before the probe',transform=ax.transAxes,ha='right',fontsize=7)
fig.savefig(ROOT/'figures/transfer_test.pdf');fig.savefig(ROOT/'figures/transfer_test.svg')
fig.savefig(ROOT/'figures/transfer_test.png',dpi=190);plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(7.35,5.3),layout='constrained')
ax=axs[0,0];title(ax,'a','A physical handover without state reset')
ax.axis('off')
# Exact topology rendered as simple axial element symbols, not decorative imagery.
for x,lbl in [(0.23,'Element 1'),(.64,'Element 2')]:
 ax.plot([x,x],[.15,.32],color='black',lw=1)
 ax.add_patch(plt.Rectangle((x-.065,.32),.13,.32,fill=False,lw=1))
 ax.plot([x,x],[.64,.67],color='black',lw=1)
 ax.add_patch(plt.Rectangle((x-.055,.67),.11,.10,fill=False,lw=.8))
 ax.text(x,.72,'a1' if x<.4 else 'a2',ha='center',va='center',fontsize=6)
 ax.plot([x,x],[.77,.80],color='black',lw=1)
 ax.text(x,.48,lbl,ha='center',va='center',fontsize=7,rotation=90)
 ax.plot([x-.1,x+.1],[.15,.15],color='black',lw=1)
ax.plot([.23,.64],[.80,.80],color='black',lw=1)
ax.plot([.435,.435],[.80,.87],color='black',lw=1)
ax.add_patch(plt.Rectangle((.36,.87),.15,.12,fill=False,lw=1))
ax.text(.435,.93,'kf',ha='center',va='center',fontsize=7)
ax.annotate('',xy=(.435,.99),xytext=(.435,1.15),arrowprops={'arrowstyle':'->','lw':1})
ax.text(.53,1.11,'Imposed U',ha='left',fontsize=7)
ax.text(.53,.90,'kf = 20 N/mm',ha='left',fontsize=7)
ax.text(.5,.05,'Source: q1 = q2 = q    Target: qi = ai Q',ha='center',fontsize=8)
ax.text(.84,.47,'Source gains (1, 1)\nTarget gains (1, 2)\n\nZero-force handover\nSame elements',ha='left',va='center',fontsize=7)
ax.set(xlim=(0,1.58),ylim=(-.03,1.20))

ax=axs[0,1];title(ax,'b','One probe changes the probabilities')
y=np.linspace(.7,3.8,400);probe=d['probe_values'];sigma=cfg['probe_noise_sd']
lw=-.5*((y[:,None]-probe[None,:])/sigma)**2;w=np.exp(lw-lw.max(axis=1,keepdims=True));w/=w.sum(axis=1,keepdims=True)
ax.plot(y,w[:,0],color=blue,label='P(realization I | probe)')
ax.axhline(.5,color='.7',ls=':',lw=.8)
for j,v in enumerate(probe):
 ax.axvline(v,color=gold,lw=.8,ls='--');ax.text(v,1.06,'I' if j==0 else 'II',ha='center',fontsize=8)
ax.set(xlabel='Observed reaction at U = 0.45 mm (N)',ylabel='Posterior probability',ylim=(-.04,1.14),xlim=(.7,3.8))
ax.text(.04,.86,'Same likelihood for every method\nσ = 0.15 N',transform=ax.transAxes,fontsize=7)

ax=axs[1,0];title(ax,'c','Committed states continue through the cycle')
t=np.arange(len(U));
ax.plot(t,d['truth'][0],color='black',lw=1.6,label='Reference I')
ax.plot(t,d['post_pact_weights']@d['post_pact'],color=blue,ls='--',label='PACT after probe')
ax.plot(t,d['post_parameter_only_weights']@d['post_parameter_only'],color=red,ls=':',label='State reset')
ax.axvline(ix,color=gold,lw=.9)
ax.axvspan(0,ix,color='.5',alpha=.1)
ax.set(xlabel='Target increment',ylabel='Reaction (N)',xlim=(0,len(U)-1))
ax.legend(frameon=False,loc='upper right',fontsize=6.5)
ax.text(.43,.08,'load → unload → reload',transform=ax.transAxes,fontsize=7)

ax=axs[1,1];title(ax,'d','The mechanism persists across 27 cases')
df=pd.read_csv(ROOT/'results/robustness_sweep.csv')
base=df[df.method=='pact'].reset_index(drop=True)
for i,(m,c) in enumerate(zip(methods[:-1],colors[:-1])):
 x=df[df.method==m].reset_index(drop=True)
 gain=100*(1-base.energy_score_N/x.energy_score_N)
 ax.scatter(np.full(len(gain),i)+np.linspace(-.12,.12,len(gain)),gain,color=c,s=11,alpha=.7)
 ax.plot([i-.22,i+.22],[gain.median()]*2,color='black',lw=1.1)
ax.set_xticks(range(3),labels=['Mean +\nreplay','State\nreset','States\nunpaired'])
ax.set(ylabel='PACT energy-score reduction (%)',ylim=(0,100))
ax.text(.02,.03,'3 stiffness ratios × 3 gains × 3 preloads',transform=ax.transAxes,fontsize=6.8)
fig.savefig(ROOT/'figures/transfer_supplement.pdf');fig.savefig(ROOT/'figures/transfer_supplement.svg')
fig.savefig(ROOT/'figures/transfer_supplement.png',dpi=180);plt.close(fig)
print('Saved two vector figures and previews.')
