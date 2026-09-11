"""Matched-information transfer in a three-element elastoplastic assembly.

Two compression-only, isotropically hardening elements act through ideal
kinematic gains; an elastic loading frame is in series. This is a discrete
mechanical / one-dimensional FE benchmark, not a corrugated-shell simulation.
Every target increment solves equilibrium and commits plastic state only once.
An independently coded event-based affine oracle verifies the numerical solve.
"""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'protocol.json').read_text())
METHODS = ['point','parameter_only','unpaired','pact','bayes_replay']

def path(legs):
    return np.concatenate([np.linspace(a,b,int(n)) if i==0 else np.linspace(a,b,int(n))[1:]
                           for i,(a,b,n) in enumerate(legs)])

def local(q, parent, theta):
    """Return (reaction, tangent, trial p); no mutation of parent."""
    k,Y,H = np.asarray(theta).T
    ftrial = k*(q-parent)
    dp = np.maximum(ftrial-Y-H*parent,0)/(k+H)
    p = parent+dp
    f = np.maximum(k*(q-p),0)
    t = np.where(ftrial<=0,0,np.where(dp>0,k*H/(k+H),k))
    return f,t,p

def preload(theta,h,n=81):
    hist = path([[0,h,n],[h,0,n]])
    p=np.zeros(2); forces=[]; states=[]
    for q in hist:
        f,_,p=local(np.full(2,q),p,theta)
        forces.append(f.sum()); states.append(p.copy())
    return hist,np.array(forces),p,np.array(states)

def numerical(theta,p0,U,gains,kf=20.0):
    p=np.array(p0,copy=True); rows=[]
    for u in U:
        saved=p.copy()
        def residual(Q):
            f,_,_=local(gains*Q,p,theta)
            return kf*(Q-u)+gains@f
        Q = 0.0 if u==0 else brentq(residual,0.0,float(u),xtol=2e-14,rtol=1e-14)
        f,t,new=local(gains*Q,p,theta)
        assert np.array_equal(p,saved), 'Trial mutated committed state'
        Kport=(gains*gains)@t
        rows.append([u,Q,gains@f,kf*Kport/(kf+Kport),residual(Q),*new])
        p=new
    return np.array(rows)

def oracle(theta,p0,U,gains,kf=20.0):
    """Independent exact active-interval inversion of U=Q+R(Q)/kf.

    On each interval between engagement/yield events the port reaction is
    A*Q+B. Direct inversion gives Q=(kf*U-B)/(kf+A); no root solver or local()
    call is used. Plastic state is computed from the maximum accepted Q.
    """
    k,Y,H=np.asarray(theta).T
    gains=np.asarray(gains); p=np.array(p0,copy=True)
    out=[]; qmax=np.zeros(2)
    for u in U:
        e=p/gains
        y=(p+(Y+H*p)/k)/gains
        breaks=np.unique(np.r_[0,e,y,np.inf])
        found=False
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            mid=(lo+hi)/2 if np.isfinite(hi) else lo+1
            eng=mid>e; plast=mid>y
            A=np.sum(np.where(eng,np.where(plast,gains*gains*k*H/(k+H),gains*gains*k),0))
            B=np.sum(np.where(eng,np.where(plast,gains*k*Y/(k+H),-gains*k*p),0))
            Q=(kf*u-B)/(kf+A)
            if lo-2e-12<=Q<=hi+2e-12:
                found=True; break
        if not found: raise RuntimeError('Oracle did not find active interval')
        Q=max(0,Q)
        qmax=np.maximum(qmax,gains*Q)
        p=np.maximum(p0,np.maximum((k*qmax-Y)/(k+H),0))
        f=np.maximum(k*(gains*Q-p),0)
        out.append([u,Q,gains@f,*p])
    return np.array(out)

def energy_balance(theta,p0,U,gains,kf):
    """Exact port work by splitting each straight loading segment at events."""
    k,Y,H=theta.T
    p=np.array(p0,copy=True); Qold=0.0; Rold=0.0
    E0=.5*np.sum(H*p*p); work=0.; dissip=0.; maxerr=0.
    for u in U:
        row=oracle(theta,p,[u],gains,kf)[0]
        Q=row[1]; pnew=row[3:]; R=row[2]
        ev=np.r_[p/gains,(p+(Y+H*p)/k)/gains]
        if Q>=Qold:
            nodes=np.r_[Qold,np.sort(ev[(ev>Qold+1e-13)&(ev<Q-1e-13)]),Q]
        else:
            ev=p/gains
            nodes=np.r_[Qold,np.sort(ev[(ev>Q+1e-13)&(ev<Qold-1e-13)])[::-1],Q]
        forces=[]
        for q in nodes:
            f,_,_=local(gains*q,p,theta)
            forces.append(gains@f)
        forces=np.array(forces)
        work+=np.sum(.5*(forces[1:]+forces[:-1])*np.diff(nodes))+(R*R-Rold*Rold)/(2*kf)
        dissip+=Y@(pnew-p)
        E=.5*np.sum(k*np.maximum(gains*Q-pnew,0)**2+H*pnew*pnew)+R*R/(2*kf)
        maxerr=max(maxerr,abs(work-(E-E0)-dissip))
        p=pnew; Qold=Q; Rold=R
    return {'max_energy_residual_Nmm':float(maxerr),'work_Nmm':float(work),'dissipation_Nmm':float(dissip)}

def rms(x,axis=None): return np.sqrt(np.mean(np.asarray(x)**2,axis=axis))

def score(curves,weights,truth):
    weights=np.asarray(weights); weights=weights/weights.sum()
    mu=weights@curves
    es=float(weights@rms(curves-truth,axis=1))
    es-=.5*float(weights @ rms(curves[:,None,:]-curves[None,:,:],axis=2) @ weights)
    assert es>=-1e-12, 'Energy score outside numerical tolerance'
    es=max(0.0,es)
    order=np.argsort(curves,axis=0)
    sorted_vals=np.take_along_axis(curves,order,axis=0)
    cum=np.cumsum(weights[order],axis=0)
    low=sorted_vals[np.argmax(cum>=.05-1e-14,axis=0),np.arange(curves.shape[1])]
    high=sorted_vals[np.argmax(cum>=.95-1e-14,axis=0),np.arange(curves.shape[1])]
    return np.array([rms(mu-truth),es,np.mean((truth>=low-1e-9)&(truth<=high+1e-9)),np.mean(high-low)])

def family(theta,h,gains,U,kf):
    T=np.stack([theta,theta[::-1]])
    P=np.array([preload(t,h)[2] for t in T])
    truth=np.array([oracle(t,p,U,gains,kf)[:,2] for t,p in zip(T,P)])
    reset=np.array([oracle(t,np.zeros(2),U,gains,kf)[:,2] for t in T])
    unpaired=np.array([oracle(t,p,U,gains,kf)[:,2] for t in T for p in P])
    replay=np.array([oracle(t,preload(t,h)[2],U,gains,kf)[:,2] for t in T])
    return T,P,truth,reset,unpaired,replay

def atomic_predictions(fam,w,U,gains,kf,h):
    T,P,truth,reset,unpaired,replay=fam
    theta=np.einsum('i,ijk->jk',w,T)
    point=oracle(theta,preload(theta,h)[2],U,gains,kf)[:,2][None,:]
    return {'point':(point,np.ones(1)), 'parameter_only':(reset,w),
            'unpaired':(unpaired,np.outer(w,w).ravel()),
            'pact':(truth,w),'bayes_replay':(replay,w)}

def posterior(y,probe,sigma):
    logw=-.5*((probe-y)/sigma)**2
    w=np.exp(logw-logw.max()); return w/w.sum()

def run():
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    theta=np.array(CFG['members']); h=CFG['preload']; gains=np.array(CFG['gains']); kf=CFG['frame_stiffness']
    U=path(CFG['target_legs']); probeidx=int(np.flatnonzero(np.isclose(U,CFG['probe_displacement']))[0])
    fam=family(theta,h,gains,U,kf); T,P,truth,reset,unpaired,replay=fam
    q,F0,_,_=preload(theta,h); _,F1,_,_=preload(theta[::-1],h)
    checks={'source_permutation_error_N':float(np.max(abs(F0-F1))),
            'bayesian_replay_error_N':float(np.max(abs(truth-replay))),
            'numerical_oracle_max_error_N':0., 'equilibrium_max_residual_N':0.,
            'tangent_max_relative_error':0., 'increment_halving_error_N':0.}
    for j in range(2):
        r=numerical(T[j],P[j],U,gains,kf)
        checks['numerical_oracle_max_error_N']=max(checks['numerical_oracle_max_error_N'],float(np.max(abs(r[:,2]-truth[j]))))
        checks['equilibrium_max_residual_N']=max(checks['equilibrium_max_residual_N'],float(np.max(abs(r[:,4]))))
        fineU=np.ravel(np.column_stack([U[:-1],.5*(U[:-1]+U[1:])]))
        fineU=np.r_[fineU,U[-1]]
        refined=numerical(T[j],P[j],fineU,gains,kf)
        checks['increment_halving_error_N']=max(checks['increment_halving_error_N'],float(np.max(abs(r[:,2]-refined[::2,2]))))
        for idx in [40,100,180,300,480,550]:
            parent=P[j] if idx==0 else r[idx-1,5:]
            u=U[idx]; eps=1e-7
            plus=numerical(T[j],parent,[u+eps],gains,kf)[0,2]
            minus=numerical(T[j],parent,[u-eps],gains,kf)[0,2]
            checks['tangent_max_relative_error']=max(checks['tangent_max_relative_error'],float(abs((plus-minus)/(2*eps)-r[idx,3])/max(1,abs(r[idx,3]))))
        checks[f'energy_branch_{j}']=energy_balance(T[j],P[j],U,gains,kf)
    sym=family(theta,h,np.ones(2),U,kf)
    virgin=family(theta,0,gains,U,kf)
    identical=family(np.array([theta[0],theta[0]]),h,gains,U,kf)
    checks['symmetric_target_branch_gap_N']=float(np.max(abs(sym[2][0]-sym[2][1])))
    checks['identical_members_branch_gap_N']=float(np.max(abs(identical[2][0]-identical[2][1])))
    checks['virgin_reset_pact_error_N']=float(np.max(abs(virgin[2]-virgin[3])))
    assert checks['source_permutation_error_N']<1e-12
    assert checks['bayesian_replay_error_N']<1e-12
    assert checks['numerical_oracle_max_error_N']<1e-9
    assert checks['equilibrium_max_residual_N']<1e-9
    assert checks['tangent_max_relative_error']<1e-6
    assert max(checks[f'energy_branch_{j}']['max_energy_residual_Nmm'] for j in [0,1])<1e-9
    for key in ['symmetric_target_branch_gap_N','identical_members_branch_gap_N','virgin_reset_pact_error_N']:
        assert checks[key]<1e-10,key
    w=np.array([.5,.5]); initial=atomic_predictions(fam,w,U,gains,kf,h)
    initial_scores={}
    for m,(curves,weights) in initial.items():
        sc=np.mean([score(curves,weights,truth[j]) for j in [0,1]],axis=0)
        initial_scores[m]={**dict(zip(['RMSE_N','energy_score_N','coverage90','width90_N'],map(float,sc))),
                           'conditional_mean_error_N':float(rms(weights@curves-w@truth))}
    rng=np.random.default_rng(CFG['seed']); n=CFG['virtual_records']
    labels=rng.integers(0,2,n); noise=rng.standard_normal(n); probe=truth[:,probeidx]
    after=slice(probeidx+1,None); rows=[]; representative={}; boot={}; pooled={}
    for sigma in CFG['noise_sensitivity']:
        scores_by_m={m:[] for m in METHODS}; means_by_m={m:[] for m in METHODS}
        for i in range(n):
            y=probe[labels[i]]+sigma*noise[i]; w=posterior(y,probe,sigma)
            forecasts=atomic_predictions(fam,w,U,gains,kf,h)
            for m,(curves,weights) in forecasts.items():
                s=score(curves[:,after],weights,truth[labels[i],after])
                cmerr=float(rms(weights@curves[:,after]-w@truth[:,after]))
                scores_by_m[m].append(s); means_by_m[m].append(cmerr)
                rows.append([sigma,i,int(labels[i]),float(y),float(w[0]),m,*s,cmerr])
        pooled[str(sigma)]={}
        for m in METHODS:
            means=np.mean(scores_by_m[m],axis=0)
            pooled[str(sigma)][m]={**dict(zip(['RMSE_N','energy_score_N','coverage90','width90_N'],map(float,means))),
                                   'conditional_mean_error_N':float(np.mean(means_by_m[m]))}
        if np.isclose(sigma,CFG['probe_noise_sd']):
            brng=np.random.default_rng(CFG['seed']+1)
            inds=brng.integers(0,n,(2000,n))
            for m in METHODS:
                base=np.array(scores_by_m[m])[:,1]; pa=np.array(scores_by_m['pact'])[:,1]
                improvement=(np.mean(base[inds],axis=1)-np.mean(pa[inds],axis=1))
                boot[m]={'energy_score_difference_N':float(np.mean(base-pa)),
                         'paired_95_percent_interval_N':np.quantile(improvement,[.025,.975]).tolist()}
    sweep=[]
    for ratio in CFG['sweep']['stiffness_ratios']:
      for a2 in CFG['sweep']['target_gains']:
       for hh in CFG['sweep']['preloads']:
        th=np.array([[4*ratio,.1*4*ratio,.05*4*ratio],[4,2,.4]])
        gg=np.array([1,a2]); ff=family(th,hh,gg,U,kf)
        pred=atomic_predictions(ff,np.array([.5,.5]),U,gg,kf,hh)
        for m,(cc,ww) in pred.items():
            sc=np.mean([score(cc,ww,ff[2][j]) for j in [0,1]],axis=0)
            sweep.append([ratio,a2,hh,m,*sc,rms(ww@cc-ff[2].mean(axis=0))])
    wprobe=posterior(probe[0],probe,CFG['probe_noise_sd'])
    postrep=atomic_predictions(fam,wprobe,U,gains,kf,h)
    arrays={'U':U,'truth':truth,'states_at_transfer':P,'source_q':q,'source_R':F0,
            'probe_weights':wprobe,'probe_values':probe}
    for prefix,pred in [('prior',initial),('post',postrep)]:
        for m,(cc,ww) in pred.items(): arrays[prefix+'_'+m]=cc; arrays[prefix+'_'+m+'_weights']=ww
    np.savez_compressed(out/'trajectories.npz',**arrays)
    with (out/'virtual_records.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['sigma_N','record','truth_branch','probe_N','weight_A','method','RMSE_N','energy_score_N','coverage90','width90_N','conditional_mean_error_N']);writer.writerows(rows)
    with (out/'robustness_sweep.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['stiffness_ratio','gain_2','preload_mm','method','RMSE_N','energy_score_N','coverage90','width90_N','conditional_mean_error_N']);writer.writerows(sweep)
    diameter=float(rms(truth[0]-truth[1]))
    summary={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
             'preload_states_mm':P.tolist(),'probe_values_N':probe.tolist(),
             'probe_posterior_A':wprobe.tolist(),'trajectory_diameter_N':diameter,
             'minimax_point_error_bound_N':diameter/2,
             'maximum_branch_separation_N':float(np.max(abs(truth[0]-truth[1])),),
             'initial':initial_scores,'conditioned':pooled,'paired_bootstrap':boot,
             'checks':checks,'sweep_cases':len(sweep)//len(METHODS),'virtual_records_per_noise':n}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': run()
