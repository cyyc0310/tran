"""Budget-accounted mechanism search; no evaluator internals or known axes."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.special import ndtri
from scipy.stats import multivariate_t
from .inference import _evaluate, _normalize, _standard_log_density, _normal_log_density, _logsumexp
from .inference_v2 import InferenceV2Result, _uniforms, infer_v2

@dataclass(frozen=True)
class Config:
    name: str
    mechanism: str = 'sobol'
    pilot_fraction: float = .125
    centers: int = 1
    defense: float = .35
    student: bool = False

CONFIGS = (
    Config('sobol'), Config('prior', 'prior'), Config('antithetic', 'antithetic'),
    Config('full_hessian_p8', 'geometry'), Config('full_hessian_p4', 'geometry', .25),
    Config('multistart_student', 'geometry', .125, 2, .35, True),
    Config('budget_policy', 'policy', .125, 2, .5, True), Config('adaptive_smc', 'smc'),
)

def full_hessian(evaluate, origin, center_ll, h=.15):
    """Central mixed differences: exactly 2*d*d forward rows."""
    d = len(origin)
    eye = np.eye(d)*h
    points = [origin+x for x in eye] + [origin-x for x in eye]
    pairs = [(i,j) for i in range(d) for j in range(i+1,d)]
    for i,j in pairs:
        points.extend([origin+eye[i]+eye[j], origin+eye[i]-eye[j],
                       origin-eye[i]+eye[j], origin-eye[i]-eye[j]])
    ll, _ = _evaluate(evaluate, np.asarray(points))
    grad = (ll[:d]-ll[d:2*d])/(2*h)-origin
    hess = np.diag((ll[:d]-2*center_ll+ll[d:2*d])/h**2)
    offset = 2*d
    for i,j in pairs:
        a,b,c,e = ll[offset:offset+4]; offset += 4
        hess[i,j] = hess[j,i] = (a-b-c+e)/(4*h*h)
    precision = np.eye(d)-hess
    values, vectors = np.linalg.eigh(precision)
    values = np.clip(values, 1., 1e5)
    covariance = (vectors / values) @ vectors.T
    delta = covariance @ grad
    delta *= min(1., 1.5/max(np.linalg.norm(delta), 1e-12))
    return origin+delta, covariance


def mixture_log_density(z, means, covariances, masses, defense, student):
    terms = [np.log(defense)+_standard_log_density(z)]
    for mean,cov,mass in zip(means,covariances,masses):
        density = multivariate_t.logpdf(z, loc=mean, shape=cov, df=5) if student else _normal_log_density(z,mean,cov)
        terms.append(np.log((1-defense)*mass)+density)
    return _logsumexp(np.stack(terms), axis=0)


def _smc(evaluate,d,budget,seed):
    start=perf_counter(); rng=np.random.default_rng(seed)
    stages=8; n=budget//(stages+1)
    z=ndtri(_uniforms(n,d,rng,True)); ll,y=_evaluate(evaluate,z)
    w=np.full(n,1/n); beta=0.; temperatures=[]; accept=0
    for stage in range(stages):
        lo,hi=beta,1.
        for _ in range(32):
            mid=(lo+hi)/2; trial=_normalize(np.log(np.maximum(w,1e-300))+(mid-beta)*ll)
            if 1/np.square(trial).sum() < .7*n: hi=mid
            else: lo=mid
        nxt=1. if stage==stages-1 else max(beta+1e-8,lo)
        w=_normalize(np.log(np.maximum(w,1e-300))+(nxt-beta)*ll)
        idx=rng.choice(n,n,p=w); z,ll,y=z[idx],ll[idx],y[idx]
        w=np.full(n,1/n)
        # pCN proposal is reversible for standard-normal prior. Its likelihood
        # acceptance ratio preserves each tempered target without fitted kernels.
        rho=.85 if stage<4 else .98
        proposed=rho*z+np.sqrt(1-rho*rho)*rng.normal(size=z.shape)
        pll,py=_evaluate(evaluate,proposed)
        move=np.log(rng.random(n)) < np.minimum(0,nxt*(pll-ll))
        z[move],ll[move],y[move]=proposed[move],pll[move],py[move]
        accept+=int(move.sum()); beta=nxt; temperatures.append(beta)
    return InferenceV2Result(z,w,y,n*9,float(n),perf_counter()-start,ll,np.full(n,np.nan),0,
        {'mechanism':'adaptive ESS temperatures, fixed pCN','temperatures':temperatures,
         'counted_rows':n*9,'acceptance':accept/(n*8),'genealogical_ess_warning':True})


def infer_v3(evaluate, dimension, budget, seed, config):
    if isinstance(config,str): config=next(c for c in CONFIGS if c.name==config)
    if dimension<1 or budget<16: raise ValueError('positive dimension and budget >=16 required')
    if config.mechanism in ('sobol','prior'): return infer_v2(evaluate,dimension,budget,seed,config.mechanism)
    if config.mechanism=='smc': return _smc(evaluate,dimension,budget,seed)
    started=perf_counter(); rng=np.random.default_rng(seed); used=0
    diagnostics={'mechanism':config.mechanism,'pilot_rows':0,'geometry_rows':0}
    if config.mechanism=='antithetic':
        half=ndtri(_uniforms((budget+1)//2,dimension,rng,True))
        z=np.concatenate([half,-half])[:budget]; logq=_standard_log_density(z)
    else:
        pilot=max(8,int(budget*config.pilot_fraction)); per=2*dimension*dimension+1
        # Cost feasibility is assessed without running a pilot. The policy
        # retains >=half the full budget for posterior production samples.
        affordable=min(config.centers,max(0,(budget//2-pilot)//per))
        if affordable==0:
            diagnostics['geometry_infeasible']=True
            z=ndtri(_uniforms(budget,dimension,rng,True)); logq=_standard_log_density(z)
        else:
            pz=ndtri(_uniforms(pilot,dimension,rng,True)); pll,py=_evaluate(evaluate,pz); used=pilot
            diagnostics['pilot_rows']=pilot
            concentration=1/(pilot*np.square(_normalize(pll)).sum())
            diagnostics['pilot_ess_fraction']=float(concentration)
            if config.mechanism=='policy' and concentration>.35:
                # A pilot with weak likelihood information does not justify
                # paying for geometry. Independent final Sobol points preserve
                # the conditional SNIS target; pilot points are discarded.
                diagnostics['fallback']='diffuse likelihood'
                z=ndtri(_uniforms(budget-used,dimension,rng,True)); logq=_standard_log_density(z)
            else:
                centers=[]
                # Posterior-weighted prediction separation ignores latent-only
                # nuisance separation; candidate likelihood ranking remains fixed.
                scale=np.maximum(np.std(py,axis=0),1.)
                for idx in np.argsort(-pll):
                    if not centers or min(np.mean(((py[idx]-py[j])/scale)**2) for j in centers)>.2:
                        centers.append(int(idx))
                    if len(centers)==affordable: break
                means=[]; covs=[]; lm=[]
                for idx in centers:
                    mean,cov=full_hessian(evaluate,pz[idx],pll[idx]); used+=2*dimension*dimension
                    ml,_=_evaluate(evaluate,mean[None]); used+=1
                    means.append(mean); covs.append(cov)
                    lm.append(ml[0]+_standard_log_density(mean[None])[0]+.5*np.linalg.slogdet(cov)[1])
                masses=_normalize(np.asarray(lm)); n=budget-used
                u=_uniforms(n,dimension+2,rng,True)
                probs=np.r_[config.defense,(1-config.defense)*masses]
                assign=np.minimum(np.searchsorted(np.cumsum(probs),u[:,0]),len(probs)-1)
                z=ndtri(u[:,1:dimension+1])
                if config.student:
                    from scipy.stats import chi2
                    radial=np.sqrt(5/chi2.ppf(u[:,-1],5))
                for k,(mean,cov) in enumerate(zip(means,covs),1):
                    chosen=assign==k
                    local=z[chosen]
                    if config.student: local=local*radial[chosen,None]
                    z[chosen]=local@np.linalg.cholesky(cov).T+mean
                logq=mixture_log_density(z,means,covs,masses,config.defense,config.student)
                diagnostics.update(geometry_rows=used-pilot,centers=len(centers),defense=config.defense)
    ll,y=_evaluate(evaluate,z); w=_normalize(_standard_log_density(z)+ll-logq)
    diagnostics.update(counted_rows=used+len(z),final_rows=len(z))
    return InferenceV2Result(z,w,y,used+len(z),float(1/np.square(w).sum()),perf_counter()-started,ll,logq,used,diagnostics)
