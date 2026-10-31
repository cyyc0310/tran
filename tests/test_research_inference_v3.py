import numpy as np
from scipy.stats import multivariate_normal, multivariate_t
from transcif.research.inference_v3 import CONFIGS, full_hessian, mixture_log_density, infer_v3


def test_full_hessian_recovers_rotated_gaussian_and_counts():
    angle=.63; rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    info=rotation@np.diag([12.,2.])@rotation.T; target=np.array([.2,-.3]); count=[0]
    def evaluate(z):
        count[0]+=len(z)
        return -.5*np.einsum('ni,ij,nj->n',z-target,info,z-target),z
    origin=np.array([.1,.1]); ll=evaluate(origin[None])[0][0]; count[0]=0
    mean,cov=full_hessian(evaluate,origin,ll)
    expected_cov=np.linalg.inv(np.eye(2)+info)
    np.testing.assert_allclose(cov,expected_cov,atol=1e-10)
    np.testing.assert_allclose(mean,expected_cov@info@target,atol=1e-10)
    assert count[0]==8


def test_complete_mixture_density():
    z=np.array([[0.,0.],[1.,-1.]])
    means=np.array([[.3,.5],[-.4,.1]]); covs=np.array([np.eye(2)*.5,np.array([[1.,.2],[.2,.7]])])
    masses=np.array([.4,.6]); defense=.35
    for student in (False,True):
        q=defense*multivariate_normal.pdf(z,mean=np.zeros(2))
        for mean,cov,mass in zip(means,covs,masses):
            density=multivariate_t.pdf(z,loc=mean,shape=cov,df=5) if student else multivariate_normal.pdf(z,mean=mean,cov=cov)
            q+=(1-defense)*mass*density
        np.testing.assert_allclose(np.exp(mixture_log_density(z,means,covs,masses,defense,student)),q)


def test_exact_cost_determinism_and_black_box_policy():
    # Plain function exposes neither active axes nor any case metadata.
    for config in CONFIGS:
        outputs=[]
        for _ in range(2):
            count=[0]
            def evaluate(z):
                count[0]+=len(z)
                return -.5*((z[:,0]**2-1)/.3)**2,np.tile(z[:,0,None],(1,24))
            result=infer_v3(evaluate,4,128,927,config)
            assert count[0]==result.evaluations<=128
            assert np.isclose(result.weights.sum(),1)
            assert np.isfinite(result.predictions).all()
            outputs.append(result)
        np.testing.assert_array_equal(outputs[0].z,outputs[1].z)
        np.testing.assert_array_equal(outputs[0].weights,outputs[1].weights)


def test_prior_target_weights_equal_and_geometry_infeasible_logged():
    def evaluate(z): return np.zeros(len(z)),z
    for name in ('sobol','prior','antithetic'):
        result=infer_v3(evaluate,3,128,7,name)
        np.testing.assert_allclose(result.weights,np.full(128,1/128))
    result=infer_v3(evaluate,16,128,7,'budget_policy')
    assert result.diagnostics['geometry_infeasible']
    assert result.evaluations==128
