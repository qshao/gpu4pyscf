"""Bounded CPU contraction oracle covers screened pairs and full shell blocks."""
import numpy as np
import pytest


def test_selected_jk_matches_dense_tensor():
    from selected_jk import selected_jk
    rng=np.random.default_rng(17)
    nao,naux=9,7
    full=rng.normal(size=(naux,nao,nao)); full=(full+full.transpose(0,2,1))*.5
    full[:,1,4]=full[:,4,1]=0.
    rows,cols=np.tril_indices(nao)
    keep=~((rows==4)&(cols==1)); rows,cols=rows[keep],cols[keep]
    rows=np.append(rows,[0,2]); cols=np.append(cols,[1,3])
    order=rng.permutation(len(rows)); rows,cols=rows[order],cols[order]
    addresses=rows*nao+cols
    diagonal=np.where((rows==cols)|((rows//2==cols//2)&(np.maximum(rows,cols)<4)))[0]
    packed=np.ascontiguousarray(full[:,rows,cols])
    class Store:
        shape=packed.shape
        def read_rows(self,a,b,out): out[:]=packed[a:b]; return out
    factor=rng.normal(size=(nao,3)); dm=factor@factor.T
    selected=np.array([1,4,7])
    j,k=selected_jk(Store(),addresses,diagonal,dm,factor,selected,buffer_bytes=2*packed.shape[1]*8)
    rho=np.einsum('Lpq,pq->L',full,dm)
    ej=np.einsum('L,Lpq->pq',rho,full)[np.ix_(selected,selected)]
    ek=np.einsum('Lpi,ij,Lqj->pq',full,dm,full)[np.ix_(selected,selected)]
    np.testing.assert_allclose(j,ej,atol=1e-10,rtol=0)
    np.testing.assert_allclose(k,ek,atol=1e-10,rtol=0)
    with pytest.raises(MemoryError):
        selected_jk(Store(),addresses,diagonal,dm,factor,selected,buffer_bytes=8)
