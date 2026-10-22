"""Bounded NumPy J/K contraction checks using the same stored DF integrals."""
import numpy as np


def selected_jk(store,addresses,diagonal,dm,factor,selected,buffer_bytes=256*1024**2):
    """Check selected output rows/columns; this is not a CPU integral oracle.

    The byte budget bounds the raw row buffer. Density/factor/lookup arrays,
    selected-row intermediates and NumPy BLAS workspaces are additional.
    ``diagonal`` follows the stored format's full diagonal-shell-block convention.
    """
    nao=dm.shape[0]
    if (dm.shape!=(nao,nao) or factor.shape[0]!=nao
            or dm.dtype!=np.float64 or factor.dtype!=np.float64):
        raise ValueError('Expected consistent real FP64 density and factors')
    block=min(store.shape[0],int(buffer_bytes)//(8*store.shape[1]))
    if block<1: raise MemoryError('CPU oracle buffer cannot hold one auxiliary row')
    r,c=np.divmod(addresses,nao)
    lookup=np.full((nao,nao),-1,dtype=np.int32)
    position=np.arange(len(addresses),dtype=np.int32)
    lookup[r,c]=position; lookup[c,r]=position
    rows=lookup[selected,:]
    pairs=lookup[np.ix_(selected,selected)]
    row_valid=rows>=0; pair_valid=pairs>=0
    rows=np.maximum(rows,0); pairs=np.maximum(pairs,0)
    weights=2*dm[r,c]; weights[diagonal]*=.5
    out_j=np.zeros((len(selected),len(selected)))
    out_k=np.zeros_like(out_j)
    buffer=np.empty((block,store.shape[1]),dtype=np.float64)
    for start in range(0,store.shape[0],block):
        stop=min(start+block,store.shape[0])
        data=store.read_rows(start,stop,buffer[:stop-start])
        rho=data@weights
        values=data[:,pairs]*pair_valid
        out_j+=np.einsum('L,Lab->ab',rho,values)
        transformed=(data[:,rows]*row_valid)@factor
        out_k+=np.einsum('Lak,Lbk->ab',transformed,transformed)
    return out_j,out_k
