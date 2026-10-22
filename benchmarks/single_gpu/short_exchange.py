"""Reversible large-system short-block exchange candidate."""
from contextlib import contextmanager
from unittest.mock import patch

from exchange_layout import eligible,transform


@contextmanager
def version():
    from gpu4pyscf.df import df_jk
    original=df_jk._exchange_transform
    def exchange(a,b,out):
        if (df_jk.contract_engine=='cupy' and a.shape[1]>=1024
                and 32<=a.shape[0]<256 and b.shape[2]>=128
                and eligible(a,b,out)):
            return transform(a,b,out)
        return original(a,b,out)
    with patch.object(df_jk,'_exchange_transform',exchange):
        yield
