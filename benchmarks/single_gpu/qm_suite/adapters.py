"""Native QM adapters. Importing this module never initializes CUDA.

CPU references build independent PySCF objects. The shared guess is an atomic
MINAO density, never a converged GPU state. Capability execution is a real tiny
calculation, not evidence that a full campaign case is qualified.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import importlib
from pathlib import Path
from time import perf_counter


class UnsupportedTask(RuntimeError):
    """The requested physical model/task is unavailable for this adapter."""


class AdapterNotImplemented(UnsupportedTask):
    """The suite has no adapter yet; this says nothing about library support."""


class DependencyMissing(RuntimeError):
    """A required optional scientific package cannot be imported."""


_DISPERSION_PARAMETER_SET = {
    '0.20*HF + 0.08*LDA_X + 0.72*GGA_X_B88, 0.81*GGA_C_LYP + 0.19*LDA_C_VWN': 'b3lyp',
}
_IMPLEMENTED = {'energy_gradient', 'hessian', 'tda', 'tddft', 'dfmp2', 'dfump2', 'pbc_energy', 'ccsd', 'polarizability', 'shielding', 'optimization'}


def _import(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise DependencyMissing(f'{name}: {exc}') from exc


def _class(obj):
    return type(obj).__module__ + '.' + type(obj).__name__


def _host(value):
    import numpy as np
    if hasattr(value, 'get'):
        value = value.get()
    return np.asarray(value)


def _array(value):
    import numpy as np
    array = _host(value)
    if np.iscomplexobj(array):
        return {'real': array.real.tolist(), 'imag': array.imag.tolist()}
    return array.tolist()


def _settings(mf, case, backend='cpu'):
    settings = case.get('settings', {})
    for key, default in [('conv_tol', 1e-10), ('conv_tol_grad', 1e-7), ('max_cycle', 100)]:
        setattr(mf, key, settings.get(key, default))
    if hasattr(mf, 'grids'):
        gen_grid = _import(('pyscf' if backend == 'cpu' else 'gpu4pyscf') + '.dft.gen_grid')
        prune = settings.get('prune', 'nwchem')
        if prune != 'nwchem':
            raise UnsupportedTask(f'unimplemented grid pruning policy {prune}')
        mf.grids.level = settings.get('grid_level', 3)
        mf.grids.prune = gen_grid.nwchem_prune
        mf.small_rho_cutoff = settings.get('small_rho_cutoff', 0.0)
        if hasattr(mf, 'nlcgrids'):
            mf.nlcgrids.level = case.get('task_options', {}).get('nlc_grid_level', settings.get('grid_level', 3))
            mf.nlcgrids.prune = gen_grid.nwchem_prune
    if hasattr(mf, 'cphf_grids'):
        mf.cphf_grids = mf.grids
    mf.conv_tol_cpscf = case.get('task_options', {}).get('conv_tol_cpscf', 1e-10)


def _configure_gpu_df(mf, case):
    """Apply storage only when the imported implementation exposes that policy."""
    fitted = mf.with_df
    options = case.get('task_options', {})
    settings = case.get('settings', {})
    storage = options.get('storage', settings.get('storage', 'resident'))
    if storage not in {'resident', 'auto', 'disk'}:
        raise UnsupportedTask(f'unknown DF storage {storage}')
    if not hasattr(fitted, 'use_gpu_memory'):
        raise UnsupportedTask(f'{_class(fitted)} cannot configure {storage} storage')
    fitted.use_gpu_memory = True if storage == 'resident' else None
    if storage == 'disk' and not hasattr(fitted, 'cderi_storage'):
        raise UnsupportedTask('imported GPU DF implementation has no disk storage')
    if hasattr(fitted, 'cderi_storage'):
        fitted.cderi_storage = 'disk' if storage == 'disk' else None
        fitted.cderi_buffer_bytes = options.get('buffer_bytes', 128 * 1024**2)
        fitted.cderi_cache_bytes = options.get('cache_bytes', 0)


def _close_df(fitted):
    """Explicitly release owned disk tensors, including range-separated fits."""
    for child in getattr(fitted, '_rsh_df', {}).values():
        _close_df(child)
    tensors = getattr(fitted, '_cderi', None)
    if isinstance(tensors, dict):
        tensors = tensors.values()
    elif not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for tensor in tensors:
        close = getattr(tensor, 'close', None)
        if close is not None:
            close()


def _frozen_density(case, expected_shape):
    """Load the exact common MINAO artifact, with file and state validation."""
    import numpy as np
    options = case.get('task_options', {})
    filename = options.get('initial_density_file')
    if filename is None:
        return None
    expected_hash = options.get('initial_density_file_sha256')
    if not expected_hash:
        raise ValueError('initial density file hash is required')
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError('initial density file hash mismatch')
    # Load the exact bytes verified above, avoiding a path replacement race.
    from io import BytesIO
    density = np.load(BytesIO(payload), allow_pickle=False)
    if density.dtype != np.dtype('float64'):
        raise ValueError('initial density artifact must have float64 dtype')
    if density.shape != expected_shape:
        raise ValueError(f'initial density shape {density.shape} != {expected_shape}')
    if not np.isfinite(density).all():
        raise ValueError('initial density artifact must be finite')
    return np.ascontiguousarray(density)


def _molecular_setup(case, backend):
    import numpy as np
    pyscf = _import('pyscf')
    geometry = Path(case['geometry'])
    if not geometry.is_absolute():
        geometry = Path(__file__).resolve().parents[3] / geometry
    mol = pyscf.gto.M(atom=str(geometry), basis=case['basis'], charge=case['charge'],
                      ecp=case.get('task_options', {}).get('ecp', {}),
                      spin=case['spin'], unit=case.get('settings', {}).get('unit', 'Angstrom'),
                      verbose=0)
    # Both backends start from this independent CPU atomic guess.
    unrestricted = bool(case['spin'])
    cpu_scf = _import('pyscf.scf')
    guess_start = perf_counter()
    shape = ((2,) if unrestricted else ()) + (mol.nao_nr(), mol.nao_nr())
    guess = _frozen_density(case, shape)
    if guess is None:
        guess = (cpu_scf.UHF if unrestricted else cpu_scf.RHF)(mol).get_init_guess(key='minao')
    guess = np.asarray(guess, dtype=np.float64, order='C')
    density_hash = hashlib.sha256(guess.tobytes()).hexdigest()
    guess_seconds = perf_counter() - guess_start
    package = 'pyscf' if backend == 'cpu' else 'gpu4pyscf'
    hf = case['xc'].upper() == 'HF'
    module = _import(package + ('.scf' if hf else '.dft'))
    constructor = getattr(module, ('U' if unrestricted else 'R') + ('HF' if hf else 'KS'))
    mf = constructor(mol) if hf else constructor(mol, xc=case['xc'])
    if case['df']:
        mf = mf.density_fit(auxbasis=case['auxbasis'])
    options = case.get('task_options', {})
    solvent = options.get('solvent', case.get('solvent'))
    if solvent:
        model = solvent['model'].upper()
        if model not in {'PCM', 'SMD'}:
            raise UnsupportedTask(f'unknown solvent model {model}')
        if backend == 'cpu':
            solvation = _import('pyscf.solvent')
            mf = getattr(solvation, model)(mf)
        else:
            wrapper = getattr(mf, model, None)
            if wrapper is None:
                raise UnsupportedTask(f'{_class(mf)} has no {model}')
            mf = wrapper()
        for key in ('method', 'eps', 'lebedev_order', 'solvent'):
            if key in solvent:
                setattr(mf.with_solvent, key, solvent[key])
    if options.get('dispersion'):
        # Dispersion parameters are looked up by functional name, but the registry pins the VWN5 B3LYP coefficients
        # (PySCF's own name 'b3lyp' is the VWN3/RPA variant, 0.037 Eh away). Keep the pinned xc and name only the
        # parameter set, using PySCF's explicit 'version:parameter-set' form on CPU and GPU alike.
        if case['xc'] not in _DISPERSION_PARAMETER_SET:
            raise UnsupportedTask('no declared dispersion parameter set for ' + case['xc'])
        mf.disp = options['dispersion'] + ':' + _DISPERSION_PARAMETER_SET[case['xc']]
    if options.get('nlc'):
        mf.nlc = options['nlc']
    _settings(mf, case, backend)
    if backend == 'gpu' and case['df']:
        _configure_gpu_df(mf, case)
    return mol, mf, guess, density_hash, guess_seconds


def _finite_difference_hessian(mf, case, step=1e-3, check_step=2e-3):
    """Hessian from central differences of the analytic gradient (auxiliary and grid response included).

    CPU PySCF's analytic density-fitted Hessian ignores grid response, so it is not an oracle for the model the GPU
    Hessian implements; the gradient is exact for that model. Returns (hessian[atom, atom, xyz, xyz] at `step`,
    diagnostics). The second step size checks stability of the finite difference.
    """
    import numpy as np
    mol = mf.mol
    natm = mol.natm
    x0 = mol.atom_coords()
    mf = mf.copy()               # displaced SCFs need tighter convergence than the case: gradient noise / step enters the Hessian
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    gradient = mf.nuc_grad_method()
    if case['df']:
        gradient.auxbasis_response = True
    if case['xc'].upper() != 'HF':
        gradient.grid_response = True
    scanner = gradient.as_scanner()

    def at(coords):
        displaced = mol.copy()
        displaced.set_geom_(coords, unit='Bohr')
        displaced.build()
        return np.asarray(scanner(displaced)[1])

    def hessian_for(h):
        hess = np.zeros((natm, natm, 3, 3))
        for i in range(natm):
            for k in range(3):
                plus, minus = x0.copy(), x0.copy()
                plus[i, k] += h
                minus[i, k] -= h
                hess[i, :, k, :] = (at(plus) - at(minus)) / (2 * h)
        return .5 * (hess + hess.transpose(1, 0, 3, 2))

    primary, check = hessian_for(step), hessian_for(check_step)
    diagnostics = {'method': 'central difference of analytic gradient, auxiliary and grid response',
                   'step_bohr': step, 'check_step_bohr': check_step,
                   'step_stability_max_abs': float(np.abs(primary - check).max()),
                   'displaced_scf_conv_tol': 1e-12, 'displaced_scf_conv_tol_grad': 1e-9,
                   'gradient_evaluations': 12 * natm}
    return primary, diagnostics


def _periodic_setup(case, backend):
    import numpy as np
    spec = case['cell']
    kmesh = spec.get('kmesh', [1, 1, 1])
    gto = _import('pyscf.pbc.gto')
    cell = gto.Cell(a=spec['a'], atom=spec['atom'], basis=spec['basis'],
                    pseudo=spec['pseudo'], charge=case['charge'], spin=case['spin'],
                    unit=spec.get('unit', 'Angstrom'), verbose=0,
                    precision=spec.get('precision', 1e-8))
    if 'mesh' in spec:
        cell.mesh = spec['mesh']
    cell.build()
    if case['xc'].upper() == 'HF':
        raise AdapterNotImplemented('periodic HF adapter is not implemented')
    cpu_dft = _import('pyscf.pbc.dft')
    sampled = kmesh != [1, 1, 1]
    constructor_name = ('K' if sampled else '') + ('UKS' if cell.spin else 'RKS')
    kwargs = {'kpts': cell.make_kpts(kmesh)} if sampled else {}
    cpu_mf = getattr(cpu_dft, constructor_name)(cell, xc=case['xc'], **kwargs)
    guess_start = perf_counter()
    shape = ((2,) if cell.spin else ()) + ((len(kwargs['kpts']),) if sampled else ()) + (cell.nao_nr(), cell.nao_nr())
    guess = _frozen_density(case, shape)
    if guess is None:
        guess = np.asarray(cpu_mf.get_init_guess(key='minao'), dtype=np.float64, order='C')
    guess_seconds = perf_counter() - guess_start
    module = _import(('pyscf' if backend == 'cpu' else 'gpu4pyscf') + '.pbc.dft')
    mf = getattr(module, constructor_name)(cell, xc=case['xc'], **kwargs)
    if case['df']:
        mf = mf.density_fit(auxbasis=case.get('auxbasis'))
    _settings(mf, case, backend)
    return cell, mf, guess, hashlib.sha256(guess.tobytes()).hexdigest(), guess_seconds


def execute(case, backend, mode='timing', tracker=None):
    """Execute one independent native task and return JSON serializable arrays.

    Setup includes molecule/SCF construction. Atomic MINAO guess construction and
    GPU upload belong to setup and have separately identified durations.
    Output conversion is inside each task boundary. CUDA synchronization occurs
    only at declared phase boundaries; callbacks record scalar iteration counts.
    """
    import numpy as np
    if case.get('settings', {}).get('dtype', 'float64') != 'float64':
        raise UnsupportedTask('suite adapters require float64')
    if backend not in {'cpu', 'gpu'}:
        raise ValueError(f'unknown backend {backend}')
    if mode not in {'capability', 'reference', 'timing', 'memory', 'profile'}:
        raise ValueError(f'unknown mode {mode}')
    if mode == 'timing' and tracker is not None:
        raise ValueError('timing mode prohibits allocation/profiling tracker')
    task = case['task']
    if task not in _IMPLEMENTED:
        raise AdapterNotImplemented(f'suite adapter not implemented for {task}; library capability unknown')
    if task in {'dfmp2', 'dfump2'} and case['xc'].upper() != 'HF':
        raise UnsupportedTask('MP2 requires a Hartree-Fock reference')
    if task in {'dfmp2', 'dfump2'}:
        opts = case.get('task_options', {})
        if opts.get('frozen', 0) != 0 or opts.get('with_t2', False):
            raise UnsupportedTask('MP2 suite requires frozen=0 and with_t2=False')
    if task == 'dfmp2' and case['spin']:
        raise UnsupportedTask('restricted MP2 requires spin zero; use dfump2')
    if task == 'ccsd' and (case['spin'] or case['df'] or case['xc'].upper() != 'HF'):
        raise UnsupportedTask('CCSD adapter requires direct closed-shell RHF')
    package = 'pyscf' if backend == 'cpu' else 'gpu4pyscf'
    # Native imports and dependency qualification precede principal timers.
    _import('pyscf')
    _import(package + ('.scf' if case['xc'].upper() == 'HF' else '.dft'))
    task_module = None
    if task in {'dfmp2', 'dfump2'}:
        task_module = _import(package + '.mp.' + task)
    elif task == 'ccsd':
        task_module = _import('pyscf.cc' if backend == 'cpu' else 'gpu4pyscf.cc.ccsd_incore')
    elif task in {'tda', 'tddft'}:
        _import(package + '.tdscf')
    elif task == 'hessian':
        _import(package + '.hessian')
        _import('pyscf.hessian.thermo')
    elif task in {'polarizability', 'shielding'}:
        if backend == 'cpu':
            _import('pyscf.prop')
        else:
            task_module = _import('gpu4pyscf.properties.' + task)
    elif task == 'optimization':
        _import('geometric')
        task_module = _import('pyscf.geomopt.geometric_solver')
    cp = _import('cupy') if backend == 'gpu' else None
    if cp is not None:
        # Identical tiny context/BLAS warmup; never prebuilds grids or integrals.
        cp.dot(cp.ones((2, 2)), cp.ones((2, 2)))
        cp.cuda.get_current_stream().synchronize()

    phases = {}
    def sync():
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()

    @contextmanager
    def phase(name):
        sync()
        start = perf_counter()
        with tracker.phase(name) if tracker is not None else nullcontext():
            try:
                yield
            finally:
                sync()
                phases[name] = perf_counter() - start

    # Construction, atomic guess preparation and upload belong to workflow setup.
    # Restore process-global build policy and close disk tensors on every exit.
    cleanup = ExitStack()
    budget = None
    budget_enforced = None
    if backend == 'gpu' and case['df'] and task != 'pbc_energy':
        config = _import('gpu4pyscf.__config__')
        key = 'df_cderi_build_workspace_bytes'
        absent = object()
        previous = getattr(config, key, absent)
        budget = case.get('settings', {}).get('build_budget_bytes', 4 * 1024**3)
        import inspect
        source = inspect.getsource(_import('gpu4pyscf.df.df')._cholesky_eri)
        budget_enforced = key in source
        setattr(config, key, budget)
        def restore_budget():
            if previous is absent:
                delattr(config, key)
            else:
                setattr(config, key, previous)
        cleanup.callback(restore_budget)
    with cleanup:
        setup_start = perf_counter()
        with phase('setup'):
            setup = _periodic_setup if task == 'pbc_energy' else _molecular_setup
            mol, mf, guess, density_hash, guess_seconds = setup(case, backend)
        phases['initial_density_build'] = guess_seconds
        if case['df']:
            cleanup.callback(_close_df, mf.with_df)
        with phase('initial_density_upload'):
            dm0 = cp.asarray(guess) if cp is not None else guess
        total_start = setup_start
        result = {'case': case, 'initial_density_sha256': density_hash,
                  'metadata': {'backend': backend, 'mode': mode,
                               'classes': {'molecule': _class(mol), 'scf': _class(mf)},
                               'placement': {'setup': 'cpu', 'scf': backend, task: backend},
                               'state': {'charge': case['charge'], 'spin': case['spin'],
                                         'nelectron': mol.nelectron, 'nao': mol.nao_nr()},
                               'guess': ('frozen_common_cpu_atomic_minao' if case.get('task_options', {}).get('initial_density_file')
                                         else 'independent_cpu_atomic_minao'),
                               'timing_scope': 'setup_including_minao+upload+scf+task+result_conversion+cleanup; imports/context excluded',
                               'effective_settings': {key: getattr(mf, key) for key in
                                                      ('conv_tol', 'conv_tol_grad', 'max_cycle')}}}
        if backend == 'gpu' and case['df'] and task != 'pbc_energy':
            fitted = mf.with_df
            result['metadata']['storage'] = {
                'requested': case.get('task_options', {}).get('storage', case.get('settings', {}).get('storage', 'resident')),
                'use_gpu_memory': fitted.use_gpu_memory,
                'cderi_storage': getattr(fitted, 'cderi_storage', None),
                'cache_bytes': getattr(fitted, 'cderi_cache_bytes', 0),
                'build_budget_bytes': budget, 'build_budget_enforced': budget_enforced}
        effective = result['metadata']['effective_settings']
        effective.update({'basis': case['basis'], 'auxbasis': case.get('auxbasis'),
                          'xc': case['xc'], 'df': case['df'], 'dtype': 'float64',
                          'conv_tol_cpscf': mf.conv_tol_cpscf})
        if case['xc'].upper() != 'HF':
            effective.update({'grid_level': mf.grids.level, 'prune': 'nwchem',
                              'small_rho_cutoff': mf.small_rho_cutoff,
                              'nlc': getattr(mf, 'nlc', '')})
        iterations = []
        def callback(env):
            iterations.append(int(env['cycle']) + 1)
        mf.callback = callback
        with phase('scf'):
            result['energy_hartree'] = float(mf.kernel(dm0=dm0))
        result['converged'] = bool(mf.converged)
        result['iterations'] = max(iterations, default=int(getattr(mf, 'cycles', 0)))
        result['task_converged'] = False
        if not result['converged']:
            with phase('cleanup'):
                cleanup.close()
            phases['total'] = perf_counter() - total_start
            result['phases_seconds'] = phases
            result['workflow_seconds'] = phases['total']
            return result
        options = case.get('task_options', {})
        if task == 'pbc_energy':
            result['metadata']['periodic'] = {
                'kmesh': case['cell'].get('kmesh', [1, 1, 1]),
                'mesh': list(mol.mesh), 'precision': mol.precision,
                'fitting_class': _class(mf.with_df)}
        elif task == 'energy_gradient':
            with phase('gradient'):
                gradient = mf.nuc_grad_method()
                if case['df']:
                    gradient.auxbasis_response = True
                if case['xc'].upper() != 'HF':
                    gradient.grid_response = True
                result['gradient_hartree_per_bohr'] = _array(gradient.kernel())
            result['metadata']['classes']['task'] = _class(gradient)
            result['metadata']['response'] = {'auxbasis_response': bool(case['df']), 'grid_response': case['xc'].upper() != 'HF'}
        elif task == 'hessian':
            with phase('hessian'):
                if backend == 'cpu':
                    raw_hessian, fd_diagnostics = _finite_difference_hessian(mf, case)
                    hessian = mf.nuc_grad_method()
                    result['metadata']['hessian_reference'] = fd_diagnostics
                else:
                    hessian = mf.Hessian()
                    if case['df']:
                        hessian.auxbasis_response = 2
                    if case['xc'].upper() != 'HF':
                        hessian.grid_response = True
                    raw_hessian = _host(hessian.kernel())
                result['hessian_hartree_per_bohr2'] = raw_hessian.transpose(0, 2, 1, 3).tolist()
            with phase('harmonic_analysis'):
                thermo = _import('pyscf.hessian.thermo')
                modes = thermo.harmonic_analysis(mol, raw_hessian)
                frequencies = np.asarray(modes['freq_wavenumber'])
                signed = np.where(np.abs(frequencies.imag) > 1e-8, -np.abs(frequencies.imag), frequencies.real)
                result['frequencies_cm1'] = signed.tolist()
            result['metadata']['placement']['harmonic_analysis'] = 'cpu'
            result['metadata']['frequency_convention'] = 'imaginary_modes_negative_cm-1'
            result['metadata']['classes']['task'] = _class(hessian)
            result['metadata']['response'] = {'auxbasis_response': 2 if case['df'] else 0,
                                              'grid_response': case['xc'].upper() != 'HF',
                                              'cphf_grid': 'same_as_scf_grid'}
            result['metadata']['hessian_order'] = 'atom,xyz,atom,xyz'
        elif task in {'tda', 'tddft'}:
            with phase('response'):
                td = getattr(mf, 'TDA' if task == 'tda' else 'TDDFT')()
                td.nstates = options.get('nstates', 5)
                td.conv_tol = options.get('conv_tol', 1e-9)
                td.max_cycle = options.get('max_cycle', 100)
                energies, amplitudes = td.kernel()
                result['excitation_energies_hartree'] = _array(energies)
                result['oscillator_strengths'] = _array(td.oscillator_strength())
                result['task_convergence'] = np.asarray(_host(td.converged), dtype=bool).reshape(-1).tolist()
                # AO transition densities are invariant to independent MO phases.
                # Spin channels remain separate for unrestricted references.
                coefficients = _host(mf.mo_coeff)
                occupations = _host(mf.mo_occ)
                vectors = []
                for x, y in amplitudes:
                    if coefficients.ndim == 2:
                        x = _host(x)
                        y = np.zeros_like(x) if np.isscalar(y) else _host(y)
                        occupied = coefficients[:, occupations > 0]
                        virtual = coefficients[:, occupations == 0]
                        vectors.append((occupied @ (x + y) @ virtual.T).ravel().tolist())
                    else:
                        spin_vectors = []
                        for spin_index in range(2):
                            xs = _host(x[spin_index])
                            ys = np.zeros_like(xs) if np.isscalar(y) else _host(y[spin_index])
                            coeff = coefficients[spin_index]
                            occ = occupations[spin_index]
                            spin_vectors.append((coeff[:, occ > 0] @ (xs + ys) @ coeff[:, occ == 0].T).ravel())
                        vectors.append(np.concatenate(spin_vectors).tolist())
                result['transition_amplitudes'] = vectors
                result['metadata']['transition_representation'] = 'ao_transition_density'
            result['metadata']['classes']['task'] = _class(td)
            result['metadata']['response'] = {'nstates': td.nstates, 'conv_tol': td.conv_tol}
        elif task == 'ccsd':
            with phase('correlation'):
                cc = task_module.CCSD(mf, frozen=options.get('frozen', 0))
                cc.conv_tol = options.get('conv_tol', 1e-9)
                cc.max_cycle = options.get('max_cycle', 100)
                if hasattr(cc, 'conv_tol_normt'):
                    cc.conv_tol_normt = options.get('conv_tol_normt', 1e-7)
                cc.kernel()
                result['correlation_energy_hartree'] = float(cc.e_corr)
                result['task_convergence'] = bool(cc.converged)
            result['metadata']['classes']['task'] = _class(cc)
            if backend == 'gpu':
                result['metadata']['placement']['ccsd'] = 'cpu+gpu'
                result['metadata']['placement']['ccsd_reference_conversion'] = 'gpu_to_cpu'
            result['metadata']['correlation'] = {'frozen': cc.frozen, 'density_fitting': False}
        elif task in {'polarizability', 'shielding'}:
            with phase('response'):
                if backend == 'gpu':
                    if task == 'polarizability':
                        value = task_module.eval_polarizability(
                            mf, tol=options.get('conv_tol_cpscf', 1e-10),
                            max_cycle=options.get('max_cycle', 100))
                    else:
                        diamagnetic, paramagnetic = task_module.eval_shielding(mf)
                        value = diamagnetic + paramagnetic
                    result['metadata']['classes']['task'] = task_module.__name__
                else:
                    if task == 'polarizability':
                        prop = mf.Polarizability()
                        value = prop.polarizability()
                    else:
                        prop = mf.NMR()
                        value = prop.kernel()
                    result['metadata']['classes']['task'] = _class(prop)
                result['polarizability_au' if task == 'polarizability' else 'shielding_ppm'] = _array(value)
        elif task == 'optimization':
            with phase('optimization'):
                gradient = mf.nuc_grad_method()
                if case['df']:
                    gradient.auxbasis_response = True
                if case['xc'].upper() != 'HF':
                    gradient.grid_response = True
                # Driver itself runs on CPU; gradient electronic work keeps backend.
                optimizer_options = {key: options[key] for key in options if key.startswith('convergence_')}
                driver_input = gradient.as_scanner() if backend == 'gpu' else gradient  # GPU scanner subclasses pyscf.lib.GradScanner
                success, optimized = task_module.kernel(driver_input, maxsteps=options.get('maxsteps', 100),
                                                         **optimizer_options)
                mf.reset(optimized)
                result['energy_hartree'] = float(mf.kernel())
                final_gradient = mf.nuc_grad_method()
                if case['df']:
                    final_gradient.auxbasis_response = True
                if case['xc'].upper() != 'HF':
                    final_gradient.grid_response = True
                result['gradient_hartree_per_bohr'] = _array(final_gradient.kernel())
                result['converged'] = bool(mf.converged)
                result['geometry_bohr'] = optimized.atom_coords().tolist()
                result['task_convergence'] = bool(success)
            result['metadata']['classes']['task'] = task_module.__name__
            result['metadata']['placement']['optimizer_driver'] = 'cpu'
        elif task in {'dfmp2', 'dfump2'}:
            with phase('correlation'):
                module = task_module
                cls = getattr(module, 'DFMP2' if task == 'dfmp2' else 'DFUMP2')
                correlation_aux = options.get('auxbasis', 'def2-svp-ri')
                if backend == 'gpu':
                    mp = cls(mf, frozen=0, auxbasis=correlation_aux)
                    mp.fp_type = 'FP64'
                else:
                    mp = cls(mf, frozen=0)
                    # Do not reuse the SCF JK-fitting object or its cached integrals.
                    from pyscf import df
                    mp.with_df = df.DF(mol, auxbasis=correlation_aux)
                cleanup.callback(_close_df, mp.with_df)
                mp.with_t2 = False
                correlation, _ = mp.kernel(with_t2=False)
                result['correlation_energy_hartree'] = float(correlation)
            result['metadata']['classes']['task'] = _class(mp)
            result['metadata']['correlation'] = {'auxbasis': correlation_aux, 'frozen': 0,
                                               'with_t2': False, 'precision': 'float64'}
        if task in {'dfmp2', 'dfump2', 'ccsd'}:
            result['hf_reference_energy_hartree'] = result['energy_hartree']
            result['energy_hartree'] += result['correlation_energy_hartree']
            result['metadata']['energy_definition'] = 'hartree_fock_reference_plus_correlation'
        convergence = result.get('task_convergence', [True])
        result['task_converged'] = bool(convergence) if isinstance(convergence, bool) else all(convergence)
        if task in {'tda', 'tddft'}:
            result['task_converged'] = result['task_converged'] and len(result['excitation_energies_hartree']) == options.get('nstates', 5)
        # Save enough SCF state to compare electronic states and interpret roots.
        result['mo_energy_hartree'] = _array(mf.mo_energy)
        result['mo_occupations'] = _array(mf.mo_occ)
        occ = _host(mf.mo_occ)
        result['metadata']['state']['nocc'] = np.count_nonzero(occ > 0, axis=-1).tolist()
        result['metadata']['state']['nvir'] = np.count_nonzero(occ == 0, axis=-1).tolist()
        auxmol = getattr(getattr(mf, 'with_df', None), 'auxmol', None)
        if auxmol is not None:
            result['metadata']['state']['naux'] = auxmol.nao_nr()
        result['mo_coefficients'] = _array(mf.mo_coeff)
        result.setdefault('geometry_bohr', mol.atom_coords().tolist())
        with phase('cleanup'):
            cleanup.close()
        phases['total'] = perf_counter() - total_start
        result['phases_seconds'] = phases
        result['workflow_seconds'] = phases['total']
        return result
