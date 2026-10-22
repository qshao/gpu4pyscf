"""Independent, CPU-only acceptance gates for saved scientific observables.

No saved ``passed`` flag participates in acceptance. TD overlap matching uses
normalized real transition vectors; absent vectors give a limited diagnostic.
Degeneracies are defined at 1e-7 Eh and compared as subspaces and summed strengths.
"""
import numpy as np


ENERGY = 'energy_hartree'
GRADIENT = 'gradient_hartree_per_bohr'
HESSIAN = 'hessian_hartree_per_bohr2'
EXCITATION = 'excitation_energies_hartree'
OSCILLATOR = 'oscillator_strengths'
GATES = {
    'energy_gradient': {ENERGY: 1e-8, GRADIENT: 1e-6},
    'hessian': {ENERGY: 1e-8, HESSIAN: 1e-5},
    'tda': {ENERGY: 1e-8, EXCITATION: 1e-6, OSCILLATOR: 5e-5},
    'tddft': {ENERGY: 1e-8, EXCITATION: 1e-6, OSCILLATOR: 5e-5},
    **{task: {ENERGY: 1e-8, 'correlation_energy_hartree': 1e-8}
       for task in ('dfmp2', 'dfump2', 'ccsd')},
    'polarizability': {ENERGY: 1e-8, 'polarizability_au': 1e-5},
    'shielding': {ENERGY: 1e-8, 'shielding_ppm': 0.1},
    'pbc_energy': {ENERGY: 1e-8},
    'optimization': {},
}


def _array(result, key):
    value = np.asarray(result[key], dtype=float)
    if not value.size or not np.isfinite(value).all():
        raise ValueError('empty or nonfinite array')
    if key in (ENERGY, 'correlation_energy_hartree') and value.shape != ():
        raise ValueError('expected scalar')
    if key in (GRADIENT, 'geometry_bohr') and (value.ndim != 2 or value.shape[1] != 3):
        raise ValueError('expected (atoms, 3)')
    if key == HESSIAN and not (value.ndim == 4 and value.shape[0] == value.shape[2]
                              and value.shape[1] == value.shape[3] == 3):
        raise ValueError('expected (atoms, 3, atoms, 3)')
    if key in (EXCITATION, OSCILLATOR) and value.ndim != 1:
        raise ValueError('expected root vector')
    if key == 'polarizability_au' and value.shape != (3, 3):
        raise ValueError('expected (3, 3)')
    if key == 'shielding_ppm' and (value.ndim != 3 or value.shape[1:] != (3, 3)):
        raise ValueError('expected (atoms, 3, 3)')
    return value


def _assignment(overlap):
    """Maximum-weight one-to-one assignment, exact for the small root sets used."""
    n = len(overlap)
    if n > 16:
        raise ValueError('more than 16 roots: state matching not qualified')
    states = {0: (0., [])}
    for row in range(n):
        next_states = {}
        for mask, (score, mapping) in states.items():
            for col in range(n):
                if mask & (1 << col):
                    continue
                new_mask = mask | (1 << col)
                proposal = (score + overlap[row, col], mapping + [col])
                if new_mask not in next_states or proposal[0] > next_states[new_mask][0]:
                    next_states[new_mask] = proposal
        states = next_states
    return states[(1 << n) - 1][1]


def _td_match(reference, candidate, arrays, diagnostics):
    ref_e, cand_e = arrays[EXCITATION]
    ref_f, cand_f = arrays[OSCILLATOR]
    n = len(ref_e)
    if len(ref_f) != n or len(cand_f) != n:
        raise ValueError('oscillator/root count mismatch')
    if 'transition_amplitudes' not in reference or 'transition_amplitudes' not in candidate:
        diagnostics['state_matching'] = 'energy_only; transition character unavailable'
        return 'limited'
    a = _array(reference, 'transition_amplitudes')
    b = _array(candidate, 'transition_amplitudes')
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != n:
        raise ValueError('transition amplitude shape mismatch')
    norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if np.any(norm_a == 0) or np.any(norm_b == 0):
        raise ValueError('zero transition amplitude norm')
    a, b = a / norm_a[:, None], b / norm_b[:, None]
    mapping = _assignment(np.abs(a @ b.T) ** 2)
    diagnostics['root_mapping'] = mapping
    diagnostics['state_matching'] = 'normalized transition-vector squared overlap'
    arrays[EXCITATION] = (ref_e, cand_e[mapping])
    # Form complete-link energy groups, avoiding transitive degeneracy chains.
    groups = []
    for i in np.argsort(ref_e):
        if groups and abs(ref_e[i] - ref_e[groups[-1][0]]) <= 1e-7:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    strengths_a, strengths_b, overlaps = [], [], []
    for group in groups:
        selected = [mapping[i] for i in group]
        # SVD bases also reject rank-deficient repeated vectors.
        ua, sa, _ = np.linalg.svd(a[group].T, full_matrices=False)
        ub, sb, _ = np.linalg.svd(b[selected].T, full_matrices=False)
        if min(sa.min(), sb.min()) < 1e-8:
            raise ValueError('rank-deficient transition subspace')
        singular = np.linalg.svd(ua.T @ ub, compute_uv=False)
        overlaps.append(float(np.min(singular) ** 2))
        strengths_a.append(float(ref_f[group].sum()))
        strengths_b.append(float(cand_f[selected].sum()))
    diagnostics['state_subspace_min_squared_overlap'] = min(overlaps)
    diagnostics['degenerate_groups'] = groups
    arrays[OSCILLATOR] = (np.asarray(strengths_a), np.asarray(strengths_b))
    return 'qualified' if min(overlaps) >= 0.99 else 'state_mismatch'


def compare_results(case, reference, candidate):
    """Recompute scientific acceptance; return JSON-compatible errors/diagnostics.

    Results must retain the frozen case and explicit SCF/task convergence. Array
    names encode units and Hessians use (atom, xyz, atom, xyz) ordering. Optimizer
    endpoints each satisfy declared force gates; paths need not coincide.
    """
    errors, diagnostics = {}, {}
    status = 'qualified'
    task = case.get('task')
    if task not in GATES:
        return {'passed': False, 'errors': {}, 'diagnostics': {'reason': 'unsupported task'},
                'qualification_status': 'unsupported'}
    for label, result in (('reference', reference), ('candidate', candidate)):
        if result.get('case') != case:
            diagnostics[label + '_case'] = 'missing or changed scientific case'
            status = 'state_mismatch'
        for key in ('converged', 'task_converged'):
            if result.get(key) is not True:
                diagnostics[label + '_' + key] = 'missing or false convergence'
                status = 'nonconverged'
    for key in ('initial_density_sha256', 'scientific_signature'):
        if reference.get(key) != candidate.get(key):
            diagnostics[key] = 'reference/candidate identity mismatch'
            status = 'state_mismatch'
    for key in ('state', 'effective_settings', 'response', 'correlation', 'units',
                'hessian_order', 'transition_representation'):
        if reference.get('metadata', {}).get(key) != candidate.get('metadata', {}).get(key):
            diagnostics['metadata_' + key] = 'effective scientific settings mismatch'
            status = 'state_mismatch'
    arrays = {}
    for key, tolerance in GATES[task].items():
        try:
            a, b = _array(reference, key), _array(candidate, key)
            if a.shape != b.shape:
                raise ValueError('shape mismatch')
            arrays[key] = (a, b)
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            errors[key] = {'passed': False, 'reason': str(exc), 'tolerance': tolerance}
    if task in ('tda', 'tddft') and EXCITATION in arrays and OSCILLATOR in arrays:
        try:
            expected_roots = case.get('task_options', {}).get('nstates')
            if expected_roots is not None and len(arrays[EXCITATION][0]) != expected_roots:
                raise ValueError('missing requested excitation roots')
            td_status = _td_match(reference, candidate, arrays, diagnostics)
            if status == 'qualified':
                status = td_status
        except (KeyError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            diagnostics['state_matching'] = str(exc)
            status = 'state_mismatch'
    for key, (a, b) in arrays.items():
        difference = float(np.max(np.abs(a - b)))
        errors[key] = {'max_abs': difference, 'tolerance': GATES[task][key],
                       'passed': difference <= GATES[task][key]}
    if task == 'optimization':
        try:
            for label, result in (('reference', reference), ('candidate', candidate)):
                g = _array(result, GRADIENT)
                maximum, rms = float(np.max(np.abs(g))), float(np.sqrt(np.mean(g ** 2)))
                errors[label + '_final_force'] = {'max_abs': maximum, 'rms': rms,
                    'passed': maximum <= 3e-4 and rms <= 1e-4}
                _array(result, ENERGY)
            a, b = _array(reference, 'geometry_bohr'), _array(candidate, 'geometry_bohr')
            if a.shape != b.shape:
                raise ValueError('endpoint geometry shape mismatch')
            # Align by proper rigid rotation, retaining atom order.
            a, b = a - a.mean(axis=0), b - b.mean(axis=0)
            u, _, vt = np.linalg.svd(b.T @ a)
            rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
            diagnostics['aligned_endpoint_rms_bohr'] = float(np.sqrt(np.mean((b @ rotation - a) ** 2)))
            diagnostics['endpoint_energy_difference_hartree'] = float(candidate[ENERGY] - reference[ENERGY])
        except (KeyError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            errors['optimization'] = {'passed': False, 'reason': str(exc)}
    if any(not value['passed'] for value in errors.values()) and status == 'qualified':
        status = 'accuracy_failed'
    return {'passed': status == 'qualified', 'errors': errors, 'diagnostics': diagnostics,
            'qualification_status': status}
