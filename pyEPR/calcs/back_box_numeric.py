'''
Numerical diagonalization of quantum Hamiltonian and parameter
extraction.

@author: Phil Reinhold, Zlatko Minev, Lysander Christakis

Original code on black_box_hamiltonian and make_dispersive functions by Phil Reinhold
Revisions and updates by Zlatko Minev & Lysander Christakis
'''
# pylint: disable=invalid-name


from __future__ import print_function

from functools import reduce

import numpy as np
import scipy as sp

from .constants import Planck as h
from .constants import fluxQ, hbar
from .hamiltonian import MatrixOps

try:
    import qutip
    from qutip import basis, tensor
except (ImportError, ModuleNotFoundError):
    pass

__all__ = [ 'epr_numerical_diagonalization',
            'make_dispersive',
            'black_box_hamiltonian_HO_basis',
            'black_box_hamiltonian',
            'black_box_hamiltonian_nq']

dot = MatrixOps.dot
cos_approx = MatrixOps.cos_approx


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def epr_numerical_diagonalization(freqs, Ljs, ϕzpf,
             cos_trunc=8,
             fock_trunc=9,
             flux=0,
             basis='std',
             use_1st_order=False,
             return_H=False,
             non_linear_potential=None):
    '''
    Numerical diagonalization for pyEPR. Ask Zlatko for details.

    :param fs: (GHz, not radians) Linearized model, H_lin, normal mode frequencies in Hz, length M
    :param ljs: (Henries) junction linearized inductances in Henries, length J
    :param fzpfs: (reduced) Reduced Zero-point fluctuation of the junction fluxes for each mode
                across each junction, shape MxJ

    :return: Hamiltonian mode freq and dispersive shifts. Shifts are in MHz.
             Shifts have flipped sign so that down shift is positive.
    '''

    freqs, Ljs, ϕzpf = map(np.array, (freqs, Ljs, ϕzpf))
    
    assert(all(freqs < 1E6)
           ), "Please input the frequencies in GHz. \N{nauseated face}"
    assert(all(Ljs < 1E-3)
           ), "Please input the inductances in Henries. \N{nauseated face}"
    
    if basis == 'HO':
        print('Using Full Cosine Potential')
        Hs = black_box_hamiltonian_HO_basis(freqs * 1E9, Ljs.astype(float), fluxQ*ϕzpf,
                 fock_trunc, flux, individual=use_1st_order)
    else:
        print('Using taylor expansion')
        Hs = black_box_hamiltonian(freqs * 1E9, Ljs.astype(float), fluxQ*ϕzpf,
                 cos_trunc, fock_trunc, individual=use_1st_order)
    
    f_ND, χ_ND, _, _ = make_dispersive(
        Hs, fock_trunc, ϕzpf, freqs, use_1st_order=use_1st_order)
    χ_ND = -1*χ_ND * 1E-6  # convert to MHz, and flip sign so that down shift is positive

    return (f_ND, χ_ND, Hs) if return_H else (f_ND, χ_ND)

def black_box_hamiltonian_HO_basis(fs, ljs, fzpfs, fock_trunc=20, flux=0, individual=False, non_linear_potential = None):
    r"""
    :param fs: Linearized model, H_lin, normal mode frequencies in Hz, length N
    :param ljs: junction linearized inductances in Henries, length M
    :param fzpfs: Zero-point fluctutation of the junction fluxes for each mode across each junction,
                 shape MxJ
    :param flux: Dimensionless external flux bias in [0,1] that modifies the interior of the cos term of the Hamiltonian.

    :return: Hamiltonian in units of Hz (i.e H / h)
    All in SI units. The ZPF fed in are the generalized, not reduced, flux.

    Description:
     Takes the linear mode frequencies, $\omega_m$, and the zero-point fluctuations, ZPFs, and
     builds the Hamiltonian matrix of $H_full$, assuming cos potential. It also takes the external
     flux bias into account.

     This method is made to analyse a single qubit, coupled to an arbitrary amount of harmonic oscillator-like modes.  
    """ 
    n_modes = len(fs)
    njuncs = len(ljs)
    fs, ljs, fzpfs = map(np.array, (fs, ljs, fzpfs))
    ejs = fluxQ**2 / ljs
    fjs = ejs / h
    fzpfs = np.transpose(fzpfs)  # Take from MxJ  to JxM

    assert np.isnan(fzpfs).any(
    ) == False, "Phi ZPF has NAN, this is NOT allowed! Fix me. \n%s" % fzpfs
    assert np.isnan(ljs).any(
    ) == False, "Ljs has NAN, this is NOT allowed! Fix me."
    assert np.isnan(
        fs).any() == False, "freqs has NAN, this is NOT allowed! Fix me."
    assert fzpfs.shape == (njuncs, n_modes), "incorrect shape for zpf array, {} not {}".format(
        fzpfs.shape, (njuncs, n_modes))
    assert fs.shape == (n_modes,), "incorrect number of mode frequencies"
    assert ejs.shape == (njuncs,), "incorrect number of qubit frequencies"

    def tensor_out(op, loc):
        "Make operator <op> tensored with identities at locations other than <loc>"
        op_list = [qutip.qeye(fock_trunc) for i in range(n_modes)]
        op_list[loc] = op
        return reduce(qutip.tensor, op_list)

    def dot(ais, bis):
        """
        Dot product
        """
        return sum(ai*bi for ai, bi in zip(ais, bis))

    a = qutip.destroy(fock_trunc)
    ad = a.dag()
    n = qutip.num(fock_trunc)
    mode_fields = [tensor_out(a + ad, i) for i in range(n_modes)]
    mode_ns = [tensor_out(n, i) for i in range(n_modes)]
    linear_part = dot(fs, mode_ns)

    nonlinear_part = 0

    for j in range(njuncs):
        phi_operator = 0
        lin = 0
        for i in range(n_modes):
            print('ϕzpf mode',i,': ', fzpfs[j,i] / fluxQ)
            phi_operator += 1j*fzpfs[j,i] / fluxQ * mode_fields[i]
            lin += (fzpfs[j,i] / fluxQ) * mode_fields[i]
        lin = 0.5*(lin**2)
        exp_matrix_no_flux = phi_operator.expm()
        exp_matrix = exp_matrix_no_flux * np.exp(1j * 2 * np.pi * flux)
        cos_matrix = 0.5 * (exp_matrix + exp_matrix.dag())
         
        im_non_zero = True in np.imag(cos_matrix+lin) != 0            
        if im_non_zero:      
            print('Imaginary part of non linear part is not zero')
        nonlinear_part += -fjs[j] * (cos_matrix+lin)    
    
    return linear_part + nonlinear_part

def black_box_hamiltonian(fs, ljs, fzpfs, cos_trunc=5, fock_trunc=8,
                          individual=False, non_linear_potential = None):
    r"""
    :param fs: Linearized model, H_lin, normal mode frequencies in Hz, length N
    :param ljs: junction linearized inductances in Henries, length M
    :param fzpfs: Zero-point fluctuation of the junction fluxes for each mode across each junction,
                 shape MxJ
    :return: Hamiltonian in units of Hz (i.e H / h)
    All in SI units. The ZPF fed in are the generalized, not reduced, flux.

    Description:
     Takes the linear mode frequencies, :math:`\omega_m`, and the zero-point fluctuations, ZPFs, and
     builds the Hamiltonian matrix of :math:`H_{full}`, assuming cos potential.
    """
    n_modes = len(fs)
    njuncs = len(ljs)
    fs, ljs, fzpfs = map(np.array, (fs, ljs, fzpfs))
    ejs = fluxQ**2 / ljs
    fjs = ejs / h

    fzpfs = np.transpose(fzpfs)  # Take from MxJ  to JxM

    assert np.isnan(fzpfs).any(
    ) == False, "Phi ZPF has NAN, this is NOT allowed! Fix me. \n%s" % fzpfs
    assert np.isnan(ljs).any(
    ) == False, "Ljs has NAN, this is NOT allowed! Fix me."
    assert np.isnan(
        fs).any() == False, "freqs has NAN, this is NOT allowed! Fix me."
    assert fzpfs.shape == (njuncs, n_modes), "incorrect shape for zpf array, {} not {}".format(
        fzpfs.shape, (njuncs, n_modes))
    assert fs.shape == (n_modes,), "incorrect number of mode frequencies"
    assert ejs.shape == (njuncs,), "incorrect number of qubit frequencies"

    def tensor_out(op, loc):
        "Make operator <op> tensored with identities at locations other than <loc>"
        op_list = [qutip.qeye(fock_trunc) for i in range(n_modes)]
        op_list[loc] = op
        return reduce(qutip.tensor, op_list)

    a = qutip.destroy(fock_trunc)
    ad = a.dag()
    n = qutip.num(fock_trunc)
    mode_fields = [tensor_out(a + ad, i) for i in range(n_modes)]
    mode_ns = [tensor_out(n, i) for i in range(n_modes)]

    def cos(x):
        return cos_approx(x, cos_trunc=cos_trunc)
    
    if non_linear_potential is None:
        non_linear_potential = cos

    linear_part = dot(fs, mode_ns)
    cos_interiors = [dot(fzpf_row/fluxQ, mode_fields) for fzpf_row in fzpfs]
    nonlinear_part = dot(-fjs, map(non_linear_potential, cos_interiors))
    if individual:
        return linear_part, nonlinear_part
    else:
        return linear_part + nonlinear_part

bbq_hmt = black_box_hamiltonian

def make_dispersive(H, fock_trunc, fzpfs=None, f0s=None, chi_prime=False,
                    use_1st_order=False):
    r"""
    Input: Hamiltonian Matrix.
        Optional: phi_zpfs and normal mode frequencies, f0s.
        use_1st_order : deprecated
    Output:
        Return dressed mode frequencies, chis, chi prime, phi_zpf flux (not reduced), and linear frequencies
    Description:
        Takes the Hamiltonian matrix `H` from bbq_hmt. It them finds the eigenvalues/eigenvectors and  
        assigns quantum numbers to them --- i.e., mode excitations,  such as, for instance, for three mode, 
        :math:`|0,0,0\rangle` or :math:`|0,0,1\rangle`, which correspond to no excitations in any of the modes 
        or one excitation in the 3rd mode, resp.    The assignment is performed based on the maximum overlap between 
        the eigenvectors of H_full and H_lin.   If this crude explanation is confusing, let me know, I will write a 
        more detailed one |:slightly_smiling_face:|
        Based on the assignment of the excitations, the function returns the dressed mode frequencies :math:`\omega_m^\prime`, 
        and the cross-Kerr matrix (including anharmonicities) extracted from the numerical diagonalization, as well as 
        from 1st order perturbation theory.
        Note, the diagonal of the CHI matrix is directly the anharmonicity term.

        This is a modified method that works for a single qubit, with arbirtrary harmonic oscillator-like modes
    """
    if hasattr(H, '__len__'):  # is it an array / list?
        [H_lin, H_nl] = H
        H = H_lin + H_nl
    else:  # make sure its a quantum object
        assert type(
            H) == qutip.qobj.Qobj, "Please pass in either a list of Qobjs or Qobj for the Hamiltonian"

    print("Starting the diagonalization")
    evals, evecs = H.eigenstates() # Diagonalize Hamiltonian matrix
    print("Finished the diagonalization")
    evals -= evals[0] # subtract ground state energy

    N = int(np.round(np.log(H.shape[0])/np.log(fock_trunc)))  # number of modes
    assert H.shape[0] == fock_trunc ** N
    
    # Find number of linear modes and distinguish trivial case N=1
    if N == 1: 
        f1s = [evals[1] - evals[0]]
        chis = [(evals[2] - evals[1]) - (evals[1] - evals[0])]

    else:
        def tensor_out(op, loc):
            "Make operator <op> tensored with identities at locations other than <loc>"
            op_list = [qutip.qeye(fock_trunc) for i in range(N)]
            op_list[loc] = op
            return reduce(qutip.tensor, op_list)

        def fock_state_on(d):
            ''' d={mode number: # of photons} '''
            return qutip.tensor(*[qutip.basis(fock_trunc, d.get(i, 0)) for i in range(N)])  # give me the value d[i] or 0 if d[i] does not exist

        if use_1st_order:
            num_modes = N
            print("Using 1st O")

            def multi_index_2_vector(d, num_modes, fock_trunc):
                return tensor([basis(fock_trunc, d.get(i, 0)) for i in range(num_modes)])
                '''this function creates a vector representation a given fock state given the data for excitations per
                            mode of the form d={mode number: # of photons}'''

            def find_multi_indices(fock_trunc):
                multi_indices = [{ind: item for ind, item in enumerate([i, j, k])} for i in range(fock_trunc)
                                for j in range(fock_trunc)
                                for k in range(fock_trunc)]
                return multi_indices
                '''this function generates all possible multi-indices for three modes for a given fock_trunc'''

            def get_expect_number(left, middle, right):
                return (left.dag()*middle*right).data.toarray()[0, 0]
                '''this function calculates the expectation value of an operator called "middle" '''

            def get_basis0(fock_trunc, num_modes):
                multi_indices = find_multi_indices(fock_trunc)
                basis0 = [multi_index_2_vector(
                    multi_indices[i], num_modes, fock_trunc) for i in range(len(multi_indices))]
                evalues0 = [get_expect_number(v0, H_lin, v0).real for v0 in basis0]
                return multi_indices, basis0, evalues0
                '''this function creates a basis of fock states and their corresponding eigenvalues'''

            def closest_state_to(vector0):

                def PT_on_vector(original_vector, original_basis, pertub, energy0, evalue):
                    new_vector = 0 * original_vector
                    for i in range(len(original_basis)):
                        if (energy0[i]-evalue) > 1e-3:
                            new_vector += ((original_basis[i].dag()*H_nl*original_vector).data.toarray()[
                                        0, 0])*original_basis[i]/(evalue-energy0[i])
                        else:
                            pass
                    return (new_vector + original_vector)/(new_vector + original_vector).norm()
                    '''this function calculates the normalized vector with the first order correction term
                    from the non-linear hamiltonian '''

                [multi_indices, basis0, evalues0] = get_basis0(
                    fock_trunc, num_modes)
                evalue0 = get_expect_number(vector0, H_lin, vector0)
                vector1 = PT_on_vector(vector0, basis0, H_nl, evalues0, evalue0)

                index = np.argmax([(vector1.dag() * evec).norm()
                                for evec in evecs])
                return evals[index], evecs[index]

        else:
            def closest_state_to(s):
                def distance(s2):
                    return (s.dag() * s2[1]).norm()
                return max(zip(evals, evecs), key=distance)
        
        fzpfs_shape = fzpfs.shape
        print(fzpfs_shape)
        if fzpfs_shape[1] == 1:
            print('Single junctions -- assuming single qubit mode')
            # Distinguish the qubit mode from the resonator modes
            qubit_mode_ind = int(np.argmax(abs(fzpfs[:,0])))  # Find the index of the highest zero-point fluctuation mode (qubit mode)

            N_HO = [i for i in range(N) if i != qubit_mode_ind]  # Create a list of indices representing the resonator modes

            psi_0 = evecs[0]  # save ground state |0,0,0>
            rho_0 = psi_0.ptrace(N_HO)  # trace out qubit mode

            # Save the modes and frequencies that correspond to fluxonium excitations.
                    # this code segment iterates over the entire Hilbert space of the coupled system. 
                    # It calculates the distance between the approximated state of |n>q|0>r and each eigenstate, 
                    # and picks out the fisrt #fock_trunc states that gives the highest fidelity, 
                    # it then ranks them based on the magnitude of corresponding eigenvalues and
                    # saves both the eigenvalue and eigenstate associated with fluxonium excitations in the f_qubit list.
            fidelity_list = []
            f_qubit = [[],[]]
            for i in range(fock_trunc**N): 
                fidelity = (rho_0.dag() * evecs[i].ptrace(N_HO)).tr() # calculates the fidelity of the eigenstate to an approximated state where is resonator part is close to ground state
                fidelity_list.append(fidelity)
            
            biggest_elements = [] # Initialize a list to store the biggest fidelity and their indices, up to fock_trunc
            for i, element in enumerate(fidelity_list):
                if len(biggest_elements) < fock_trunc:
                    biggest_elements.append((element, i))
                    biggest_elements.sort(reverse=True)  # Sort in descending order based on element value
                else:
                    if element > biggest_elements[-1][0]:
                        biggest_elements[-1] = (element, i)
                        biggest_elements.sort(reverse=True)  # Sort in descending order based on element value

            sorted_biggest_elements = sorted(biggest_elements, key=lambda x: x[1])
            print("The biggest elements in the fidelity list are:")
            for element, index in sorted_biggest_elements:
                print(f"Element: {element}, Index: {index}")
                # Save the modes and frequencies that correspond to fluxonium excitations.
                f_qubit[0].append(evals[index])
                f_qubit[1].append(evecs[index])

            # Sanity check to make sure the fidelity of the found lowest 3 states to the approximate states are larger than 0.5
            def validate_excitation_identification(fidelity_list):
                fidelity_min = np.min(fidelity_list)
                if fidelity_min <= 0.5:
                    raise ValueError("No close qubit states found up to the second excited states, check modes at zero flux bias")

            fidelity_list = np.array([x[0] for x in sorted_biggest_elements[:3]])
            
            try:
                validate_excitation_identification(fidelity_list)
            except ValueError as e:
                print(f"Invalid input: {str(e)}")
                raise e    

            # Define list with all harmonic mode creation operator
            a_m = [[0] for _ in range(N)]
            for i in N_HO: # looping through all resonator modes
                a_m[i] = tensor_out(qutip.create(fock_trunc),i) # create an annihilation operator for the harmonic mode at index i.

    
            # Determine 0-1 frequencies for each mode
            f1s = [0 for _ in range(N)]
            for i in range(N):
                if i == qubit_mode_ind:
                    f1s[i] = f_qubit[0][1] - f_qubit[0][0]
                else:
                    f1s[i] = closest_state_to(a_m[i]*psi_0)[0]
            # ### add the higher energy levels of the qubit
            f1s.append(f_qubit[0][0])
            f1s.append(f_qubit[0][1])
            f1s.append(f_qubit[0][2])
            f1s.append(f_qubit[0][3])
            f1s.append(f_qubit[0][4])
            

            # Determine Kerr coefficients
            chis = [[0]*N for _ in range(N)]
            chips = [[0]*N for _ in range(N)]

            for i in range(N):
                for j in range(i, N):
                    if i == qubit_mode_ind and j == qubit_mode_ind:
                        fs = f_qubit[1][2] # found second excitation of qubit mode, used to calculate qubit anharmonicity
                    elif i == qubit_mode_ind or j == qubit_mode_ind:
                        if i == qubit_mode_ind:
                            fs = a_m[j]*f_qubit[1][1] # Found |1>q|1>r, used to calculate Cross-Kerr coefficient
                        else:
                            fs = a_m[i]*f_qubit[1][1]
                    else:
                        fs = a_m[i]*a_m[j]*psi_0 # found second excitation of resonator mode, used to calculate resonator anharmonicity

                    ev, evec = closest_state_to(fs)
                    chi = (ev - (f1s[i] + f1s[j]))
                    chis[i][j] = chi
                    chis[j][i] = chi

                    if chi_prime:
                        d[j] += 1
                        fs = fock_state_on(d)
                        ev, evec = closest_state_to(fs)
                        chip = (ev - (f1s[i] + 2*f1s[j]) - 2 * chis[i][j])
                        chips[i][j] = chip
                        chips[j][i] = chip

        else:
            print('Multiple junctions -- assuming transmons/harmonics oscillators only')
            f1s = [closest_state_to(fock_state_on({i: 1}))[0] for i in range(N)]
            chis = [[0] * N for _ in range(N)]
            chips = [[0] * N for _ in range(N)]
            for i in range(N):
                for j in range(i, N):
                    d = {k: 0 for k in range(N)}  # put 0 photons in each mode (k)
                    d[i] += 1
                    d[j] += 1
                    # load ith mode and jth mode with 1 photon
                    fs = fock_state_on(d)
                    ev, evec = closest_state_to(fs)
                    chi = (ev - (f1s[i] + f1s[j]))
                    chis[i][j] = chi
                    chis[j][i] = chi

                    if chi_prime:
                        d[j] += 1
                        fs = fock_state_on(d)
                        ev, evec = closest_state_to(fs)
                        chip = (ev - (f1s[i] + 2 * f1s[j]) - 2 * chis[i][j])
                        chips[i][j] = chip
                        chips[j][i] = chip

    if chi_prime:
        return np.array(f1s), np.array(chis), np.array(chips), np.array(fzpfs), np.array(f0s)
    else:
        return np.array(f1s), np.array(chis), np.array(fzpfs), np.array(f0s)

def black_box_hamiltonian_nq(freqs, zmat, ljs, cos_trunc=6, fock_trunc=8, show_fit=False):
    """
    N-Qubit version of bbq, based on the full Z-matrix
    Currently reproduces 1-qubit data, untested on n-qubit data
    Assume: Solve the model without loss in HFSS.
    """
    nf = len(freqs)
    nj = len(ljs)
    assert zmat.shape == (nf, nj, nj)

    imY = (1/zmat[:, 0, 0]).imag
    # zeros where the sign changes from negative to positive

    (zeros,) = np.where((imY[:-1] <= 0) & (imY[1:] > 0))
    nz = len(zeros)

    imYs = np.array([1 / zmat[:, i, i] for i in range(nj)]).imag
    f0s = np.zeros(nz)
    slopes = np.zeros((nj, nz))
    import matplotlib.pyplot as plt
    # Fit a second order polynomial in the region around the zero
    # Extract the exact location of the zero and the associated slope
    # If you need better than second order fit, you're not sampling finely enough
    for i, z in enumerate(zeros):
        f0_guess = (freqs[z+1] + freqs[z]) / 2
        zero_polys = np.polyfit(
            freqs[z-1:z+3] - f0_guess, imYs[:, z-1:z+3].transpose(), 2)
        zero_polys = zero_polys.transpose()
        f0s[i] = f0 = min(np.roots(zero_polys[0]),
                          key=lambda r: abs(r)) + f0_guess
        for j, p in enumerate(zero_polys):
            slopes[j, i] = np.polyval(np.polyder(p), f0 - f0_guess)
        if show_fit:
            plt.plot(freqs[z-1:z+3] - f0_guess, imYs[:, z-1:z +
                                                     3].transpose(), lw=1, ls='--', marker='o', label=str(f0))
            p = np.poly1d(zero_polys[0, :])
            p2 = np.poly1d(zero_polys[1, :])
            plt.plot(freqs[z-1:z+3] - f0_guess, p(freqs[z-1:z+3] - f0_guess))
            plt.plot(freqs[z-1:z+3] - f0_guess, p2(freqs[z-1:z+3] - f0_guess))
            plt.legend(loc=0)

    zeffs = 2 / (slopes * f0s[np.newaxis, :])
    # Take signs with respect to first port
    zsigns = np.sign(zmat[zeros, 0, :])
    fzpfs = zsigns.transpose() * np.sqrt(hbar * abs(zeffs) / 2)
    # fzpfs = [abs(x) for x in fzpfs]

    H = black_box_hamiltonian(f0s, ljs, fzpfs, cos_trunc, fock_trunc)
    
    return make_dispersive(H, fock_trunc, fzpfs, f0s)

black_box_hamiltonian_nq = black_box_hamiltonian_nq
