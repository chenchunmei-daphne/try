from fealpy.backend import backend as bm

def equalpower_subray(as_deg):
    """
    Get angle spacing for equal-power Laplacian PAS in SCM model.

    Parameters:
        as_deg (float): Angular spread in degrees.
            Valid values for BS: 2, 5
            Valid values for MS: 35

    Returns:
        list: Offset angles for M=20 case as shown in Table 2.2

    Raises:
        ValueError: If as_deg is not one of the supported values (2, 5, or 35)
    """
    if as_deg == 2:
        theta = [0.0894, 0.2826, 0.4984, 0.7431, 1.0257, 
                 1.3594, 1.7688, 2.2961, 3.0389, 4.3101]
    elif as_deg == 5:
        theta = [0.2236, 0.7064, 1.2461, 1.8578, 2.5642, 
                 3.3986, 4.4220, 5.7403, 7.5974, 10.7753]
    elif as_deg == 35:
        theta = [1.5649, 4.9447, 8.7224, 13.0045, 17.9492, 
                 23.7899, 30.9538, 40.1824, 53.1816, 75.4274]
    else:
        raise ValueError('Not support AS')
    
    return theta

def assign_offset(aoa_deg, as_deg):
    """
    Corrected version of assign_offset with symmetric offsets.
    
    Parameters:
        aoa_deg (np.ndarray): Average AoA/AoD angles in degrees.
        as_deg (float): Angular spread in degrees.
    
    Returns:
        np.ndarray: Array of angles with symmetric offsets (Npath × M).
    """
    # Get offset angles from equalpower_subray
    offset = bm.array(equalpower_subray(as_deg))
    
    # M = 20, so offset has 10 elements
    M = 20
    
    # Initialize result array: each path gets M angles
    num_paths = len(aoa_deg)
    result = bm.zeros((num_paths, M))
    
    for n in range(num_paths):
        # Generate symmetric offsets: +offset and -offset
        positive_offsets = aoa_deg[n] + offset
        negative_offsets = aoa_deg[n] - offset
        
        # Combine into one array (10 positive + 10 negative = 20 total)
        result[n, :] = bm.concatenate([positive_offsets, negative_offsets])
    
    return result

#################################################################################################

#################################################################################################
import numpy as np
from typing import Union, Optional, List

def gen_phase(bs_theta_los_deg: Union[float, List[float], np.ndarray],
              bs_as_deg: float,
              bs_aod_deg: Union[float, List[float], np.ndarray],
              ms_theta_los_deg: Union[float, List[float], np.ndarray],
              ms_as_deg: float,
              ms_aoa_deg: Union[float, List[float], np.ndarray],
              M: int = 20) -> tuple:
    """
    Generate phase information at BS and MS.
    
    Parameters:
        bs_theta_los_deg (float | list | np.ndarray): LOS path AoD at BS in degrees.
        bs_as_deg (float): Angular spread at BS in degrees.
        bs_aod_deg (float | list | np.ndarray): AoD at BS in degrees.
        ms_theta_los_deg (float | list | np.ndarray): LOS path AoA at MS in degrees.
        ms_as_deg (float): Angular spread at MS in degrees.
        ms_aoa_deg (float | list | np.ndarray): AoA at MS in degrees.
        M (int, optional): Number of subpaths. Defaults to 20.
    
    Returns:
        tuple: (bs_theta_deg, ms_theta_deg, bs_phi_rad)
            bs_theta_deg (np.ndarray): DoA for each path at BS (Npath × M) in degrees.
            ms_theta_deg (np.ndarray): DoA for each path at MS (Npath × M) in degrees.
            bs_phi_rad (np.ndarray): Random phase at BS (Npath × M) in radians.
    
    Notes:
        This function generates angle information for multipath channels and performs
        random pairing between BS and MS subpaths.
    """
    # Convert inputs to numpy arrays for uniform processing
    if not isinstance(bs_theta_los_deg, (list, np.ndarray)):
        bs_theta_los_deg = np.array([bs_theta_los_deg])
    if not isinstance(bs_aod_deg, (list, np.ndarray)):
        bs_aod_deg = np.array([bs_aod_deg])
    if not isinstance(ms_theta_los_deg, (list, np.ndarray)):
        ms_theta_los_deg = np.array([ms_theta_los_deg])
    if not isinstance(ms_aoa_deg, (list, np.ndarray)):
        ms_aoa_deg = np.array([ms_aoa_deg])
    
    # Ensure all inputs are numpy arrays
    bs_theta_los_deg = np.asarray(bs_theta_los_deg)
    bs_aod_deg = np.asarray(bs_aod_deg)
    ms_theta_los_deg = np.asarray(ms_theta_los_deg)
    ms_aoa_deg = np.asarray(ms_aoa_deg)
    
    # Generate random phase at BS (uniform distribution [0, 2π])
    num_paths = len(bs_aod_deg)
    bs_phi_rad = 2 * np.pi * np.random.rand(num_paths, M)
    
    # Calculate total angles (LOS + offset)
    bs_total_angles = bs_theta_los_deg + bs_aod_deg
    ms_total_angles = ms_theta_los_deg + ms_aoa_deg
    
    # Generate offset angles (corrected version with symmetric offsets)
    bs_theta_deg = _assign_offset_corrected(bs_total_angles, bs_as_deg)
    ms_theta_deg = _assign_offset_corrected(ms_total_angles, ms_as_deg)
    
    # Random pairing (shuffle MS angles for each path independently)
    ms1 = ms_theta_deg.shape[0]
    for n in range(ms1):
        indices = np.random.permutation(M)
        ms_theta_deg[n, :] = ms_theta_deg[n, indices]
    
    return bs_theta_deg, ms_theta_deg, bs_phi_rad



def ray_fading(M: int,
               pdp: np.ndarray,
               bs_theta_deg: np.ndarray,
               bs_phi_rad: np.ndarray,
               ms_theta_deg: np.ndarray,
               v_ms: float,
               theta_v_deg: float,
               wavelength: float,
               t: Union[float, np.ndarray]) -> np.ndarray:
    """
    Generate fading for each subpath.
    
    Parameters:
        M (int): Number of subpaths.
        pdp (np.ndarray): 1×Npath power delay profile.
        bs_theta_deg (np.ndarray): DoA for each path at BS (Npath × M) in degrees.
        bs_phi_rad (np.ndarray): Random phase at BS (Npath × M) in radians.
        ms_theta_deg (np.ndarray): DoA for each path at MS (Npath × M) in degrees.
        v_ms (float): Mobile speed in m/s.
        theta_v_deg (float): Direction of travel in degrees.
        wavelength (float): Wavelength in meters.
        t (float | np.ndarray): Current time or time vector.
    
    Returns:
        np.ndarray: Channel coefficients (1 × length(PDP)).
    
    Notes:
        This function implements equation (2.32) from the textbook.
    """
    # Ensure t is a numpy array for vectorized operations
    if not isinstance(t, np.ndarray):
        t = np.array([t])
    
    # Convert degrees to radians
    ms_theta_rad = np.deg2rad(ms_theta_deg)
    theta_v_rad = np.deg2rad(theta_v_deg)
    
    # Pre-calculate Doppler frequency component
    # Doppler shift: f_d = (v/λ) * cos(θ - θ_v)
    doppler_factor = 2 * np.pi / wavelength * v_ms
    
    # Initialize channel coefficients
    num_paths = len(pdp)
    num_time_samples = len(t)
    h = np.zeros((num_paths, num_time_samples), dtype=complex)
    
    for n in range(num_paths):
        # Calculate Doppler shift for each subpath
        # Shape: (M, 1) * (1, num_time_samples) = (M, num_time_samples)
        cos_term = np.cos(ms_theta_rad[n, :, np.newaxis] - theta_v_rad)
        doppler_shift = doppler_factor * cos_term * t[np.newaxis, :]
        
        # Combine BS phase and Doppler shift
        # BS phase term: shape (M, 1)
        bs_phase_term = np.exp(-1j * bs_phi_rad[n, :, np.newaxis])
        
        # Doppler term: shape (M, num_time_samples)
        doppler_term = np.exp(-1j * doppler_shift)
        
        # Combined effect: shape (M, num_time_samples)
        combined = bs_phase_term * doppler_term
        
        # Sum over subpaths and scale by power
        # sqrt(PDP(n)/M) * sum over M subpaths
        scaling_factor = np.sqrt(pdp[n] / M)
        h[n, :] = scaling_factor * np.sum(combined, axis=0)
    
    return h


def db2w(dB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert dB to linear (Watts) scale.
    
    Parameters:
        dB (float | np.ndarray): Value(s) in dB.
    
    Returns:
        float | np.ndarray: Value(s) in linear scale.
    
    Formula:
        y = 10^(dB/10)
    """
    return 10 ** (0.1 * dB)


def deg2rad(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert degrees to radians.
    
    Parameters:
        deg (float | np.ndarray): Angle(s) in degrees.
    
    Returns:
        float | np.ndarray: Angle(s) in radians.
    """
    return deg * np.pi / 180.0