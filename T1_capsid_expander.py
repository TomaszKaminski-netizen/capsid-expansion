"""This script converts a viral capsid with T=1 triangulation number into
either a T=3 or T=4 capsid. Developed with Python 3.10.16, Pandas 2.2.3,
NumPy 2.2.3, PyMol-open-source 3.1.0, SciPy 1.15.2, Matplotlib 3.10.0, Boltons
24.0.0, and PDBFixer 1.11, all in an Anaconda environment."""
#pylint: disable=invalid-name, logging-fstring-interpolation, unnecessary-lambda-assignment, too-many-lines

from os.path import isfile, dirname, abspath # Works with both Linux and Windows
from os import chdir, remove
from sys import argv
from itertools import combinations, cycle, product
from statistics import mean
from math import isclose, degrees, sqrt, atan, acos, asin, pi, cos, sin
from re import match, sub
from string import ascii_uppercase
import logging
import argparse

from boltons.funcutils import wraps as bolton_wraps
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde, linregress
from matplotlib import pyplot as plt
from pymol import cmd, finish_launching
import numpy as np
import pandas as pd
from openmm.app import PDBFile
from pdbfixer import PDBFixer

# Constants used in the functions below.
PHI_NUM = (1 + 5**0.5) / 2 # The golden ratio
ICO_DIHED = degrees(acos(-sqrt(5)/3))
DODEC_DIHED = degrees(2 * atan(PHI_NUM))
# The angle between the base of a regular pentagonal pyramid and one of its flat sides.
PENT_PYR_ANG = degrees(asin(sqrt(2 - 0.4 * sqrt(5)) / sqrt(3)))

def icosahedron_spherical_cordinates():
    """Any capsid with icosahedral symmetry can be imagined as two staggered
    rings of 5 "pentamers" each, plus one "pentamer" on the north and south
    poles. This function returns a list of spherical coordinates of these
    "pentamers" (i.e. corners).

    Returns:
        list: list of tuples, containing 12 pairs of an angle + an axis around
            which to rotate in order to reach each capsid corner
    """
    latitude_1, latitude_2 = degrees(atan(0.5)) - 90, 90 + degrees(atan(0.5))
    # See mathworld.wolfram.com/RegularPentagon.html for detailed explanation.
    c1, c2, s1, s2 = cos(0.4 * pi), cos(0.2 * pi), sin(0.4 * pi), sin(0.8 * pi)
    longitudes = ([1, 0, 0], [c1, -s1, 0], [-c2, -s2, 0], [-c2, s2, 0], [c1, s1, 0])
    ring_1 = list(zip([latitude_1] * 5, longitudes))
    ring_2 = list(zip([latitude_2] * 5, longitudes))
    return [(180, "z"), *ring_1, *ring_2, (180, "y")]

ICO_SPH_COORDS = icosahedron_spherical_cordinates()

# Utility functions
normalise = lambda vector: (vector / np.linalg.norm(vector)).flatten()
# Some methods & functions (cdist) work only with 2D numpy arrays, whilst others demand 1D arrays.
centerofmass_1D = lambda selection: np.array(cmd.centerofmass(selection))
centerofmass_2D = lambda selection: np.array(cmd.centerofmass(selection)).reshape(1, -1)
# By default the rotation happens relative to camera coordinates, which is ridiculous. Setting
# camera=0 makes the function use absolute coordinates instead.
pymol_rotate = lambda *args, **kwargs: cmd.rotate(*args, camera=0, **kwargs)

####################################################################################################

def load_file(rest_of_filename):
    """This decorator factory has three purposes: clear the PyMol window of all
    structures, set the pdb_of_interest and multimer_type arguments to all
    lowercase, and try to load a new file in PyMol, taking into account what
    multimer type the user is interested in. If the desired file can be found,
    the decorated function (called func here) is activated. If not, return
    False.

    Args:
        rest_of_filename (str): name of the file that the decorated function
            wants to load, missing the first 4 characters. If the type of
            multimer needs to be specified within this string, write ?multimer?
            in that position.

    Returns:
        any: either the results of the decorated function or False.
    """
    def decorator(func):

        @bolton_wraps(func) #? Use boltons wraps or functools wraps?
        def wrapper(pdb_of_interest, multimer_type, *args, **kwargs):
            pdb_of_interest = pdb_of_interest.lower()
            multimer_type = multimer_type.lower()
            if multimer_type not in ["trimer", "pentamer"]:
                log.error(f"Wrong multimer_type specified '{multimer_type}', stopping.") #pylint: disable=possibly-used-before-assignment
                return False
            full_filename = pdb_of_interest + rest_of_filename.replace("?multimer?", multimer_type)
            if not isfile(full_filename):
                log.error(f"No {full_filename} file has been found, stopping.")
                return False
            cmd.delete("all")
            cmd.load(full_filename)
            return func(pdb_of_interest, multimer_type, *args, **kwargs)

        return wrapper
    return decorator


def prepare_pdb(pdb_of_interest, t_num):
    """This function takes raw pdb files of capsids, removes all non-protein
    atoms, puts each protein monomer into its own object, and centers the
    capsid around origin. The cleaned-up structure is saved as a new .cif file.

    Args:
        t_num (int): the triangulation number of the input structure
        pdb_of_interest (str): 4-character pdb code, case-insensitive

    Returns:
        bool: a statement whether the function has been ran successfully
    """
    cmd.delete("all")
    # Fetched files are named in lower-case. Cases matter for Linux isfile(), but not Windows.
    pdb_of_interest = pdb_of_interest.lower()
    cmd.fetch(pdb_of_interest, async_=0, type="pdb1") # For X-ray structures
    #? How is it even possible that Python can't detect PyMol errors/exceptions?
    if not isfile(f"{pdb_of_interest}.pdb1"):
        cmd.fetch(pdb_of_interest, async_=0, type="cif") # For cryo EM structures
        if not isfile(f"{pdb_of_interest}.cif"):
            log.error(f"{pdb_of_interest} Neither cif nor pdb1 file could be fetched, stopping.")
            return False

    # Inelegant, but list comprehensions are not compatible with cmd.iterate.
    atoms = {"atoms": []}
    selection_criteria = "byres first not resn GLY & present & polymer.protein" # Must use the &
    cmd.iterate(selection_criteria, "atoms.append(name)", space=atoms)
    if "N" not in atoms["atoms"]:
        log.error(f"{pdb_of_interest} Stucture contains only alpha carbons, stopping.")
        return False
    if "CB" not in atoms["atoms"]:
        log.error(f"{pdb_of_interest} Stucture contains only backbone atoms, stopping.")
        return False

    cmd.remove("not polymer.protein")
    cmd.remove("not alt ''+A") # Removing alternate atom locations.
    cmd.alter("(all)", "alt=''")
    #* This function can't process pdb files that utilise segments, but such files are very rare.
    cmd.alter("(all)", "segi=''")

    if cmd.count_states(pdb_of_interest) > 1:
        cmd.split_states(pdb_of_interest)
        # Splitting the states does not automatically delete the multi-state object.
        cmd.delete(pdb_of_interest)
    #TODO Find a way to preserve inter-chain S-S bonds, as they will be broken by split_chains().
    for pymol_object in cmd.get_names("objects"):
        if len(cmd.get_chains(pymol_object)) > 1:
            # The prefix is necessary - without it some chains are lost in files where both upper-
            # and lower-case letters are used as chain designations (e.g. pdb 6V10).
            #? PyMol 1.8.2 is expected to add an ignore_case_chain setting. Use it here?
            cmd.split_chains(pymol_object, prefix=f"{pymol_object}_ch")
            cmd.delete(pymol_object)
    if len(cmd.get_names("objects")) != 60 * t_num:
        log.error(f"{pdb_of_interest} Structure doesn't contain {60 * t_num} subunits, stopping.")
        return False

    # Centers the capsid around the origin.
    [x_offset, y_offset, z_offset] = cmd.centerofmass("(all)")
    cmd.translate([-x_offset, -y_offset, -z_offset], "(all)", camera=0)

    cmd.save(f"{pdb_of_interest}_clean.cif") # Not .pdb
    log.info(f"{pdb_of_interest} File fas been cleaned, continuing.")
    return True


@load_file("_clean.cif")
def group_monomers(pdb_of_interest, multimer_type):
    """This function takes a cleaned pdb structure and finds the 20 trimers or
    12 pentamers within it using geometric analysis. For each multimer, the
    monomers are grouped together as three/five chains in a single PyMol
    object, respectively. This re-arranges structure is saved as a new .cif
    file.

    Args:
        pdb_of_interest (str): 4-character pdb code, case-insensitive
        multimer_type (str): what multimer the function will search for, either
            "pentamer" or "trimer".

    Returns:
        bool: a statement whether the function has been ran successfully
    """
    monomers = cmd.get_names("objects")
    monomer_coords = [centerofmass_2D(mono) for mono in monomers]
    df_monomer_coords = pd.DataFrame({"xyz": monomer_coords}, index=monomers)

    chain_desig = cycle("abc" if multimer_type == "trimer" else "abcde")
    # Allows for up to 260 designations; A1, B1, C1, etc.
    multimer_desig = iter("".join(item[::-1]) for item in product("0123456789", ascii_uppercase))
    # This loop searches through df_monomer_coords for multimers, removing the constituent monomers
    # each time a new trimer/pentamer is found, until the array is empty.
    while df_monomer_coords.shape != (0, 1):
        multimer_monos = find_multimer(df_monomer_coords, multimer_type)
        if not multimer_monos:
            log.error(f"{pdb_of_interest} Couldn't detect all {multimer_type}s, "
                      f"{df_monomer_coords.shape} monomers remaining, stopping.")
            return False
        df_monomer_coords = df_monomer_coords.drop(index=multimer_monos)

        # The monomers need to be given distinct chain names before they are merged into a single
        # PyMol object. By default the names are ascending in the clockwise (when viewed from the
        # exterior of the capsid) direction.
        np_multimer = centerofmass_1D(" ".join(multimer_monos))
        rot_angles = [0] # Self-comparison yields an angle of 0
        for mono_2 in multimer_monos[1:]:
            rot_angles.append(calculate_rotation(multimer_monos[0], centerofmass_1D(mono_2),
                                                 fixed_point=np_multimer, rotate=False))

        for monomer in [multimer_monos[i] for i in np.argsort(rot_angles)]:
            cmd.alter(monomer, f"chain='{next(chain_desig)}'")
        #* Merging the monomers introduces incorrect inter-chain bonds, but this can't be avoided.
        cmd.create(f"{pdb_of_interest}_{multimer_type[:3]}{next(multimer_desig)}",
                   " ".join(multimer_monos))
        cmd.delete(" ".join(multimer_monos))

    cmd.save(f"{pdb_of_interest}_{multimer_type}.cif")
    log.info(f"{pdb_of_interest} All {multimer_type}s have been found, continuing.")
    return True


def find_multimer(coords, multimer_type, prot_of_interest=None):
    """This utility function uses distance data to find the trimer, pentamer,
    or superpentamer that contains given protein(s) of interest. The trimer and
    super-pentamer detection works for T1, T3, and T4, but the pentamer
    detection works for T1. Note that a suitable PyMol file must already be
    loaded for this function to work.

    Args:
        coords (DataFrame): pandas DataFrame specifying the locations and names
            of all the proteins that have not been assigned to a multimer yet
        multimer_type (str): what multimer the function will search for, either
            "pentamer" or "trimer". Use "pentamer" when looking for super-
            pentamers, and make sure that the PyMol objects in your file are
            trimers, rather than individual monomer proteins.
        prot_of_interest (str, optional): name of the PyMol object(s)
            corresponding to the protein(s) of interest. None by default, in
            the case which the first object in the PyMol object list is picked
            to be of interest.

    Returns:
        list or False: a list with names of objects comprising the multimer
    """
    # Making a copy, so that addition of the "dist" column does not propagate backwards to the
    # original DataFrame.
    coords = coords.copy(deep=False)
    central_prot = coords.index[0] if prot_of_interest is None else prot_of_interest
    np_central_prot = coords.loc[central_prot]["xyz"]
    coords["dist"] = coords["xyz"].apply(lambda x: cdist(x, np_central_prot))
    # Finding the six closest neighbors was enough for almost all cases, but 2C9F needed nine to
    # find its trimers. 6NXE needed 12 neighbors to find its pentamers.
    neighbor_count = 12 if multimer_type == "pentamer" else 9
    neighbors = coords.sort_values("dist").iloc[:neighbor_count+1, :1]

    # Skipping over central_prot. Combinations returns ([A, B], [B, C], [A, C]) from [A, B, C].
    for pair in combinations(neighbors.index[1:], r=2):
        if multimer_type == "pentamer":
            np_two_of_five = centerofmass_2D(" ".join(pair))
            # The center of mass of two non-adjacent monomers in a pentamer lies close to the center
            # of mass of the pentamer / super-pentamer itself, which I exploit in this step.
            neighbors["dist"] = neighbors["xyz"].apply(lambda x: cdist(x, np_two_of_five)) #pylint: disable=cell-var-from-loop
            multimer = neighbors.sort_values("dist").iloc[0:5, :1]
        else:
            multimer = neighbors.filter(items=[central_prot, *pair], axis="index")

        if central_prot not in multimer.index:
            log.info("Multimer did not contain the protein of interest, continuing.")
            continue

        # Validating whether all the monomers are equally distant from the multimer center.
        multimer_proteins = " ".join(multimer.index)
        # multimer["xyz"] gives a nested array, which vstack() converts to a 2D array. Using stack()
        # here instead gave an incorrect 3D-array.
        dist_from_multi = cdist(np.vstack(multimer["xyz"]), centerofmass_2D(multimer_proteins))
        mean_dist = dist_from_multi.mean()
        if not all(isclose(dist, mean_dist, rel_tol=0.02) for dist in dist_from_multi.flatten()):
            log.info("Wrong geometry, continuing.")
            continue

        # Confirming that the monomers are actually touching each other.
        for one in multimer.index:
            others = multimer_proteins.replace(one, "")
            # 4 Angstroms was not enough for 4Y4Y trimers (despite having decent interface area).
            cmd.select(f"({one} within 6 of ({others})) or (({others}) within 6 of {one})")
            if cmd.count_atoms(r"%sele") < 60:
                log.info("Geometric multimer had no contacts, continuing.")
                break
        else:
            return list(multimer.index)
    return False


@load_file("_?multimer?.cif")
def find_typical_multimer(pdb_of_interest, multimer_type): #pylint: disable=unused-argument
    """This function looks for the most typical trimer/pentamer in a capsid,
    which could be used as a representative multimer for the whole structure.
    It does so by comparing the mean RMSD of every multimer aligned
    sequentially with every other multimer.

    Args:
        pdb_of_interest (str): 4-character pdb code, case-insensitive

    Returns:
        str or False: name of the PyMol object containing the typical trimer
    """
    lowest_rmsd_multimer = (None, 2)
    multimer_names = cmd.get_names("objects")
    for index, multimer_1 in enumerate(multimer_names):
        multimer_1_rmsd = []
        # Skipping over multimer_1 when choosing multimer_2, to skip the useless self-comparison.
        for multimer_2 in multimer_names[:index] + multimer_names[index+1:]:
            # Doing 20 alignments with multimer_1 itself would cause the RMSD to roughly triple with
            # every alignment, so instead we make a fresh copy of multimer_1 every time.
            cmd.copy(f"{multimer_1}_copy", multimer_1)
            #? Why is it impossible to align 6ih9_triA to 6ih9_triI when cycles is set to 0?
            alignment = cmd.align(f"{multimer_1}_copy", multimer_2, cycles=1)
            # We are only interested in RMSD after (1 cycle of) refinement if the refinement had a
            # huge effect. This happens only in alignments that give "no convergence" PyMol warning.
            pair_rmsd = alignment[3] if (alignment[3] - alignment[0]) < 10 else alignment[0]
            cmd.delete(f"{multimer_1}_copy")
            multimer_1_rmsd.append(pair_rmsd)
        if lowest_rmsd_multimer[1] > mean(multimer_1_rmsd):
            lowest_rmsd_multimer = (multimer_1, mean(multimer_1_rmsd))
    log.info(f"{lowest_rmsd_multimer[0]} Typical {multimer_type} found, continuing.")
    return lowest_rmsd_multimer[0]


def find_two_superpentamers(trimer_of_interest=None):
    """This function finds two adjacent (overlapping) super-pentamers in a T1
    capsid. Note that a suitable PyMol file must already be loaded for this
    function to work.

    Args:
        trimer_of_interest (str, optional): name of the (single) PyMol object
            corresponding to the trimer of interest. None by default, in the
            case which the first trimer in the PyMol object list is picked to
            be of interest.

    Returns:
        list or False: a nested list with the centers of mass of the two found
            super-pentamers and the names of their constituent trimers
    """
    all_trimers = cmd.get_names("objects")
    central_trimer = all_trimers[0] if trimer_of_interest is None else trimer_of_interest
    trimer_coords = [centerofmass_2D(trimer) for trimer in all_trimers]
    df_trimer_coords = pd.DataFrame({"xyz": trimer_coords}, index=all_trimers)

    superpents = []
    superpents.append(find_multimer(df_trimer_coords, "pentamer", central_trimer))
    if False in superpents:
        log.error(f"{central_trimer[0:4]} Could not find the first superpentamer, stopping.")
        return False
    # Selecting a random trimer (but not the trimer of interest, which is at [0][0]) and removing it
    # from the DataFrame, so that the second superpentamer won't be identical to the first.
    df_trimer_coords = df_trimer_coords.drop(index=superpents[0][-1])
    superpents.append(find_multimer(df_trimer_coords, "pentamer", central_trimer))
    if False in superpents:
        log.error(f"{central_trimer[0:4]} Could not find the second superpentamer, stopping.")
        return False

    log.info(f"{central_trimer[0:4]} Found two superpentamers, continuing.")
    return [[" ".join(superpent), centerofmass_1D(" ".join(superpent))] for superpent in superpents]


def orient_capsid(two_superpents, trimer_of_interest=None):
    """This function uses the coordinates of two adjacent super-pentamers to
    rotate a T1 capsid into the following orientation:
    i.pinimg.com/originals/06/2e/a2/062ea2dc61f1fb54b94da4f561554ace.png
    If a trimer of interest is specified, the function will make sure it ends
    up between the (-1,0,phi), (1,0,phi), and (0,-phi,1) vertexes. Lastly, the
    function marks the locations of the (-1,0,phi) and (1,0,phi) vertexes with
    pseudoatoms and saves the rotated capsid as a new .cif file. Note that a
    suitable PyMol file must already be loaded for this function to work.

    Args:
        list: a nested list with the centers of mass of the two adjacent super-
            pentamers and the names of their constituent trimers
        trimer_of_interest (str, optional): name of the PyMol object
            corresponding to the trimer of interest. Note that this trimer must
            be a member of both specified super-pentamers. None by default.

    Returns:
        bool: a statement whether the function has been ran successfully
    """
    [superpent_1, superpent_2] = two_superpents
    # Calculating the desired coordinates of the first super-pentamer.
    dist_from_origin = np.linalg.norm(superpent_1[1])
    # Pythagoras theorem and the golden ratio; dist_from_origin**2 = x**2 + (x*PHI_NUM)**2
    x = sqrt(dist_from_origin**2 / (1 + PHI_NUM**2))
    np_wanted_superpent_1 = np.array([x, 0, x*PHI_NUM])
    # Rotating the whole capsid to move the first super-pentamer to its desired position.
    superpent_1[1] = calculate_rotation(superpent_1[0], np_wanted_superpent_1)
    if isinstance(superpent_1[1], bool):
        return False
    # This rotation also moved the second super-pentamer.
    superpent_2[1] = centerofmass_1D(superpent_2[0])

    # The desired coordinate of the second super-pentamer is the same as the first, but with -ve X.
    np_wanted_superpent_2 = superpent_1[1] * [-1, 1, 1]
    # Rotating the second super-pentamer into desired position without disturbing the first one.
    superpent_2[1] = calculate_rotation(superpent_2[0], np_wanted_superpent_2, superpent_1[1])
    if isinstance(superpent_2[1], bool):
        return False

    # Marking the vertex position now (not later) in case the next step rotates the capsid again.
    cmd.pseudoatom("vertex_10p", pos=superpent_1[1].tolist())
    # Ensuring the trimer of interest forms the correct icosahedron face: (-1,0,phi), (1,0,phi), and
    # (0,-phi,1), rather than (0,phi,1), which is the second possibility.
    if trimer_of_interest:
        if trimer_of_interest not in superpent_1[0] or trimer_of_interest not in superpent_2[0]:
            log.error(f"{trimer_of_interest} Trimer of interest is not a member of the specified "
                       "super-pentamers, stopping.")
            return False
        if cmd.centerofmass(trimer_of_interest)[1] > 0: # Is the y coordinate positive
            pymol_rotate(selection="not vertex_10p", angle=180, origin=[0, 0, 0], axis="z")
        log.info(f"{trimer_of_interest} Trimer lays on the correct face, continuing.")

    cmd.save(f"{superpent_1[0][0:4]}_trimer_oriented.cif")
    log.info(f"{superpent_1[0][0:4]} Capsid orientation complete, continuing.")
    return True


def calculate_rotation(proteins, wanted_coord, fixed_point=None, rotate=True):
    """This utility function performs calculations to figure out how to rotate
    the capsid to to move a particular protein / group of proteins to a
    specified new location. If desired, the function can then actually perform
    the rotation. All the maths done here is using numpy arrays. Note that a
    a suitable PyMol file must already be loaded for this function to work, and
    that pseudoatoms can not be provided instead of proteins (because they lack
    mass).

    Args:
        proteins (list): PyMol selection with the protein(s) of interest
        wanted_coord (array): 1-by-3 numpy arrray specifying the desired
            location of the center of mass of the protein(s) of interest
        fixed_point (array, optional): 1-by-3 numpy arrray specifying the
            [x, y, z] coordinates of a point which we wish to remain
            unperturbed by the rotation. Defaults to None.
        rotate [bool, optional]: whether to actually perform the rotation.
            Default to True.

    Returns:
        array or False or int: either a 1-by-3 numpy arrray specifying the new
            center of mass (or rotation was performed) or the rotation angle
            (if no rotation was done)
    """
    current_coord = centerofmass_1D(proteins)
    if fixed_point is None:
        norm_wanted, norm_current = normalise(wanted_coord), normalise(current_coord)
        rot_axis = normalise(np.cross(norm_wanted, norm_current)).tolist()
    else:
        norm_fixed = normalise(fixed_point)
        rot_axis = norm_fixed.tolist()
        # Finding the closest point to the current (and wanted) coords, laying on the rotation axis.
        closest_point = norm_fixed * np.dot(current_coord, norm_fixed)
        # Using it to get the angle of rotation around my axis.
        norm_wanted = normalise(wanted_coord - closest_point)
        norm_current = normalise(current_coord - closest_point)

    rot_angle = degrees(np.arccos(np.dot(norm_wanted, norm_current)))
    rot_sign = float(np.sign(np.dot(rot_axis, np.cross(norm_wanted, norm_current))))

    if rotate: #pylint: disable=no-else-return
        # For PyMol, a positive angle would mean counter-clockwise rotation, which we don't want.
        pymol_rotate(angle=rot_angle * rot_sign * -1, axis=rot_axis, origin=[0, 0, 0])
        new_coord = centerofmass_1D(proteins)
        if np.abs(new_coord - wanted_coord).max() > 2: # 2 Angstrom leeway
            log.error(f"{proteins[0:4]} Rotation not successful, stopping.")
            return False
        return new_coord
    else:
        # The angles are always measured in a clockwise rotation direction, if viewed by looking
        # from the outside of the capsid inwards.
        return rot_angle if rot_sign > 0 else 360 - rot_angle


@load_file("_?multimer?_oriented.cif")
def prepare_for_interface_morph(pdb_of_interest, multimer_type, hydrogens=False): #pylint: disable=unused-argument
    """This function takes one trimer from an oriented T1 capsid and prepares
    it for morphing into a T3/T4 capsid trimer. Missing atoms are filled in,
    including hydrogens if so desired, and the origin around which the trimer
    will be rotated to create the T3/T4 geometry is found.

    Args:
        pdb_of_interest (str): 4-character pdb code, case-insensitive
        hydrogens (bool, optional): Whether or not hydrogen atoms should be
            simulated during PyMol sculpting. Defaults to False.

    Returns:
        str: name of the PyMol object containing the expansion trimer
    """
    # Finding the expansion trimer, located between vertexes (-1,0,phi), (1,0,phi), and (0,-phi,1).
    [one, _, phi] = cmd.get_coords("vertex_10p").flatten()
    np_3_vertex_mean = np.array([0, -phi / 3, (phi + phi + one) / 3]).reshape(1, -1)
    trimers = cmd.get_names("objects", selection="polymer.protein")
    np_trimers = np.array([cmd.centerofmass(tri) for tri in trimers])
    distances = cdist(np_trimers, np_3_vertex_mean)
    expansion_trimer = trimers[np.argsort(distances, axis=None)[0]] # Nearest to np_3_vertex_mean

    # Duplicating the expansion trimer to create the new T3/T4 inter-trimer interface.
    cmd.delete(f"not {expansion_trimer}")
    cmd.pseudoatom("vertex_0-p1", pos=[0, -phi, one])
    cmd.copy("tri_1", expansion_trimer)
    pymol_rotate(angle=180, axis="z", origin=[0, 0, 0], selection="tri_1")

    # Finding the origin of rotation by looking for maximum atom density on the XZ plane (which is
    # running right across the inter-trimer interface).
    y_leeway = 5 #? Optimise this value?
    cmd.select(f"y > -{y_leeway} & y < {y_leeway} & ({expansion_trimer} within 5 of tri_1)")
    interface_atom_z_coords = cmd.get_coords(r"%sele")[:, 2]
    # The data is smoothed using gaussian kernel density estimation
    smoothed_data = gaussian_kde(interface_atom_z_coords).pdf(interface_atom_z_coords)
    maximum_density_z_coord = interface_atom_z_coords[np.argmax(smoothed_data)]
    cmd.pseudoatom("interface_rot_origin", pos=[0, 0, maximum_density_z_coord])

    # Saving the data graphically for manual analysis.
    plt.title(pdb_of_interest + " interface density")
    plt.scatter(interface_atom_z_coords, smoothed_data)
    plt.axvline(maximum_density_z_coord)
    plt.text(maximum_density_z_coord, plt.ylim()[0],
             # The spaces & newlines are necessary for correct textbox positioning. Also, for some
             # reason using round() in f-strings does not work correctly.
             f" Maximum \n density at \n {maximum_density_z_coord:.1f} Å \n ",
             verticalalignment="bottom", horizontalalignment="right")
    plt.xlabel("Z coordinate")
    plt.ylabel("Atomic density within 5 Å of the interface plane")
    #? Try to make the y-axis units equal to the number of atoms per cubic Å?
    plt.yticks([])
    plt.savefig(pdb_of_interest + "_interface_density.png", bbox_inches="tight")
    plt.clf()

    # Filling in missing atoms (e.g. ends of lysine side-chains) and hydrogens with PDBFixer.
    cmd.save(f"{pdb_of_interest}_temp1.pdb", selection=expansion_trimer)
    cmd.delete(f"{expansion_trimer} tri_1")
    fixer = PDBFixer(f"{pdb_of_interest}_temp1.pdb")
    remove(f"{pdb_of_interest}_temp1.pdb")
    fixer.findMissingResidues()
    if len(fixer.missingResidues) > 10: #? Users are unlikely to want to add huge domains back in?
        fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, f"{pdb_of_interest}_temp2.pdb", keepIds=True)
    cmd.load(f"{pdb_of_interest}_temp2.pdb", object=expansion_trimer)
    remove(f"{pdb_of_interest}_temp2.pdb")

    # Having all the hydrogens present roughly doubles the simulation time.
    if not hydrogens:
        cmd.remove("hydrogens")
    # Keeping track of the initial xyz origin with a pseudoatom, as it will move during morphing.
    cmd.pseudoatom("xyz_origin", pos=[0, 0, 0])
    cmd.save(f"{pdb_of_interest}_for_morphing.cif")
    log.info(f"{expansion_trimer} Trimer has been prepared for morphing, continuing.")
    return f"{pdb_of_interest}_for_morphing.cif"


def create_tx_interface(t_num, rot_steps=30, cutoff=20):
    """This function takes a T1 capsid trimer, surrounds it with copies of
    itself, and slowly changes the angle between the copies until they reach a
    T3/T4-like geometric arrangement. Steric clashes arising at the inter-
    trimer interface(s) during the rotation are resolved via PyMol sculpting.
    The final morphed structure is saved as a new .cif file. Note that a
    suitable PyMol file must already be loaded for this function to work. For
    expansion to T3 geometry, this function can deal with both standard trimers
    (three chains) and linked trimers (one chain), but for T4 geometry only
    standard trimers are accepted.

    Args:
        t_num (int): either 3 or 4, specifies which capsid geometry to aim for
        rot_steps (int, optional): How many steps the rotation should be split
            into. Defaults to 30.
        cutoff (int, optional): Atoms located farther from the newly created
            interface(s) than this cutoff distance are fixed during sculpting,
            in order to speed up the simulation. Defaults to 20 Angstroms.
    """
    # Preparing for trimer rotation.
    t_num = int(t_num)
    supplement_steps  = (t4_supplement if t_num == 4 else t3_supplement)(rot_steps, cutoff)
    rot_angle = next(supplement_steps) #* Step 1, finding the rotation angle for T3 & T4.
    interface_rot_origin = cmd.get_coords("interface_rot_origin").tolist()[0]
    trimer_name = cmd.get_names("objects", selection="polymer.protein")[0]
    cmd.set_name(trimer_name, "tri_0") # A short & consistent name
    #? As seen in 2C9F, tri_1 can clash heavily with tri_2/3. Fix this by using tri_4/5 even for T3?
    partners = [f"tri_{i}" for i in ([2, 3, 4, 5] if t_num == 4 else [1, 2, 3])]
    next(supplement_steps) #* Step 2, making tri_1 for T4.

    # Slowly rotating the trimers until we reach the new interface angle.
    for rot_step in range(1, rot_steps + 1):
        cmd.delete("sculpt_object")
        sele = ("not tri_1_nonsymm", "tri_1_nonsymm") if t_num == 4 else ("(all)", "none")
        pymol_rotate(origin=interface_rot_origin, axis="x", angle=rot_angle, selection=sele[0])
        pymol_rotate(origin=interface_rot_origin, axis="x", angle=rot_angle * -1, selection=sele[1])

        # Figuring out the (five-fold) super-pentamer axis of symmetry after rotation.
        np_superpent = cmd.get_coords("vertex_0-p1")
        np_shifted_origin = cmd.get_coords("xyz_origin")
        superpent_axis = (np_superpent - np_shifted_origin).tolist()[0]
        shifted_origin = np_shifted_origin.tolist()[0]

        next(supplement_steps) #* Step 3, symmetrising tri_1 for T4.

        chain_identi = iter("ghijklmnopqr" if t_num == 4 else "defghijkl")
        # Re-making all the interaction partners fresh after every sculpt and rotation.
        for partner in partners:
            cmd.create(partner, "tri_0")
            for chain in "abc":
                cmd.alter(f"{partner} & chain {chain}", f"chain='{next(chain_identi)}'")
        pymol_rotate(angle=72, axis=superpent_axis, origin=shifted_origin, selection="tri_2")
        pymol_rotate(angle=-72, axis=superpent_axis, origin=shifted_origin, selection="tri_3")
        # send() works just like next() with the added benefit of sending values into the generator.
        #* Step 4, rotating tri_1 into position for T3 and tri_4/5 for T4.
        supplement_steps.send([shifted_origin, superpent_axis])

        # Fixing certain atoms in place to make the simulation faster and restraining the rest.
        mask = next(supplement_steps) #* Step 5, choosing which atoms to fix for T3 & T4.
        cmd.flag("3", "not byres ((" + ") (".join(mask) + "))") # Fixed atoms.

        # All the trimers need to be part of the same PyMol object for PyMol sculpt to work.
        cmd.create("sculpt_object", "tri_*")
        cmd.delete("tri_*")
        for chain in cmd.get_chains("sculpt_object"):
            cmd.unbond(f"chain {chain}", f"not chain {chain}")

        # PyMol sculpting removes steric clashes and bad bond geometries, but ignores solvation and
        # electrostatic effects.
        cmd.sculpt_activate("sculpt_object")
        cmd.set("sculpt_field_mask", 255) # Turning on all possible sculpting parameters
        cmd.sculpt_iterate("sculpt_object", cycles=2000)
        cmd.sculpt_deactivate("sculpt_object")
        cmd.create("tri_0", "chain a+b+c")
        next(supplement_steps) #* Step 6, creating tri_1_nonsymm for T4.
        log.info(f"{trimer_name[:4]} Rotation step {rot_step} complete, continuing.")

    next(supplement_steps) #* Step 7, preparing T4 for capsid expansion.
    morphed_trimer_file = f"{trimer_name[:4]}_T{t_num}_z{interface_rot_origin[2]:.0f}_morphed.cif"
    cmd.save(morphed_trimer_file)
    log.info(f"Morphing to {morphed_trimer_file} complete, continuing.")
    return morphed_trimer_file


def t3_supplement(rot_steps, cutoff):
    """Supplemental generator function for the create_tx_interface function,
    see its docstring for more documentation.
    """
    # The starting angle should be the icosahedron dihedral angle (~138.19). A T3 capsid can be
    # though of as a dodecahedron in which every face was replaced by a regular pentagonal pyramid.
    # As such, the angle of the new interface is the dodecahedron dihedral angle (~116.565) plus two
    # times the angle between the base of a regular pentagonal pyramid and its flat sides (~37.377).
    yield (ICO_DIHED - (DODEC_DIHED + 2 * PENT_PYR_ANG)) / (2 * rot_steps) # Yields rot_angle
    yield None # Step 2.
    for rot_step in range(1, rot_steps + 1):
        yield None # Step 3.
        yield pymol_rotate(angle=180, axis="z", origin=[0, 0, 0], selection="tri_1") # Yields None
        mask = [f"tri_1 within {cutoff} of tri_0"]
        # The mask includes more (free-to-move) atoms after the final rotation step.
        mask.append(f"tri_0 within {cutoff} of tri_1" if rot_step < rot_steps else "tri_0")
        yield mask # End of step 5.
        yield None # Step 6.
    yield None # Step 7.


def t4_supplement(rot_steps, cutoff):
    """Supplemental generator function for the create_tx_interface function,
    see its docstring for more documentation.
    """
    # The starting angle should be the icosahedron dihedral angle (~138.19). A T4 capsid is just a
    # sub-divided icosahedron, so the angle of the new interface is 180 (i.e. flat).
    yield (ICO_DIHED - 180) / (2 * rot_steps) # Step 1, yields rot_angle.

    # Creating tri_1.
    cmd.copy("tri_1_nonsymm", "tri_0")
    pymol_rotate(angle=180, axis="z", origin=[0, 0, 0], selection="tri_1_nonsymm")
    # Finding the monomer within tri_1 that has the most contacts with tri_0.
    contacts = {"a": None, "b": None, "c": None}
    for chain in contacts:
        cmd.select(f"(tri_1_nonsymm & chain {chain}) within 5 of tri_0")
        contacts[chain] = cmd.count_atoms(r"%sele")
    sorted_contacts = sorted(contacts, key=contacts.get, reverse=True) # Sorted in descending order.
    # Renaming the chains, making sure that the max contact chain is called "d".
    for old_name, new_name in zip(sorted_contacts, "def"):
        cmd.alter(f"tri_1_nonsymm & chain {old_name}", f"chain='{new_name}'")
    yield None # End of step 2.

    for rot_step in range(1, rot_steps + 1):
        # Symmetrising tri_1.
        for chain in "def":
            cmd.create(f"copy_{chain}", "tri_1_nonsymm & chain d")
            cmd.align(f"copy_{chain}", f"tri_1_nonsymm & chain {chain}")
            cmd.alter(f"copy_{chain}", f"chain='{chain}'")
        cmd.create("tri_1", "copy_d copy_e copy_f")
        cmd.delete("tri_1_nonsymm copy_d copy_e copy_f")
        # End of step 3 (yield None) and beginning of step 4 (accepting external variable values).
        [shifted_origin, superpent_axis] = yield None

        # Rotating tri_4 & tri_5 into correct positions.
        flip_y = lambda coords: np.array(coords) * [1, -1, 1]
        np_tri_1 = np.array(cmd.centerofmass("tri_1")) # 1D arrays, instead of the usual 2D
        three_fold_axis = (np_tri_1 - flip_y(shifted_origin)).tolist()
        flipped_origin = flip_y(shifted_origin).tolist()
        pymol_rotate(angle=120, axis=three_fold_axis, origin=flipped_origin, selection="tri_4")
        pymol_rotate(angle=-120, axis=three_fold_axis, origin=flipped_origin, selection="tri_5")
        yield None # End of step 4.

        mask = ["tri_0", "(tri_1 within 10 of chain d)",
               f"tri_1 within {cutoff} of tri_0", f"(tri_4 tri_5) within {cutoff} of chain d"]
        # The mask includes more (free-to-move) atoms after the final rotation step.
        if rot_step < rot_steps:
            mask[0] += f" within {cutoff} of tri_1"
            mask[1] += f" & (tri_1 within {cutoff} of (tri_4 tri_5))"
        yield mask # End of step 5.

        yield cmd.create("tri_1_nonsymm", "chain d+e+f") # Step 6, yields None.

    cmd.pseudoatom("3fold_symm", pos=np_tri_1.tolist())
    pymol_rotate(angle=36, axis=superpent_axis, origin=shifted_origin, selection="(all)")
    yield None # End of step 7.


def expand_capsid(morphed_trimer_file):
    """This function extracts the newly designed T3/T4 asymmetric unit from a
    file, translates it away from origin to obtain the correct capsid radius,
    and copies it 60 times to create the expanded T3/T4 capsid, which is saved
    as a new .cif file.

    Args:
        morphed_trimer_file (str): name of the file containing a trimer that
            has been morphed to create the new T3 or T4 interface

    Returns:
        bool: a statement whether the function has been ran successfully
    """
    if not isfile(morphed_trimer_file):
        log.error(f"No {morphed_trimer_file} file has been found, stopping.")
        return False
    cmd.delete("all")
    cmd.load(morphed_trimer_file)
    try:
        t_num = int(match(r"...._T(\d).*_morphed", morphed_trimer_file).group(1))
    except AttributeError:
        log.error(f"Couldn't parse file {morphed_trimer_file} for t_num variable, stopping.")
        return False
    # Establishing the asymmetric unit.
    asymm_unit = "a+b+c+d 3fold_symm" if t_num == 4 else "a+b+c"
    cmd.create("asymm", f"sculpt_object & chain {asymm_unit}")

    # Moving the asymmetric unit away from origin to increase the capsid's radius.
    np_t1_vertex = cmd.get_coords("vertex_0-p1")
    np_shifted_origin = cmd.get_coords("xyz_origin")
    np_combined_coords = np.concatenate((np_t1_vertex, np_shifted_origin))
    # Ignoring the X coordinate values here, as they should both be 0.
    y_coords, z_coords = (np_combined_coords[:, 1], np_combined_coords[:, 2])
    z_intercept = linregress(y_coords, z_coords)[1]
    # The correct translation distance is the one which makes the five-fold axis of symmetry pass
    # through [0, 0, 0].
    cmd.translate([0, 0, -z_intercept], "(all)", camera=0)
    cmd.delete("not asymm & not vertex_0-p1")

    # Placing the first vertex & asymmetric unit on the very top of the capsid (and the z axis).
    norm_t34_vertex = normalise(cmd.get_coords("vertex_0-p1"))
    z_axis = np.array([0, 0, 1])
    angle_from_z = degrees(np.arccos(np.dot(norm_t34_vertex, z_axis)))
    pymol_rotate(selection="(all)", origin=[0, 0, 0], angle=angle_from_z * -1, axis="x")
    pymol_rotate(selection="(all)", origin=[0, 0, 0], angle=180, axis="z")

    if t_num == 4: # Moving the asymmetric unit even farther away from origin to get correct radius.
        adjustment_steps = t4_distance_adjustment()
        next(adjustment_steps)

    # Creating the expanded capsid by repeatedly copying & rotating the asymmetric unit.
    failure = create_corners(ICO_SPH_COORDS)
    if isinstance(failure, int):
        log.error(f"{morphed_trimer_file} Couldn't make corner {failure}, stopping.")
        return False

    if t_num == 4: # Checking if the T4 capsid radius is correct.
        if next(adjustment_steps) is False:
            log.error(f"{morphed_trimer_file} translation distance was incorrect, stopping.")
            return False

    cmd.delete("vertex* asymm")
    cmd.save(morphed_trimer_file.replace("morphed", "capsid"))
    log.info(f"{morphed_trimer_file} Capsid expansion complete, continuing.")
    return True


def t4_distance_adjustment():
    """This supplemental generator function modifies the expand_capsid function
    to allow the latter to create T4 capsids with the correct radius. Due to
    the use of yield statements, this function is executed in two parts.

    Yields:
        None or bool: first None, as there is nothing to return. Then a bool,
            which states whether the function has been ran successfully.
    """
    # Making two adjacent expanded capsid corners.
    create_corners(ICO_SPH_COORDS[:2])
    # Lines made by points a1-a2 and b1-b2 show the travel paths of the 3fold_symm pseudoatoms
    # during outward translation of capsid corners. The correct translation distance is the one
    # which places both pseudoatoms in the same location, i.e. the intercept of these two lines.
    a1 = cmd.get_coords("asymm_corner_0_288 & hetatm").flatten()
    b1 = cmd.get_coords("asymm_corner_1_0 & hetatm").flatten()
    a2 = a1 + cmd.get_coords("vertex_0").flatten()
    b2 = b1 + cmd.get_coords("vertex_1").flatten()
    # Solved using Cramer's rule, see rosettacode.org/wiki/Find_the_intersection_of_two_lines#Python
    # Y coordinate values are [1], z are [2], and x are being ignored.
    denominator = (b2 - b1)[2] * (a2 - a1)[1] - (b2 - b1)[1] * (a2 - a1)[2]
    nominator = (b2 - b1)[1] * (a1 - b1)[2] - (b2 - b1)[2] * (a1 - b1)[1]
    z_translation = (a2 - a1)[2] * nominator / denominator
    # Adjusting the distance of the original asymmetric unit from the capsid's center.
    cmd.translate([0, 0, z_translation], "asymm vertex_0-p1", camera=0)
    cmd.delete("asymm_corner_* vertex_0 vertex_1")
    yield None

    # Checking if the translation distances were correct, once all 12 corners have been created.
    np_3fold_coords = cmd.get_coords("asymm_corner_* & name PS1")
    np_3fold_distances = cdist(np_3fold_coords, np_3fold_coords)
    # Inspecting the distance of each 3fold_symm pseudoatom from its two nearest neighbors.
    if np.sort(np_3fold_distances, axis=0)[1:3].max() > 0.4: # 0.4 Angstrom leeway
        yield False
    else:
        cmd.remove("asymm_corner* & name PS1") # Removing the 3fold_symm pseudoatoms
        yield True


def create_corners(spherical_coords):
    """This utility function creates copies of the T3/T4 asymmetric unit and
    rotates them into positions specified by a list of spherical coordinates.
    Each coordinate corresponds to one corner of an icosahedron, and 12 corners
    make an entire capsid, regardless of its T number. Note that a suitable
    PyMol file must already be be loaded for this function to work.

    Args:
        spherical_coords (list): list of lists, containing 12 pairs of an angle
            + an axis around which to rotate in order to reach each capsid
            corner. Partial capsid structures can be constructed by providing
            fewer than 12 pairs.

    Returns:
        int or None: number of the first corner that couldn't be constructed
            correctly, or None if all corners were succesfully built.
    """
    corner_numbers = iter(range(12))
    for angle, axis in spherical_coords:
        corner_num = next(corner_numbers)
        # Placing one asymmetric unit at the new vertex.
        asymm_units = [f"asymm_corner_{corner_num}_0"]
        cmd.copy(asymm_units[0], "asymm")
        cmd.copy(f"vertex_{corner_num}", "vertex_0-p1")
        pymol_rotate(selection=f"{asymm_units[0]} vertex_{corner_num}", origin=[0, 0, 0],
                     angle=angle, axis=axis)

        # Filling in the remaining 4 asymmetric units.
        np_vertex = cmd.get_coords(f"vertex_{corner_num}")
        for pent_angle in [72, 144, 216, 288]:
            asymm_units.append(f"asymm_corner_{corner_num}_{pent_angle}")
            cmd.copy(asymm_units[-1], asymm_units[0]) # Copying the first asymmetric_unit
            pymol_rotate(selection=asymm_units[-1], origin=[0, 0, 0], angle=pent_angle,
                         axis=np_vertex.tolist()[0])

        # Checking that the predicted vertex location and the corner's actual center of mass match.
        # Because of how the vertex location was previously calculated, the corner's center of
        # mass must now also be obtained based solely on asymmetric unit chains a+b+c (i.e. not d).
        np_corner = centerofmass_2D("(" + " ".join(asymm_units) + ") & chain a+b+c")
        if np.abs(np_corner - np_vertex).max() > 3: # 3 Angstrom tolerance
            return corner_num
    return None


def logging_setup(path="capsid_converter_logs.log"):
    """This function establishes logging of messages, both to the python
    console and a dedicated log file.

    Args:
        path (str, optional): Path to the file where the logs should be
            recorded. Defaults to "capsid_converter_logs.log", saved in the
            same folder as the T1_to_T4_capsid_converter.py script.

    Returns:
        object: exposes the logging object to other functions by returning it
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(path, mode="a")
    file_handler.setLevel(logging.ERROR)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def from_command_line():
    """A wrapper function needed to be able to pass arguments to the
    run_pipeline function from the command line interface."""
    class CustomFormatter(argparse.HelpFormatter): #pylint: disable=missing-class-docstring
        # Overriding the default _fill_text method.
        def _fill_text(self, text, width, indent): #pylint: disable=unused-argument
            return "".join(indent + line for line in text.splitlines(keepends=True))
    def str_list(string):
        string = sub(r"[\[\]\(\)\{\}'\"\s]", "", string)
        # Works even for a single pdb without brackets.
        return string.split(",")
    def int_list(string):
        string = sub(r"[\[\]\(\)\{\}'\"\s]", "", string)
        return [int(x) for x in string.split(",")]

    parser = argparse.ArgumentParser(description=run_pipeline.__doc__,
                                     formatter_class=CustomFormatter)
    parser.add_argument("pdb_list", type=str_list, help="Use commas and no spaces")
    parser.add_argument("t_nums", type=int_list, help="Use commas and no spaces")
    parser.add_argument("--mode", type=int, metavar="")
    parser.add_argument("--vary_radius", type=int_list, metavar="", help="Use commas and no spaces")
    args = parser.parse_args()
    kwargs = {}
    if args.mode is not None:
        kwargs["mode"] = args.mode
    if args.vary_radius is not None:
        kwargs["vary_radius"] = args.vary_radius

    # If the user is requesting help, there is no need to run the rest of the script nor open PyMol.
    if not getattr(args, "help", False):
        finish_launching(["pymol", "-qx"]) # Quiet, no external GUI
        run_pipeline(args.pdb_list, args.t_nums, **kwargs)


def run_pipeline(pdb_list, t_nums, mode=1, vary_radius=(0,1,1)):
    """Top-level function that users should be interacting with.

    Args:
        pdb_list (list): list of strings, specifying the 4-character pdb codes
            of T1 viral capsids with 60 subunits.
        t_nums (list): list of integers, containing either 3, 4, or both. These
            are the triangulation numbers to which the provided T1 capsids will
            be expanded.
        mode (int): whether to run the entire pipeline (1), skip the initial
            stages of the process and jump straight to interface morphing (2),
            or create an expansion trimer file and then stop (3). Mode 2 will
            only work if you have already prepared an expansion trimer file.
            Defaults to 1.
        vary_radius (tuple): three-member tuple specifying the inputs for a
            range function that will vary the position of the interface origin
            of rotation during morphing. By default no variation will be added.
    """
    # Confirming correct user inputs
    mode = int(mode)
    if mode not in [1, 2, 3]:
        log.error(f"Incorrect mode '{mode}', stopping.")
        return False
    if not set(t_nums).issubset(set([3, 4])):
        log.error(f"Incorrect t_nums '{t_nums}', stopping.")
        return False
    try:
        [_ for _ in range(*vary_radius)]
    except TypeError:
        log.error(f"Incorrect vary_radius '{vary_radius}', stopping.")
        return False
    multimer_type = "trimer"

    # If a single pdb string is provided, it has to be converted to a one-member list - otherwise
    # the for loop will iterate over the individual characters in the string.
    for pdb in pdb_list if not isinstance(pdb_list, str) else [pdb_list]:
        log.critical(f"Starting work on {pdb}")
        if mode in [1, 3]:
            prepare_pdb(pdb, 1)
            group_monomers(pdb, multimer_type)
            typical_multimer = find_typical_multimer(pdb, multimer_type)
            if not typical_multimer:
                continue # Despite how it sounds, this makes the function move onto the next pdb.
            two_superpents = find_two_superpentamers(typical_multimer)
            if not two_superpents:
                continue
            oriented = orient_capsid(two_superpents, typical_multimer)
            if not oriented:
                continue
            expansion_trimer_file = prepare_for_interface_morph(pdb, multimer_type)
            if not expansion_trimer_file:
                continue
            # If you want the trimer to be e.g. clockwise with chain 'a' furthest from the new
            # T3/T4 interface, manually edit the expansion trimer file at this point.
            if mode == 3:
                log.critical(f" {pdb} succeeded in making an expansion trimer file.")
                continue
        else:
            expansion_trimer_file = f"{pdb.lower()}_for_morphing.cif"
            if not isfile(expansion_trimer_file):
                log.error(f"No {expansion_trimer_file} file has been found, stopping.")
                continue

        for t_num in list(t_nums):
            for radius_shift in range(*vary_radius):
                cmd.delete("all")
                cmd.load(expansion_trimer_file)
                cmd.translate([0, 0, radius_shift], "interface_rot_origin", camera=0)
                morphed_trimer_file = create_tx_interface(t_num)
                result = expand_capsid(morphed_trimer_file)
                if result:
                    log.critical(f" {morphed_trimer_file[:-12]} succeeded in making a capsid with a"
                                 f" radius shift of {radius_shift}")
    log.info("Finished all work")

####################################################################################################

# Numpy arrays require tolist() to be accepted by PyMol cmd functions, but not by other functions.
# Round brackets are extremely important when writing PyMol selections.

if __name__ == "__main__":
    # This sets the working directory (where to all files will be saved and where from they will be
    # loaded) to the same directory that contains the T1_capsid_expander.py script. Note that
    # __file__ is sometimes (but not always) the full path name.
    chdir(dirname(abspath(__file__)))
    # Turning on the logging.
    log = logging_setup()

    if len(argv) > 1:
        from_command_line()
    else:
        # Without this line the script still runs fine, but the PyMol window doesn't open, hence png
        # images are recorded differently.
        finish_launching(["pymol", "-qx"]) # Quiet, no external GUI
        run_pipeline(["4Y5Z"], [3, 4]) # Example T1 viral structure, being expanded to both T3 & T4
