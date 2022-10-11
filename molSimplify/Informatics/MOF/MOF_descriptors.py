import os
import re
import copy
from scipy import sparse
import numpy as np
import pandas as pd
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Scripts.cellbuilder_tools import import_from_cif
from molSimplify.Informatics.RACassemble import append_descriptors
from molSimplify.Informatics.autocorrelation import (
    generate_atomonly_autocorrelations, 
    generate_atomonly_deltametrics, 
    generate_full_complex_autocorrelations, 
    generate_multimetal_autocorrelations, 
    generate_multimetal_deltametrics,
    full_autocorrelation, 
    )
from molSimplify.Informatics.MOF.PBC_functions import (
    compute_adj_matrix, 
    compute_distance_matrix3, 
    fractional2cart, 
    get_closed_subgraph, 
    include_extra_shells, 
    ligand_detect, 
    linker_length, 
    mkcell, 
    readcif, 
    slice_mat, 
    write2file, 
    writeXYZandGraph, 
    XYZ_connected,
    )


#### NOTE: In addition to molSimplify's dependencies, this portion requires
#### pymatgen to be installed. The RACs are intended to be computed
#### on the primitive cell of the material. You can compute them
#### using the commented out snippet of code if necessary.

# Example usage is given at the bottom of the script.

'''<<<< CODE TO COMPUTE PRIMITIVE UNIT CELLS >>>>'''
#########################################################################################
# This MOF RAC generator assumes that pymatgen is installed.                            #
# Pymatgen is used to get the primitive cell.                                           #
#########################################################################################

### Defining pymatgen function for getting primitive, since molSimplify does not depend on pymatgen.
from pymatgen.io.cif import CifParser

def get_primitive(datapath, writepath, occupancy_tolerance=1):
    """
    Calculates and writes the primitive cell of the provided structure.

    Parameters
    ----------
    datapath : str
        The path to the cif file for which the primitive cell will be calculated.
    writepath : str
        The path to where the cif of the primitive cell will be written.
    occupancy_tolerance : float
        Scales down occupancies for a site to one, if they sum to below occupancy_tolerance.

    Returns
    -------
    None

    """

    s = CifParser(datapath, occupancy_tolerance=occupancy_tolerance).get_structures()[0]
    sprim = s.get_primitive_structure()
    sprim.to("cif", writepath) # Output structure to a file.

'''<<<< END OF CODE TO COMPUTE PRIMITIVE UNIT CELLS >>>>'''

def load_sbu_lc_descriptors(sbupath):
    """
    Loads the sbu and lc descriptors.

    Parameters
    ----------
    sbupath : str
        Path of the folder to make a csv file in.  

    Returns
    -------
    sbu_descriptors : pandas.core.frame.DataFrame
        The existing SBU descriptors.
    lc_descriptors : pandas.core.frame.DataFrame
        The existing lc descriptors.

    """
    sbu_descriptor_path = os.path.dirname(sbupath)
    if os.path.getsize(sbu_descriptor_path+'/sbu_descriptors.csv')>0: # Checking if there is a file there.
        sbu_descriptors = pd.read_csv(sbu_descriptor_path+'/sbu_descriptors.csv')
    else:
        sbu_descriptors = pd.DataFrame()
    if os.path.getsize(sbu_descriptor_path+'/lc_descriptors.csv')>0: # Checking if there is a file there.
        lc_descriptors = pd.read_csv(sbu_descriptor_path+'/lc_descriptors.csv')
    else:
        lc_descriptors = pd.DataFrame()

    return sbu_descriptor_path, sbu_descriptors, lc_descriptors

def gen_and_append_desc(temp_mol, target_list, depth, descriptor_names, descriptors, feature_type):
    """
    Generate and append descriptors, both standard and delta.

    Parameters
    ----------
    temp_mol : molSimplify.Classes.mol3D.mol3D
        mol3D object of the linker of interest. 
    target_list : list
        The indices of the atoms of interest in the linker.
    depth : int
        The depth of the RACs that are generated. See https://doi.org/10.1021/acs.jpca.7b08750 for more information.         
    descriptor_names : list
        The RAC descriptor names. Will be appended to.    
    descriptors : list
        The RAC descriptor values. Will be appended to.
    feature_type : str
        Either 'lc' or 'func'.
      
    Returns
    -------
    results_dictionary : dict
        deltametrics RACs
    descriptor_names : list
        The updated RAC descriptor names.
    descriptors : list
        The updated RAC descriptor values. 
    """
    
    results_dictionary = generate_atomonly_autocorrelations(temp_mol, target_list, depth=depth, loud=False, oct=False, polarizability=True)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors, results_dictionary['colnames'], results_dictionary['results'], feature_type, 'all')

    results_dictionary = generate_atomonly_deltametrics(temp_mol, target_list, depth=depth, loud=False, oct=False, polarizability=True)
    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors, results_dictionary['colnames'], results_dictionary['results'], f'D_{feature_type}', 'all')

    return results_dictionary, descriptor_names, descriptors

#########################################################################################
# The RAC functions here average over the different SBUs or linkers present. This is    #
# because one MOF could have multiple different linkers or multiple SBUs, and we need   #
# the vector to be of constant dimension so we can correlate the output property.       #
#########################################################################################

def make_MOF_SBU_RACs(SBUlist, SBU_subgraph, molcif, depth, name, cell, anchoring_atoms, sbupath, connections_list=False, connections_subgraphlist=False):
    """
    Generated RACs on the SBUs of the MOF, as well as on the lc atoms (SBU-connected atoms of linkers).

    Parameters
    ----------
    SBUlist : list of lists of numpy.int64
        Each inner list is its own separate SBU. The ints are the atom indices of that SBU. Length is # of SBUs.
    SBU_subgraph : list of scipy.sparse.csr.csr_matrix
        The atom connections in the SBU subgraph. Length is # of SBU.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    depth : int
        The depth of the RACs that are generated. See https://doi.org/10.1021/acs.jpca.7b08750 for more information.
    name : str
        The name of the cif being analyzed.
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    anchoring_atoms : set
        Linker atoms that are bonded to a metal.
    sbupath : str
        Path of the folder to make a csv file in.  
    connections_list : list of lists of int
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    connections_subgraphlist : list of scipy.sparse.csr.csr_matrix
        The atom connections in the linker subgraph. Length is # of linkers.        

    Returns
    -------
    names : list of str
        The names of the RAC SBU features.
    averaged_SBU_descriptors : numpy.ndarray
        The values of the RAC SBU features, averaged over all SBUs.
    lc_names : list of str
        The names of the RAC lc (ligand coordinating I think) features.
    averaged_lc_descriptors : numpy.ndarray
        The values of the RAC lc features, averaged over all lc atoms.

    """
    descriptor_list = []
    lc_descriptor_list = []
    lc_names = []
    names = []
    descriptor_names = []
    descriptors = []

    sbu_descriptor_path, sbu_descriptors, lc_descriptors = load_sbu_lc_descriptors(sbupath)

    """""""""
    Loop over all SBUs as identified by subgraphs. Then create the mol3Ds for each SBU.
    """""""""
    for i, SBU in enumerate(SBUlist):
        descriptor_names = []
        descriptors = []
        SBU_mol = mol3D()
        for val in SBU:
            SBU_mol.addAtom(molcif.getAtom(val))
        SBU_mol.graph = SBU_subgraph[i].todense()

        """""""""
        For each linker connected to the SBU, find the lc atoms for the lc-RACs.
        lc atoms are those bonded to a metal.
        """""""""
        for j, linker in enumerate(connections_list): # Iterating over the different linkers
            descriptor_names = []
            descriptors = []
            if len(set(SBU).intersection(linker))>0:
                #### This means that the SBU and the current linker are connected.
                temp_mol = mol3D()
                link_list = [] # Will hold the lc atoms for the current linker.
                for jj, val2 in enumerate(linker): # linker is a list of global atom indices for the atoms in that linker
                    if val2 in anchoring_atoms:
                        link_list.append(jj)
                    # This builds a mol object for the linker --> even though it is in the SBU section.
                    temp_mol.addAtom(molcif.getAtom(val2))

                temp_mol.graph = connections_subgraphlist[j].todense()
                """""""""
                Generate all of the lc autocorrelations (from the connecting atoms)
                """""""""
                results_dictionary, descriptor_names, descriptors = gen_and_append_desc(temp_mol, link_list, depth, descriptor_names, descriptors, 'lc')
                """""""""
                If heteroatom functional groups exist (anything that is not C or H, so methyl is missed, also excludes anything lc, so carboxylic metal-coordinating oxygens skipped),
                compile the list of atoms
                """""""""
                functional_atoms = []
                for jj in range(len(temp_mol.graph)):
                    if not jj in link_list: # linker atom is not bonded to a metal
                        if not set({temp_mol.atoms[jj].sym}) & set({"C","H"}): # not a carbon nor a hydrogen. syms get symbols.
                            functional_atoms.append(jj)
                """""""""
                Generate all of the functional group autocorrelations
                """""""""                
                if len(functional_atoms) > 0:
                    results_dictionary, descriptor_names, descriptors = gen_and_append_desc(temp_mol, functional_atoms, depth, descriptor_names, descriptors, 'func')
                else: # There are no functional atoms.
                    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,results_dictionary['colnames'],list([np.zeros(int(6*(depth + 1)))]),'func','all')
                    descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,results_dictionary['colnames'],list([np.zeros(int(6*(depth + 1)))]),'D_func','all')

                for val in descriptors:
                    if not (type(val) == float or isinstance(val, np.float64)):
                        print('Mixed typing. Please convert to python float, and avoid np float')
                        raise AssertionError('Mixed typing creates issues. Please convert your typing.')

                # Some formatting
                descriptor_names += ['name']
                descriptors += [name]
                desc_dict = {key2: descriptors[kk] for kk, key2 in enumerate(descriptor_names)}
                descriptors.remove(name)
                descriptor_names.remove('name')
                lc_descriptors = lc_descriptors.append(desc_dict, ignore_index=True)
                lc_descriptor_list.append(descriptors)
                if j == 0:
                    lc_names = descriptor_names
        
        averaged_lc_descriptors = np.mean(np.array(lc_descriptor_list), axis=0) # Average the lc RACs over all of the lc atoms in the MOF.
        lc_descriptors.to_csv(sbu_descriptor_path+'/lc_descriptors.csv',index=False)
        descriptors = []
        descriptor_names = []
        SBU_mol_cart_coords=np.array([atom.coords() for atom in SBU_mol.atoms])
        SBU_mol_atom_labels=[atom.sym for atom in SBU_mol.atoms]
        SBU_mol_adj_mat = np.array(SBU_mol.graph)

        ###### WRITE THE SBU MOL TO THE PLACE
        xyz_path = f'{sbupath}/{name}_sbu_{i}.xyz'
        if not os.path.exists(xyz_path):            
            SBU_mol_fcoords_connected = XYZ_connected(cell, SBU_mol_cart_coords, SBU_mol_adj_mat)
            writeXYZandGraph(xyz_path, SBU_mol_atom_labels, cell, SBU_mol_fcoords_connected,SBU_mol_adj_mat)
        """""""""
        Generate all of the SBU based RACs (full scope, mc)
        """""""""
        results_dictionary = generate_full_complex_autocorrelations(SBU_mol,depth=depth,loud=False,flag_name=False)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,results_dictionary['colnames'],results_dictionary['results'],'f','all')
        #### Now starts at every metal on the graph and autocorrelates
        results_dictionary = generate_multimetal_autocorrelations(molcif,depth=depth,loud=False)
        descriptor_names, descriptors =  append_descriptors(descriptor_names, descriptors, results_dictionary['colnames'],results_dictionary['results'],'mc','all')
        results_dictionary = generate_multimetal_deltametrics(molcif,depth=depth,loud=False)
        descriptor_names, descriptors = append_descriptors(descriptor_names, descriptors,results_dictionary['colnames'],results_dictionary['results'],'D_mc','all')

        # Some formatting
        descriptor_names += ['name']
        descriptors += [name]
        descriptors == list(descriptors)
        desc_dict = {key: descriptors[ii] for ii, key in enumerate(descriptor_names)}
        descriptors.remove(name)
        descriptor_names.remove('name')
        sbu_descriptors = sbu_descriptors.append(desc_dict, ignore_index=True)
        descriptor_list.append(descriptors)
        if i == 0:
            names = descriptor_names

    sbu_descriptors.to_csv(sbu_descriptor_path+'/sbu_descriptors.csv',index=False)
    averaged_SBU_descriptors = np.mean(np.array(descriptor_list), axis=0) # Average the SBU RACs over all of the SBUs in the MOF.
    return names, averaged_SBU_descriptors, lc_names, averaged_lc_descriptors

def make_MOF_linker_RACs(linkerlist, linker_subgraphlist, molcif, depth, name, cell, linkerpath):
    """
    Generate RACs on the linkers of the MOF.

    Parameters
    ----------
    linkerlist : list of lists of int
        Each inner list is its own separate linker. The ints are the atom indices of that linker. Length is # of linkers.
    linker_subgraphlist : list of scipy.sparse.csr.csr_matrix
        The atom connections in the linker subgraph. Length is # of linkers.
    molcif : molSimplify.Classes.mol3D.mol3D
        The cell of the cif file being analyzed.
    depth : int
        The depth of the RACs that are generated. See https://doi.org/10.1021/acs.jpca.7b08750 for more information.
    name : str
        The name of the cif being analyzed.
    cell : numpy.ndarray
        The three Cartesian vectors representing the edges of the crystal cell. Shape is (3,3).
    linkerpath : str
        Path of the folder to make a csv file in.    

    Returns
    -------
    colnames : list of str
        The names of the RAC linker features.
    averaged_ligand_descriptors : numpy.ndarray
        The values of the RAC linker features, averaged over all linkers.

    """

    #### This function makes full scope linker RACs for MOFs ####
    descriptor_list = []
    descriptors = []
    if linkerpath:
        linker_descriptor_path = os.path.dirname(linkerpath)
        if os.path.getsize(linker_descriptor_path+'/linker_descriptors.csv')>0: # Checking if there is a file there.
            linker_descriptors = pd.read_csv(linker_descriptor_path+'/linker_descriptors.csv')
        else:
            linker_descriptors = pd.DataFrame()
    for i, linker in enumerate(linkerlist): # Iterating through the linkers.
        # Preparing a mol3D object for the current linker.
        linker_mol = mol3D() 
        for val in linker:
            linker_mol.addAtom(molcif.getAtom(val))
        linker_mol.graph = linker_subgraphlist[i].todense()

        linker_mol_cart_coords=np.array([atom.coords() for atom in linker_mol.atoms])
        linker_mol_atom_labels=[atom.sym for atom in linker_mol.atoms]
        linker_mol_adj_mat = np.array(linker_mol.graph)

        ###### WRITE THE LINKER MOL TO THE PLACE
        xyz_path = f'{linkerpath}/{name}_linker_{i}.xyz'
        if not os.path.exists(xyz_path):
            linker_mol_fcoords_connected = XYZ_connected(cell, linker_mol_cart_coords, linker_mol_adj_mat)
            writeXYZandGraph(xyz_path, linker_mol_atom_labels, cell, linker_mol_fcoords_connected, linker_mol_adj_mat)
        allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
        labels_strings = ['chi', 'Z', 'I', 'T', 'S']
        colnames = []
        lig_full = list()

        """""""""
        Generate all of the linker based RACs
        """""""""
        for ii, properties in enumerate(allowed_strings):
            if not list(descriptors): # This is the case when just starting and the list is empty.
                ligand_ac_full = full_autocorrelation(linker_mol, properties, depth) # RACs
            else:
                ligand_ac_full += full_autocorrelation(linker_mol, properties, depth)
            this_colnames = []
            for j in range(0,depth+1):
                this_colnames.append('f-lig-'+labels_strings[ii] + '-' + str(j))
            colnames.append(this_colnames)
            lig_full.append(ligand_ac_full)
        
        # Some formatting
        lig_full = [item for sublist in lig_full for item in sublist] #flatten lists
        colnames = [item for sublist in colnames for item in sublist]
        colnames += ['name']
        lig_full += [name]
        desc_dict = {key: lig_full[i] for i, key in enumerate(colnames)}
        linker_descriptors = linker_descriptors.append(desc_dict, ignore_index=True)
        lig_full.remove(name)
        colnames.remove('name')
        descriptor_list.append(lig_full)

    #### We dump the standard lc descriptors without averaging or summing so that the user
    #### can make the modifications that they want. By default we take the average ones.
    linker_descriptors.to_csv(linker_descriptor_path+'/linker_descriptors.csv', index=False)
    averaged_ligand_descriptors = np.mean(np.array(descriptor_list), axis=0)
    return colnames, averaged_ligand_descriptors

def mkdir_if_absent(folder_paths):
    """
    Makes a folder at each path in folder_path if it does not yet exist.

    Parameters
    ----------
    folder_path : list of str
        The folder paths to check, and potentially at which to make a folder.

    Returns
    -------
    None

    """

    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

def make_file_if_absent(path, filenames):
    """
    Makes the specified files if they do not yet exist.

    Parameters
    ----------
    path : str
        The base path.
    filenames : list of str
        The file names.

    Returns
    -------
    None

    """

    for filename in filenames:
        if not os.path.exists(f'{path}/{filename}'):
            with open(f'{path}/{filename}','w') as f:
                f.close()

def path_maker(path):
    """
    Makes the required folders and files.

    Parameters
    ----------
    path : str
        The path to which output will be written. 

    Returns
    -------
    None

    """
    # Making the required folders
    required_folders = [f'{path}/ligands', f'{path}/linkers', f'{path}/sbus', f'{path}/xyz', f'{path}/logs']
    mkdir_if_absent(required_folders)
    
    # Making the required files
    required_files = ['sbu_descriptors.csv', 'linker_descriptors.csv', 'lc_descriptors.csv']
    make_file_if_absent(path, required_files)

def failure_response(path, failure_str):
    """
    Writes to the log file about the encountered failure.

    Parameters
    ----------
    path : str
        The path to which output will be written. 

    Returns
    -------
    Two zero lists to indicate failure.

    """
    full_names = [0]
    full_descriptors = [0]
    write2file(path,"/FailedStructures.log",failure_str)
    return full_names, full_descriptors

def get_MOF_descriptors(data, depth, path=False, xyzpath=False, graph_provided=False):
    """
    Generates RAC descriptors on a MOF.
    Writes three files: sbu_descriptors.csv, linker_descriptors.csv, and lc_descriptors.csv
    These files contain the RAC descriptors of the MOF.

    Parameters
    ----------
    data : str
        The path to the cif file for which descriptors are generated.
    depth : int
        The depth of the RACs that are generated. See https://doi.org/10.1021/acs.jpca.7b08750 for more information.
    path : str
        The path to which output will be written. 
        This output includes three csv files.
        This output also includes three folders called ligands, linkers, and sbus.
            These folders contain net and xyz files of the components of the MOF.
        This output also includes a folder called logs.
        This output also includes the xyz and net files for the cif being analyzed, written with the function writeXYZandGraph.
    xyzpath : str
        The path to where the xyz of the MOF structure will be written.
    graph_provided : bool
        Whether or not the cif file has graph information of the structure (i.e. what atoms are bonded to what atoms). 
        If not, computes the N^2 pairwise distance matrix, which is expensive.

    Returns
    -------
    full_names : list of str
        The names of the RAC features.
    full_descriptors : list of numpy.float64
        The values of the RAC features.

    """

    if not path: # Throw an error if the user did not supply a path to which to write the output.
        print('Need a directory to place all of the linker, SBU, and ligand objects. Exiting now.')
        raise ValueError('Base path must be specified in order to write descriptors.')
    else:
        if path.endswith('/'):
            path = path[:-1]

        # Making the required folders and files.
        path_maker(path)

    ligandpath = path+'/ligands'
    linkerpath = path+'/linkers'
    sbupath = path+'/sbus'
    logpath = path+"/logs"

    """""""""
    Input cif file and get the cell parameters and adjacency matrix. If too large or has overlap, do not featurize.
    Simultaneously prepare mol3D class for MOF for future RAC featurization (molcif).
    """""""""
    cpar, allatomtypes, fcoords = readcif(data)
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    name = os.path.basename(data).strip(".cif")
    if len(cart_coords) > 2000: # Don't deal with large cifs because of computational resources required for their treatment.
        print("Too large cif file, skipping it for now...")
        failure_str = f"Failed to featurize {name}: large primitive cell\n"
        full_names, full_descriptors = failure_response(path, failure_str)
        return full_names, full_descriptors

    """""""""
    Getting the adjacency matrix.
    """""""""
    if not graph_provided: # Make the adjacency matrix.
        distance_mat = compute_distance_matrix3(cell_v,cart_coords)
        try:
            adj_matrix=compute_adj_matrix(distance_mat,allatomtypes)
        except NotImplementedError:
            failure_str = f"Failed to featurize {name}: atomic overlap\n"
            full_names, full_descriptors = failure_response(path, failure_str)
            return full_names, full_descriptors
    else: # Grab the adjacency matrix from the cif file.
        adj_matrix_list = []
        max_sofar = 0
        with open(data.replace('primitive','cif'),'r') as f:
            readdata = f.readlines()
            flag = False
            for i, row in enumerate(readdata):
                if '_ccdc_geom_bond_type' in row:
                    flag = True
                    continue
                if flag:
                    splitrow = row.split()
                    atom1 = int(re.findall(r'\d+',splitrow[0])[0])
                    atom2 = int(re.findall(r'\d+',splitrow[1])[0])
                    max_sofar = max(atom1, max_sofar)
                    max_sofar = max(atom2, max_sofar)
                    adj_matrix_list.append((atom1,atom2))
        adj_matrix = np.zeros((max_sofar+1,max_sofar+1)) # 0 indicates the absence of a bond.
        for i, row in enumerate(adj_matrix_list):
            adj_matrix[row[0],row[1]] = 1 # 1 is indicative of a bond.
            adj_matrix[row[1],row[0]] = 1
    adj_matrix = sparse.csr_matrix(adj_matrix)

    writeXYZandGraph(xyzpath, allatomtypes, cell_v, fcoords, adj_matrix.todense())
    molcif,_,_,_,_ = import_from_cif(data, True) # molcif is a mol3D class of a single unit cell (or the cell of the cif file)
    molcif.graph = adj_matrix.todense()

    """""""""
    check number of connected components.
    if more than 1: it checks if the structure is interpenetrated. Fails if no metal in one of the connected components (identified by the graph).
    This includes floating solvent molecules.
    """""""""

    n_components, labels_components = sparse.csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    metal_list = set([at for at in molcif.findMetal(transition_metals_only=False)]) # the atom indices of the metals
    # print('##### METAL LIST', metal_list, [molcif.getAtom(val).symbol() for val in list(metal_list)])
    # print('##### METAL LIST', metal_list, [val.symbol() for val in molcif.atoms])
    if not len(metal_list) > 0:
        failure_str = f"Failed to featurize {name}: no metal found\n" 
        full_names, full_descriptors = failure_response(path, failure_str)
        return full_names, full_descriptors
    for comp in range(n_components):
        inds_in_comp = [i for i in range(len(labels_components)) if labels_components[i]==comp]
        if not set(inds_in_comp) & metal_list: # In the context of sets, & is the intersection. If the intersection is null, the (&) expression is False; the `not` would then make it True.
            # If this if statement is entered, there is an entire connected component that has no metals in it. No connections to any metal.
            failure_str = f"Failed to featurize {name}: solvent molecules\n"
            full_names, full_descriptors = failure_response(path, failure_str)
            return full_names, full_descriptors

    if n_components > 1 : # There are multiple connected components that have a metal in them in this case.
        print("structure is interpenetrated")
        tmpstr = "%s found to be an interpenetrated structure\n"%(name)
        write2file(logpath,"/%s.log"%name,tmpstr)

    """""""""
    step 1: metallic part
        removelist = metals (1) + atoms only connected to metals (2) + H connected to (1+2)
            Actually, it looks like only (1) and (2). Not H connected to (1+2)
        SBUlist = removelist + 1st coordination shell of the metals
    removelist = set()
    Logs the atom types of the connecting atoms to the metal in logpath.
    """""""""
    SBUlist = set()
    metal_list = set([at for at in molcif.findMetal(transition_metals_only=False)]) # the atom indices of the metals
    # print('##### METAL LIST2', metal_list, [molcif.getAtom(val).symbol() for val in list(metal_list)])
    # print('##### all LIST2', metal_list, [val.symbol() for val in molcif.atoms])
    [SBUlist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] # Consider all metals as part of the SBUs.
    [SBUlist.update(set(molcif.getBondedAtomsSmart(metal))) for metal in molcif.findMetal(transition_metals_only=False)] # atoms connected to metals. # TODO why use this over adj_matrix?
    removelist = set()
    [removelist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] # Remove all metals as part of the SBUs.
    for metal in removelist:
        bonded_atoms = set(molcif.getBondedAtomsSmart(metal))
        bonded_atoms_types = set([str(allatomtypes[at]) for at in set(molcif.getBondedAtomsSmart(metal))]) # The types of elements bonded to metals. E.g. oxygen, carbon, etc.
        cn = len(bonded_atoms)
        cn_atom = ",".join([at for at in bonded_atoms_types])
        tmpstr = "atom %i with type of %s found to have %i coordinates with atom types of %s\n"%(metal,allatomtypes[metal],cn,cn_atom)
        write2file(logpath,"/%s.log"%name,tmpstr)
    [removelist.update(set([atom])) for atom in SBUlist if all((molcif.getAtom(val).ismetal() or molcif.getAtom(val).symbol().upper() == 'H') for val in molcif.getBondedAtomsSmart(atom))]
    """""""""
    adding hydrogens connected to atoms which are only connected to metals. In particular interstitial OH, like in UiO SBU.
    """""""""
    for atom in SBUlist:
        for val in molcif.getBondedAtomsSmart(atom):
            if molcif.getAtom(val).symbol().upper() == 'H':
               removelist.update(set([val]))

    """""""""
    At this point:
    The remove list only removes metals and things ONLY connected to metals or hydrogens.
    Thus the coordinating atoms are double counted in the linker.

    step 2: organic part
        removelist = linkers are all atoms - the removelist (assuming no bond between
        organiclinkers)
    """""""""
    allatoms = set(range(0, adj_matrix.shape[0])) # The indices of all the atoms.
    linkers = allatoms - removelist
    linker_list, linker_subgraphlist = get_closed_subgraph(linkers.copy(), removelist.copy(), adj_matrix)
    connections_list = copy.deepcopy(linker_list)
    connections_subgraphlist = copy.deepcopy(linker_subgraphlist)
    linker_length_list = [len(linker_val) for linker_val in linker_list] # The number of atoms in each linker.
    """""""""
    find all anchoring atoms on linkers and ligands (lc identification)
        The atoms that are bonded to a metal.
    """""""""
    anc_atoms = set()
    for linker in linker_list: # Checking all of the linkers one by one.
        for atom_linker in linker: # Checking each atom in the current linker.
            bonded2atom = np.nonzero(adj_matrix[atom_linker,:])[1] # indices of atoms with bonds to the atom with the index atom_linker
            if set(bonded2atom) & metal_list: # This means one of the atoms bonded to the atom with the index atom_linker is a metal.
                anc_atoms.add(atom_linker)
    """""""""
    step 3: determine whether linker or ligand
    checking to find the anchors and #SBUs that are connected to an organic part
    anchor <= 1 -> ligand
    anchor > 1 and #SBU > 1 -> linker
    else: walk over the linker graph and count #crossing PBC
        if #crossing is odd -> linker
        else -> ligand
    """""""""
    initial_SBU_list, initial_SBU_subgraphlist = get_closed_subgraph(removelist.copy(), linkers.copy(), adj_matrix)
    templist = linker_list.copy()
    long_ligands = False
    max_min_linker_length , min_max_linker_length = (0,100) # The maximum value of the minimum linker length, and the minimum value of the maximum linker length. Updated later.
    for ii, atoms_list in reversed(list(enumerate(linker_list))): # Loop over all linker subgraphs
        linkeranchors_list = set()
        linkeranchors_atoms = set()
        sbuanchors_list = set()
        sbu_connect_list = set() # Will contain the indices of SBU subgraphs that have a connection to the current linker described by atoms_list
        """""""""
        Here, we are trying to identify what is actually a linker and what is a ligand.
        To do this, we check if something is connected to more than one SBU. Set to
        handle cases where primitive cell is small, ambiguous cases are recorded.
        """""""""
        for iii,atoms in enumerate(atoms_list): # loop over all atom indices in a linker
            connected_atoms = np.nonzero(adj_matrix[atoms,:])[1] # indices of atoms with bonds to the atom with the index atoms
            for kk, sbu_atoms_list in enumerate(initial_SBU_list): # loop over all SBU subgraphs
                for sbu_atoms in sbu_atoms_list: # Loop over SBU
                    if sbu_atoms in connected_atoms: # found an SBU atom bonded to an atom in the linker defined by atoms_list
                        linkeranchors_list.add(iii)
                        linkeranchors_atoms.add(atoms)
                        sbuanchors_list.add(sbu_atoms)
                        sbu_connect_list.add(kk) #Add if unique SBUs
        min_length,max_length = linker_length(linker_subgraphlist[ii].todense(), linkeranchors_list)

        if len(linkeranchors_list) >= 2 : # linker, and in one ambiguous case, could be a ligand.
            if len(sbu_connect_list) >= 2: # Something that connects two SBUs is certain to be a linker
                max_min_linker_length = max(min_length,max_min_linker_length)
                min_max_linker_length = min(max_length,min_max_linker_length)
                continue
            else:
                # check number of times we cross PBC :
                # TODO: we still can fail in multidentate ligands!
                linker_cart_coords = np.array([
                    at.coords() for at in [molcif.getAtom(val) for val in atoms_list]]) # Cartesian coordinates of the atoms in the linker
                linker_adjmat = np.array(linker_subgraphlist[ii].todense())
                pr_image_organic = ligand_detect(cell_v,linker_cart_coords,linker_adjmat,linkeranchors_list) # Periodic images for the organic component
                sbu_temp = linkeranchors_atoms.copy()
                sbu_temp.update({val for val in initial_SBU_list[list(sbu_connect_list)[0]]}) # Adding atoms. Not sure why the [0] is there? TODO 
                sbu_temp = list(sbu_temp)
                sbu_cart_coords = np.array([
                    at.coords() for at in [molcif.getAtom(val) for val in sbu_temp]])
                sbu_adjmat = slice_mat(adj_matrix.todense(),sbu_temp)
                pr_image_sbu = ligand_detect(cell_v,sbu_cart_coords,sbu_adjmat,set(range(len(linkeranchors_list)))) # Periodic images for the SBU
                if not (len(np.unique(pr_image_sbu, axis=0))==1 and len(np.unique(pr_image_organic, axis=0))==1): # linker. More than one periodic image for sbu or organic component
                    max_min_linker_length = max(min_length,max_min_linker_length)
                    min_max_linker_length = min(max_length,min_max_linker_length)
                    tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                            +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be linker\n'
                    write2file(ligandpath,"/ambiguous.txt",tmpstr)
                    continue
                else: #  all anchoring atoms are in the same unitcell -> ligand
                    removelist.update(set(templist[ii])) # we also want to remove these ligands
                    SBUlist.update(set(templist[ii])) # we also want to remove these SBUs
                    linker_list.pop(ii)
                    linker_subgraphlist.pop(ii)
                    tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                            +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be ligand\n'
                    write2file(ligandpath,"/ambiguous.txt",tmpstr)
                    tmpstr = str(name)+str(ii)+','+' Anchors list: '+ \
                            str(sbuanchors_list)+','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
                    write2file(ligandpath,"/ligand.txt",tmpstr)
        else: # definite ligand
            write2file(logpath,"/%s.log"%name,"found ligand\n")
            removelist.update(set(templist[ii])) # we also want to remove these ligands
            SBUlist.update(set(templist[ii])) # we also want to remove these ligands
            linker_list.pop(ii)
            linker_subgraphlist.pop(ii)
            tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
         +','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
            write2file(ligandpath,"/ligand.txt",tmpstr)

    tmpstr = str(name) + ", (min_max_linker_length,max_min_linker_length): " + \
                str(min_max_linker_length) + " , " +str(max_min_linker_length) + "\n"
    write2file(logpath,"/%s.log"%name,tmpstr)
    if min_max_linker_length < 3:
        write2file(linkerpath,"/short_ligands.txt",tmpstr)
    if min_max_linker_length > 2:
        # for N-C-C-N ligand ligand
        if max_min_linker_length == min_max_linker_length:
            long_ligands = True
        elif min_max_linker_length > 3:
            long_ligands = True

    """""""""
    In the case of long linkers, add second coordination shell without further checks. In the case of short linkers, start from metal
    and grow outwards using the include_extra_shells function
    """""""""
    linker_length_list = [len(linker_val) for linker_val in linker_list]
    if len(set(linker_length_list)) != 1:
        write2file(linkerpath,"/uneven.txt",str(name)+'\n') # Linkers are different lengths.
    if not min_max_linker_length < 2: # treating the 2 atom ligands differently! Need caution
        if long_ligands:
            tmpstr = "\nStructure has LONG ligand\n\n"
            write2file(logpath,"/%s.log"%name,tmpstr)
            # Expanding the number of atoms considered to be part of the SBU
            [[SBUlist.add(val) for val in  molcif.getBondedAtomsSmart(zero_first_shell)] for zero_first_shell in SBUlist.copy()] #First account for all of the carboxylic acid type linkers, add in the carbons.
        truncated_linkers = allatoms - SBUlist # Taking the difference of sets
        SBU_list, SBU_subgraphlist = get_closed_subgraph(SBUlist, truncated_linkers, adj_matrix)
        if not long_ligands:
            tmpstr = "\nStructure has SHORT ligand\n\n"
            write2file(logpath,"/%s.log"%name,tmpstr)
            SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list,SBU_subgraphlist,molcif,adj_matrix)
    else:
        tmpstr = "Structure %s has extremely short ligands, check the outputs\n"%name
        write2file(ligandpath,"/ambiguous.txt",tmpstr)
        tmpstr = "Structure has extremely short ligands\n"
        write2file(logpath,"/%s.log"%name,tmpstr)
        tmpstr = "Structure has extremely short ligands\n"
        write2file(logpath,"/%s.log"%name,tmpstr)
        truncated_linkers = allatoms - removelist
        SBU_list, SBU_subgraphlist = get_closed_subgraph(removelist, truncated_linkers, adj_matrix)
        SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list, SBU_subgraphlist, molcif, adj_matrix)
        SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list, SBU_subgraphlist, molcif, adj_matrix)

    """""""""
    For the cases that have a linker subgraph, do the featurization.
    """""""""
    if len(linker_subgraphlist)>=1: # Featurize cases that did not fail.
        # try:
            descriptor_names, descriptors, lc_descriptor_names, lc_descriptors = make_MOF_SBU_RACs(SBU_list, SBU_subgraphlist, molcif, depth, name, cell_v, anc_atoms, sbupath, connections_list, connections_subgraphlist)
            lig_descriptor_names, lig_descriptors = make_MOF_linker_RACs(linker_list, linker_subgraphlist, molcif, depth, name, cell_v, linkerpath)
            full_names = descriptor_names+lig_descriptor_names+lc_descriptor_names #+ ECFP_names
            full_descriptors = list(descriptors)+list(lig_descriptors)+list(lc_descriptors)
            print(len(full_names),len(full_descriptors))
        # except:
        #     full_names = [0]
        #     full_descriptors = [0]
    elif len(linker_subgraphlist) == 1: # this never happens, right?
        print('Suspicious featurization')
        full_names = [1]
        full_descriptors = [1]
    else:
        print('Failed to featurize this MOF.')
        full_names = [0]
        full_descriptors = [0]
    if (len(full_names) <= 1) and (len(full_descriptors) <= 1):
        tmpstr = "Failed to featurize %s\n"%(name)
        write2file(path,"/FailedStructures.log",tmpstr)
    return full_names, full_descriptors


##### Example of usage over a set of cif files.
# featurization_list = []
# import sys
# featurization_directory = sys.argv[1]
# for cif_file in os.listdir(featurization_directory+'/cif/'):
#     #### This first part gets the primitive cells ####
#     get_primitive(featurization_directory+'/cif/'+cif_file, featurization_directory+'/primitive/'+cif_file)
#     full_names, full_descriptors = get_MOF_descriptors(featurization_directory+'/primitive/'+cif_file,3,path=featurization_directory+'/',
#         xyzpath=featurization_directory+'/xyz/'+cif_file.replace('cif','xyz'))
#     full_names.append('filename')
#     full_descriptors.append(cif_file)
#     featurization = dict(zip(full_names, full_descriptors))
#     featurization_list.append(featurization)
# df = pd.DataFrame(featurization_list)
# df.to_csv('./full_featurization_frame.csv',index=False)
