#  @file protein3D.py
#  Defines protein3D class and contains useful manipulation/retrieval routines.
#
#  Written by HJK Group
#
#  Dpt of Chemical Engineering, MIT

# imports
from molSimplify.Classes.AA3D import AA3D
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D
from molSimplify.Classes.helpers import read_atom, makeMol
from molSimplify.Classes.globalvars import globalvars
import urllib.request
import urllib.error
import requests
from bs4 import BeautifulSoup
import pandas as pd
import subprocess
import shlex
import ast
import time
from scipy.spatial import ConvexHull
# from pymol import cmd, stored

# no GUI support for now


class protein3D:
    """Holds information about a protein, used to do manipulations.  Reads
    information from structure file (pdb, cif) or is directly built from
    molsimplify.

    """

    def __init__(self, pdbCode='undef'):
        # Number of amino acids
        self.naas = 0
        # Number of heteromolecules
        self.nhetmols = 0
        # Number of chains
        self.nchains = 0
        # Dictionary of amino acids
        self.aas = {}
        # Dictionary of all atoms
        self.atoms = {}
        # Dictionary of all atom indices
        self.a_ids = {}
        # Dictionary of heteromolecules
        self.hetmols = {}
        # Dictionary of chains
        self.chains = {}
        # Dictionary of missing atoms
        self.missing_atoms = {}
        # List of missing amino acids
        self.missing_aas = []
        # List of chain locations with more than one conformation
        self.conf = []
        # R value
        self.R = -1
        # Rfree value
        self.Rfree = -1
        # PDB code
        self.pdbCode = pdbCode
        # Holder for metals
        self.metals = False
        # Bonds
        self.bonds = {}
        # Data completeness
        self.DataCompleteness = 0
        # RSRZ value
        self.RSRZ = 100
        # TwinL score
        self.TwinL = 0
        # TwinL^2 score
        self.TwinL2 = 0
        # center of mass
        self.com = []
        # centroid
        self.centroid = []
        # convex hull
        self.hull = []

    def setAAs(self, aas):
        """ Set amino acids of a protein3D class to different amino acids.

        Parameters
        ----------
            aas : dictionary
                Keyed by chain and location
                Valued by AA3D amino acids
        """
        self.aas = aas
        self.naas = len(aas)

    def setAtoms(self, atoms):
        """ Set atom indices of a protein3D class to atoms.

        Parameters
        ----------
            atoms : dictionary
                Keyed by atom index
                Valued by atom3D atom that has that index
        """
        self.atoms = atoms

    def setIndices(self, a_ids):
        """ Set atom indices of a protein3D class to atoms.

        Parameters
        ----------
            a_ids : dictionary
                Keyed by atom3D atom
                Valued by its index
        """
        self.a_ids = a_ids

    def setHetmols(self, hetmols):
        """ Set heteromolecules of a protein3D class to different ones.

        Parameters
        ----------
            hetmols : dictionary
                Keyed by chain and location
                Valued by mol3D heteromolecules
        """
        self.hetmols = hetmols
        self.nhetmols = len(hetmols.keys())

    def setChains(self, chains):
        """ Set chains of a protein3D class to different chains.

        Parameters
        ----------
            chains : dictionary
                Keyed by desired chain IDs.
                Valued by the list of molecules in the chain.
        """
        self.chains = chains
        self.nchains = len(chains.keys())

    def setMissingAtoms(self, missing_atoms):
        """ Set missing atoms of a protein3D class to a new dictionary.

        Parameters
        ----------
            missing_atoms : dictionary
                Keyed by amino acid residues of origin
                Valued by missing atoms
        """
        self.missing_atoms = missing_atoms

    def setMissingAAs(self, missing_aas):
        """ Set missing amino acids of a protein3D class to a new list.

        Parameters
        ----------
            missing_aas : list
                List of missing amino acids.
        """
        self.missing_aas = missing_aas

    def setConf(self, conf):
        """ Set possible conformations of a protein3D class to a new list.

        Parameters
        ----------
            conf : list
                List of possible conformations for applicable amino acids.
        """
        self.conf = conf

    def autoChooseConf(self):
        """ Automatically choose the conformation of a protein3D class
        instance based first on what the greatest occupancy level is and then
        the first conformation ihe alphabet with all else equal.

        """
        for c in self.conf:
            c_ids = []
            if c in self.aas.keys():
                lst = self.aas[c]
            else:
                lst = self.hetmols[c]
            if len(lst) == 1:
                self.chains[c[0]].insert(c[1]-1, lst[0])
            else:
                for li in lst:
                    if li not in self.chains[c[0]]:
                        for j in li.atoms:
                            in_more_confs = False
                            for m in lst:
                                if m != li and j in m.atoms:
                                    in_more_confs = True
                            if type(j) != atom3D and not in_more_confs:
                                c_ids.append(j[0])
                            elif not in_more_confs:
                                c_ids.append(self.getIndex(j))
                        # print(c_ids)
                        self.stripAtoms(c_ids)
                        if type(li) == AA3D and li in self.aas[c]:
                            self.aas[c].remove(li)
                        elif type(li) == mol3D and li in self.hetmols[c]:
                            self.hetmols[c].remove(li)
        self.setConf([])

    def setR(self, R):
        """ Set R value of protein3D class.

        Parameters
        ----------
            R : float
                The desired new R value.
        """
        self.R = R

    def setRfree(self, Rfree):
        """ Set Rfree value of protein3D class.

        Parameters
        ----------
            Rfree : float
                The desired new Rfree value.
        """
        self.Rfree = Rfree

    def setRSRZ(self, RSRZ):
        """ Set RSRZ score of protein3D class.

        Parameters
        ----------
            RSRZ : float
                The desired new RSRZ score.
        """
        self.RSRZ = RSRZ

    def getMissingAtoms(self):
        """ Get missing atoms of a protein3D class.

        Example demonstration of this method:
        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7') # Fetch a PDB
        >>> for symbol_list in pdb_system.getMissingAtoms():
        >>>     for symbol in symbol_list:
        >>>         print(symbol.sym) # Prints the symbol of missing atom
        >>>         print(symbol.coords()) # Prints the coordinates of the missing atom - they are all the
        >>>                         # coordinates of origin by default (0.0,0.0,0.0) for missing atoms
        """
        return self.missing_atoms.values()

    def getMissingAAs(self):
        """ Get missing amino acid residues of a protein3D class.

        Example demonstration of this method:

        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7') # Fetch a PDB
        >>> pdb_system.getMissingAAs()   # This gives a list of AA3D objects
        >>> [pdb_system.getMissingAAs()[x].three_lc for x in range(len(val.getMissingAAs()))] # This returns
        >>>                     # the list of missing AAs by their 3-letter codes
        """
        return self.missing_aas

    def countAAs(self):
        """ Return the number of amino acid residues in a protein3D class.

        Example demonstration of this method:

        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7') # Fetch a PDB
        >>> pdb_system.countAAs() # This return the number of AAs in the PDB for all the chains.
        """
        return self.naas

    def findAtom(self, sym="X", aa=True):
        """
        Find atoms with a specific symbol that are contained in amino acids
        or heteromolecules.

        Parameters
        ----------
            sym : str
                element symbol, default as X.
            aa : boolean
                True if we want atoms contained in amino acids
                False if we want atoms contained in heteromolecules

        Returns
        ----------
            inds: list
                a list of atom indices with the specified symbol.

        Example demonstration of this method:
        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7') # Fetch a PDB
        >>> pdb_system.findAtom(sym="S", aa=True) # Returns indices of sulphur atoms present in amino acids
        >>> pdb_system.findAtom(sym="S", aa=False) # Returns indices of sulphur atoms present in heteromolecules
        """
        inds = []
        if aa:
            mols = self.aas.values()
        else:
            mols = self.hetmols.values()
        for s in mols:
            for m in s:
                for a in m.atoms:
                    if type(a) == tuple:
                        ii = a[0]
                        a = a[1]
                    else:
                        ii = self.getIndex(a)
                    if a.symbol() == sym:
                        inds.append(ii)
        return inds

    def findAA(self, three_lc="XAA"):
        """
        Find amino acids with a specific three-letter code.

        Parameters
        ----------
            three_lc: str
                three-letter code, default as XAA.

        Returns
        -------
            inds: set
                a set of amino acid indices with the specified symbol.

        Example demonstration of this method:
        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7') # Fetch a PDB
        >>> pdb_system.findAA(three_lc = 'MET') # Returns a set of pairs where each pair is a combination of the chain name
        >>>                              # and the index of the amino acid specified (in this case, 'MET')
        """
        inds = set()
        for aa in self.aas.values():
            if aa[0].three_lc == three_lc:
                inds.add((aa[0].chain, aa[0].id))
        return inds

    def getChain(self, chain_id):
        # BUGGY
        """ Takes a chain of interest and turns it into its own protein3D class instance.

        Parameters
        ----------
            chain_id : string
                The letter name of the chain of interest

        Returns
        -------
            p : protein3D
                A protein3D instance consisting of just the chain of interest

        Example demonstration of this method:
        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7') # Fetch a PDB
        >>> pdb_system.getChain('A') # Get chain A of the PDB
        """
        p = protein3D()
        p.setChains({chain_id: self.chains[chain_id]})
        p.setAAs(set(self.chains[chain_id]))
        p.setR(self.R)
        p.setRfree(self.Rfree)
        missing_aas = []
        for aa in self.missing_aas:
            if aa.chain == chain_id:
                missing_aas.append(aa)
        p.setMissingAAs(missing_aas)
        gone_atoms = {}
        for aa in self.missing_atoms.keys():
            if aa.chain == chain_id:
                gone_atoms[aa] = self.missing_atoms[aa]
        p.setMissingAtoms(gone_atoms)
        gone_hets = self.hetatms
        atoms = {}
        for a_id in self.atoms:
            aa = self.getResidue(a_id)
            if aa is not None:
                if aa.chain == chain_id:
                    atoms[a_id] = self.atoms[a_id]
            else:
                if chain_id not in gone_hets[(a_id, self.atoms[a_id])]:
                    del gone_hets[(a_id, self.atoms[a_id])]
                else:
                    atoms[a_id] = self.atoms[a_id]
        p.setHetatms(gone_hets)
        p.setAtoms(atoms)
        bonds = {}
        for a in self.bonds.keys():
            if a in p.atoms.values():
                bonds[a] = set()
                for b in self.bonds[a]:
                    if b in p.atoms.values():
                        bonds[a].add(b)
        p.setBonds(bonds)
        return p

    def getMolecule(self, a_id, aas_only=False):
        """ Finds the molecule that the atom is contained in.

        Parameters
        ----------
            a_id : int
                the index of the desired atom whose molecule we want to find
            aas_only : boolean
                True if we want ito find atoms contained in amino acids only.
                False if we want atoms contained in all molecules. Default is False.

        Returns
        -------
            mol : AA3D or mol3D
                the amino acid residue or heteromolecule containing the atom

        Example demonstration of this method:
        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7') # Fetch a PDB
        >>> pdb_system.getMolecule(a_id=2166) # This returns an molSimplify.Classes.AA3D.AA3D obejct indicating
        >>>                                   # we that the atom is part of an amino acid
        >>> pdb_system.getMolecule(a_id=2166).three_lc() # This prints the three letter code of the amino acid of which
        >>>                                              # atom 2166 is a part of
        >>> pdb_system.getMolecule(a_id=9164) # This returns a mol3D object indicating that the atom is part of a molecule
        >>>                                   # that is not an amino acid
        >>> pdb_system.getMolecule(a_id=9164).name # This prints the name of the molecule, in this case, it is 'TAU'
        """
        for s in self.aas.values():
            for mol in s:  # mol is AA3D
                if (a_id, self.atoms[a_id]) in mol.atoms:
                    return mol
        for mol in self.missing_atoms.keys():  # mol is incomplete AA3D
            if (a_id, self.atoms[a_id]) in self.missing_atoms[mol]:
                return mol
        if not aas_only:
            for s in self.hetmols.values():
                for mol in s:  # mol is mol3D
                    if self.atoms[a_id] in mol.atoms:
                        return mol
        return None  # something is wrong

    def stripAtoms(self, atoms_stripped):
        """ Removes certain atoms from the protein3D class instance.

        Parameters
        ----------
            atoms_stripped : list
                list of atom3D indices that should be removed

        Example demonstration of this method:
        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7') # Fetch a PDB
        >>> pdb_system.stripAtoms([2166, 4442, 6733, 2165]) # This removes the list of atoms with
        >>>                                                # indices listedin the code
        """
        atoms = self.atoms
        a_ids = self.a_ids
        keys = list(self.aas.keys()) + list(self.hetmols.keys())
        for tup in keys:
            if tup in self.aas.keys():
                mol_set = self.aas[tup].copy()
            else:
                mol_set = self.hetmols[tup].copy()
            for elt in mol_set:
                for a in elt.atoms:
                    if type(a) != atom3D:
                        atom = a[1]
                    else:
                        atom = a
                    if atom not in self.a_ids.keys():
                        continue
                    a_id = self.getIndex(atom)
                    if a_id in atoms_stripped:
                        if (a_id, atom) in elt.atoms:
                            elt.atoms.remove((a_id, atom))
                            if atom in elt.c:
                                elt.c.remove(atom)
                            elif atom in elt.n:
                                elt.n.remove(atom)
                        elif atom in elt.atoms:
                            elt.atoms.remove(atom)
                        atoms_stripped.remove(a_id)
                        if atom in self.bonds.keys():
                            for at in self.bonds[atom]:
                                if at in self.bonds.keys():
                                    temp = self.bonds[at].copy()
                                    if atom in temp:
                                        temp.remove(atom)
                                    self.bonds[at] = temp
                            del self.bonds[atom]
                        del atoms[a_id]
                        del a_ids[atom]
                if len(elt.atoms) == 0:
                    if tup in self.aas.keys():
                        self.aas[tup].remove(elt)
                        if len(self.aas[tup]) == 0:
                            del self.aas[tup]
                    else:
                        self.hetmols[tup].remove(elt)
                        if len(self.hetmols[tup]) == 0:
                            del self.hetmols[tup]
        while len(atoms_stripped) != 0:
            a_id = atoms_stripped[0]
            atoms_stripped.pop(0)
            if a_id not in atoms.keys():
                continue
            atom = atoms[a_id]
            if atom in self.bonds.keys():
                for at in self.bonds[atom]:
                    temp = self.bonds[at].copy()
                    if atom in temp:
                        temp.remove(atom)
                    self.bonds[at] = temp
                del self.bonds[atom]
            del atoms[a_id]
            del a_ids[atom]
        self.setAtoms(atoms)
        self.setIndices(a_ids)

    def stripHetMol(self, hetmol):
        """ Removes all heteroatoms part of the specified heteromolecule from
            the protein3D class instance.

        Parameters
        ----------
            hetmol : str
                String representing the name of a heteromolecule whose
                heteroatoms should be stripped from the protein3D class instance

        Example demonstration of this method:
        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7') # Fetch a PDB
        >>> pdb_system.stripHetMol()
        """
        hets = self.hetmols.copy()
        for k in hets.keys():
            if k not in self.hetmols.keys():
                continue
            for m in hets[k]:
                if m.name == hetmol:
                    ids = []
                    for a in m.atoms:
                        ids.append(self.a_ids[a])
                    self.stripAtoms(ids)
                    del self.hetmols[k]

    def findMetal(self, transition_metals_only=True):
        """Find metal(s) in a protein3D class.
        Parameters
        ----------
            transition_metals_only : bool, optional
                Only find transition metals. Default is true.

        Returns
        -------
            metal_list : list
                List of indices of metal atoms in protein3D.

        Example of fetching a PDB file:

        >>> pdb_system = protein3D()
        >>> pdb_system.fetch_pdb('1os7')
        """
        if not self.metals:
            metal_list = []
            for li in self.hetmols.values():  # no metals in AAs
                for m in li:
                    for a in m.atoms:
                        if a.ismetal(transition_metals_only=transition_metals_only):
                            if a.occup == 1 or a in self.bonds.keys():
                                metal_list.append(self.getIndex(a))
            self.metals = metal_list
        return (self.metals)

    def freezeatom(self, atomIdx):
        """Set the freeze attribute to be true for a given atom3D class.

        Parameters
        ----------
            atomIdx : int
                Index for atom to be frozen.
        """

        self.atoms[atomIdx].frozen = True

    def freezeatoms(self, Alist):
        """Set the freeze attribute to be true for a given set of atom3D classes,
        given their indices. Preserves ordering, starts from largest index.

        Parameters
        ----------
            Alist : list
                List of indices for atom3D instances to remove.
        """

        for h in sorted(Alist, reverse=True):
            self.freezeatom(h)

    def getAtom(self, idx):
        """Get atom with a given index.

        Parameters
        ----------
            idx : int
                Index of desired atom.

        Returns
        -------
            atom : atom3D
                atom3D class for element at given index.

        """
        return self.atoms[idx]

    def getIndex(self, atom):
        """ Get index of a given atom

        Parameters
        ----------
            atom : atom3D
                atom3D class for element at given index.

        Returns
        -------
            idx : int
                Index of desired atom.

        """
        if hasattr(self, 'a_ids') and atom in self.a_ids.keys():
            idx = self.a_ids[atom]
        else:
            print(atom.sym)
            idx = list(self.atoms.keys())[list(self.atoms.values()).index(atom)]
        return idx

    def getBoundMols(self, h_id, aas_only=False):
        """Get a list of molecules bound to a heteroatom, usually a metal.

        Parameters
        ----------
            h_id : int
                the index of the desired (hetero)atom origin
            aas_only : boolean
                whether or not to only consider amino acids, defaults False

        Returns
        -------
            bound_mols : list
                list of AA3D and/or mol3D instances of molecules bound to hetatm
        """
        bound_mols = []
        for b_id in self.atoms.keys():
            b = self.atoms[b_id]
            if self.atoms[h_id] not in self.bonds.keys():
                return None
            elif b in self.bonds[self.atoms[h_id]]:
                if self.getMolecule(b_id, aas_only) is not None:
                    bound_mols.append(self.getMolecule(b_id, aas_only))
        return bound_mols

    def readfrompdb(self, text):
        """ Read PDB into a protein3D class instance.

        Parameters
        ----------
            text : str
                String of path to PDB file. Path may be local or global.
                May also be the text of a PDB file from the internet.
        """

        # read in PDB file
        if '.pdb' in text:  # means this is a filename
            self.pdbfile = text
            fname = text.split('.pdb')[0]
            with open(fname + '.pdb', 'r') as f:
                text = f.read()
            enter = '\n'
        else:
            enter = "\\n"

        # class attributes
        aas = {}
        hetmols = {}
        atoms = {}
        a_ids = {}
        chains = {}
        missing_atoms = {}
        missing_aas = []
        conf = []
        bonds = {}

        # get R and Rfree values (text is full file)
        if "R VALUE            (WORKING SET)" in text:
            temp = text.split("R VALUE            (WORKING SET)")
            temp2 = temp[-1].split()
            if temp2[1] != 'NULL':
                R = float(temp2[1])
            else:
                R = -100
            if temp2[8] != 'NULL':
                Rfree = float(temp2[8])
            else:
                Rfree = 100
        elif "R VALUE          (WORKING SET, NO CUTOFF)" in text:
            temp = text.split("R VALUE          (WORKING SET, NO CUTOFF)")
            temp2 = temp[-1].split()
            if temp2[1] != 'NULL':
                R = float(temp2[1])
            else:
                R = -100
            if temp2[10] != 'NULL':
                Rfree = float(temp2[10])
            else:
                Rfree = 100
        else:
            R = -100
            Rfree = 100

        # start getting missing amino acids
        if "M RES C SSSEQI" in text:
            text = text.split("M RES C SSSEQI")
            want = text[-1]
            text = text[0].split(enter)
            split = text[-1]
            want = want.split(split)
            for line in want:
                if line == want[-1]:
                    text = line
                    line = line.split(enter)
                    line = line[0]
                    text = text.replace(line, '')
                sp = line.split()
                if len(sp) > 2:
                    a = AA3D(sp[0], sp[1], sp[2])
                    missing_aas.append(a)

        # start getting missing atoms
        if "M RES CSSEQI  ATOMS" in text:
            text = text.split("M RES CSSEQI  ATOMS")
            want = text[-1]
            text = text[0].split(enter)
            split = text[-1]
            want = want.split(split)
            for line in want:
                if line == want[-1]:
                    text = line
                    line = line.split(enter)
                    line = line[0]
                    text = text.replace(line, '')
                sp = line.split()
                if len(sp) > 2:
                    missing_atoms[(sp[1], sp[2])] = []
                    for atom in sp[3:]:
                        if atom != enter and atom[0] in ['C', 'N', 'O', 'H']:
                            missing_atoms[(sp[1], sp[2])].append(
                                atom3D(Sym=atom[0], greek=atom))
        # start getting amino acids and heteroatoms
        pa_dict = {'AltLoc': ""}
        if "ENDMDL" in text:
            text.split("ENDMDL")
            text = text[-2] + text[-1]
        text = text.split(enter)
        text = text[1:]
        for line in text:
            if line == text[-1]:
                text = line
                line = line.split(enter)
                line = line[0]
                text = text.replace(line, '')
            l_type = line[:6]
            if "ATOM" in l_type or "HETATM" in l_type:
                line = line.replace("\\'", "\'")
                a_dict = read_atom(line)
                if a_dict['ResName'] in globalvars().getAllAAs() or "ATOM" in l_type:
                    # have an amino acid or biomolecule monomer
                    a, aas, conf, chains, pa_dict, bonds = makeMol(a_dict, aas, conf, chains, pa_dict, bonds)
                else:  # have a normal heteromolecule
                    a, hetmols, conf, chains, pa_dict, bonds = makeMol(a_dict, hetmols, conf, chains, pa_dict, bonds, False)
                atoms[a_dict['SerialNum']] = a
                a_ids[a] = a_dict['SerialNum']

            elif "CONECT" in l_type:  # get extra connections
                line = line[6:]  # remove type
                li = [line[i:i+5] for i in range(0, len(line), 5)]
                if int(li[0]) in atoms.keys() and atoms[int(li[0])] not in bonds.keys():
                    bonds[atoms[int(li[0])]] = set()
                for i in li[1:]:
                    try:
                        bonds[atoms[int(li[0])]].add(atoms[int(i)])
                        if atoms[int(li[0])].loc != '':
                            for j in {1, -1}:
                                if atoms[int(li[0]) + j].greek == atoms[int(li[0])].greek:
                                    if atoms[int(li[0]) + j] not in bonds.keys():
                                        bonds[atoms[int(li[0]) + j]] = {atoms[int(i)]}
                                    else:
                                        bonds[atoms[int(li[0]) + j]].add(atoms[int(i)])
                                    if atoms[int(i)] not in bonds.keys():
                                        bonds[atoms[int(i)]] = {atoms[int(li[0]) + j]}
                                    else:
                                        bonds[atoms[int(i)]].add(atoms[int(li[0]) + j])
                    except ValueError:
                        # if "  " not in i and i != " ":
                        #    print("likely OXT")
                        continue
        # deal with conformations in chains
        for i in conf:
            if i in aas.keys():
                c = aas[i]
            else:
                c = hetmols[i]
            for j in range(len(c)):
                # pick chain with higher occupancy or the A chain if tie
                if type(c[j]) == mol3D:
                    for a in c[j].atoms:
                        full = True
                        if a.occup <= 1/len(c):
                            full = False
                    if full:
                        chains[i[0]].append(c[j])
                    elif c[j].atoms[0].occup*100 == 100//len(c) and j == 0:
                        chains[i[0]].append(c[j])
                elif c[j].occup > 1/len(c):
                    chains[i[0]].append(c[j])
                elif c[j].occup*100 == 100//len(c) and j == 0:
                    chains[i[0]].append(c[j])
        self.setChains(chains)
        self.setAAs(aas)
        self.setAtoms(atoms)
        self.setIndices(a_ids)
        self.setHetmols(hetmols)
        self.setMissingAtoms(missing_atoms)
        self.setMissingAAs(missing_aas)
        self.setConf(conf)
        self.setR(R)
        self.setRfree(Rfree)
        self.setBonds(bonds)

    def fetch_pdb(self, pdbCode):
        """ API query to fetch a pdb and write it as a protein3D class instance

        Parameters
        ----------
            pdbCode : str
                code for protein, e.g. 1os7
        """
        remoteCode = pdbCode.upper()
        try:
            data = urllib.request.urlopen(
                'https://files.rcsb.org/view/' + remoteCode +
                '.pdb').read()
        except urllib.error.URLError:
            print("warning: %s not found.\n" % pdbCode)
        else:
            try:
                self.readfrompdb(str(data))
                self.setPDBCode(pdbCode)
                print("fetched: %s" % (pdbCode))
            except IOError:
                print('aborted')
            else:
                if len(data) == 0:
                    print("warning: %s not valid.\n" % pdbCode)

    def setBonds(self, bonds):
        """Sets the bonded atoms in the protein.
        This is effectively the molecular graph.

        Parameters
        ----------
            bonds : dictionary
                Keyed by atom3D atoms in the protein
                Valued by a set consisting of bonded atoms
        """
        self.bonds = bonds

    def readMetaData(self):
        """ API query to fetch XML data from a pdb and add its useful attributes
        to a protein3D class.

        Parameters
        ----------
            pdbCode : str
                code for protein, e.g. 1os7
        """
        pdbCode = self.pdbCode
        try:
            start = 'https://files.rcsb.org/pub/pdb/validation_reports/' + pdbCode[1] + pdbCode[2]
            link = start + '/' + pdbCode + '/' + pdbCode + '_validation.xml'
            xml_doc = requests.get(link)
        except urllib.error.URLError:
            print("warning: %s not found.\n" % pdbCode)
        else:
            try:
                ### We then use beautiful soup to read the XML doc. LXML is an XML reader. The soup object is what we then use to parse!
                soup = BeautifulSoup(xml_doc.content, 'lxml-xml')

                ### We can then use methods of the soup object to find "tags" within the XML file. This is how we would extract sections.
                ### This is an example of getting everything with a "sec" tag.
                body = soup.find_all('wwPDB-validation-information')
                entry = body[0].find_all("Entry")
                if "DataCompleteness" not in entry[0].attrs.keys():
                    self.setDataCompleteness(0)
                    print("warning: %s has no DataCompleteness." % pdbCode)
                else:
                    self.setDataCompleteness(float(entry[0].attrs["DataCompleteness"]))
                if "percent-RSRZ-outliers" not in entry[0].attrs.keys():
                    self.setRSRZ(100)
                    print("warning: %s has no RSRZ.\n" % pdbCode)
                else:
                    self.setRSRZ(float(entry[0].attrs["percent-RSRZ-outliers"]))
                if "TwinL" not in entry[0].attrs.keys():
                    print("warning: %s has no TwinL." % pdbCode)
                    self.setTwinL(0)
                else:
                    self.setTwinL(float(entry[0].attrs["TwinL"]))
                if "TwinL2" not in entry[0].attrs.keys():
                    print("warning: %s has no TwinL2." % pdbCode)
                    self.setTwinL2(0)
                else:
                    self.setTwinL2(float(entry[0].attrs["TwinL2"]))
            except IOError:
                print('aborted')
            else:
                if xml_doc is None:
                    print("warning: %s not valid.\n" % pdbCode)

    def setDataCompleteness(self, DataCompleteness):
        """ Set DataCompleteness value of protein3D class.

        Parameters
        ----------
            DataCompleteness : float
                The desired new R value.
        """
        self.DataCompleteness = DataCompleteness

    def setTwinL(self, TwinL):
        """ Set TwinL score of protein3D class.

        Parameters
        ----------
            TwinL : float
                The desired new TwinL score.
        """
        self.TwinL = TwinL

    def setTwinL2(self, TwinL2):
        """ Set TwinL squared score of protein3D class.

        Parameters
        ----------
            TwinL2 : float
                The desired new TwinL squared score.
        """
        self.TwinL2 = TwinL2

    def setEDIAScores(self):
        """ Sets the EDIA score of a protein3D class.

        Parameters
        ----------
            pdbCode : string
                The 4-character code of the protein3D class.
        """
        code = self.pdbCode
        cmd = 'curl -d \'{"edia":{ "pdbCode":"'+code+'"}}\' -H "Accept: application/json" -H "Content-Type: application/json" -X POST https://proteins.plus/api/edia_rest'
        args = shlex.split(cmd)
        result = subprocess.Popen(args, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        result.wait()
        out, err = result.communicate()
        dict_str = out.decode("UTF-8")
        int_dict = ast.literal_eval(dict_str)
        res2 = subprocess.Popen(['curl', int_dict['location']],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out2, err2 = res2.communicate()
        dict2_str = out2.decode("UTF-8")
        dictionary = ast.literal_eval(dict2_str)
        t = 5  # can change depending on how frequently to loop
        while dictionary["status_code"] == 202:
            res2 = subprocess.Popen(['curl', int_dict['location']],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            # print('sleeping', t)
            time.sleep(t)
            res2.wait()
            out2, err2 = res2.communicate()
            dict2_str = out2.decode("UTF-8")
            dictionary = ast.literal_eval(dict2_str)
        link = dictionary["atom_scores"]
        df = pd.read_csv(link, error_bad_lines=False)
        for i, row in df.iterrows():
            EDIA = row["EDIA"]
            index = row["Infile id"]
            if index in self.atoms.keys():
                a = self.atoms[index]
                a.setEDIA(EDIA)
                if a.occup < 1:  # more than one conformation
                    subdf = df[df["Infile id"] == index+1]
                    if subdf.shape[0] == 0 and index+1 in self.atoms.keys():
                        self.atoms[index+1].setEDIA(EDIA)
                    elif subdf.shape[0] == 0 and index-1 in self.atoms.keys():
                        self.atoms[index-1].setEDIA(EDIA)
            else:
                print("OXT is missing")

    def setPDBCode(self, pdbCode):
        """ Sets the 4-letter PDB code of a protein3D class instance

        Parameters
        ----------
            pdbCode : string
                Desired 4-letter PDB code
        """
        self.pdbCode = pdbCode

    def centermass(self):
        """Computes coordinates of center of mass of protein.

        """

        center_of_mass = [0, 0, 0]  # coordinates of center of mass (X, Y, Z)
        mmass = 0
        # loop over atoms in molecule
        if len(self.atoms.keys()) > 0:
            for atom in self.atoms.values():
                # calculate center of mass (relative weight according to atomic mass)
                xyz = atom.coords()
                center_of_mass[0] += xyz[0] * atom.mass
                center_of_mass[1] += xyz[1] * atom.mass
                center_of_mass[2] += xyz[2] * atom.mass
                mmass += atom.mass
            # normalize
            center_of_mass[0] /= mmass
            center_of_mass[1] /= mmass
            center_of_mass[2] /= mmass
        else:
            center_of_mass = False
            print(
                'ERROR: Center of mass calculation failed. Structure will be inaccurate.\n')
        self.com = center_of_mass

    def setCentroid(self):
        """Computes coordinates of center of mass of protein.

        """

        centroid = [0, 0, 0]  # coordinates of centroid (X, Y, Z)
        # loop over atoms in protein
        if len(self.atoms.keys()) > 0:
            for atom in self.atoms.values():
                # calculate center of mass (relative weight according to atomic mass)
                xyz = atom.coords()
                centroid[0] += xyz[0]
                centroid[1] += xyz[1]
                centroid[2] += xyz[2]
            # normalize
            centroid[0] /= len(self.atoms.keys())
            centroid[1] /= len(self.atoms.keys())
            centroid[2] /= len(self.atoms.keys())
        else:
            centroid = False
            print(
                'ERROR: Centroid calculation failed. Structure will be inaccurate.\n')
        self.centroid = centroid

    def convexhull(self):
        """Computes convex hull of protein.

        Returns
        -------
            hull : array
                Coordinates of convex hull.
        """
        points = []
        # loop over atoms in protein
        if len(self.atoms.keys()) > 0:
            for atom in self.atoms.values():
                points.append(atom.coords())
            hull = ConvexHull(points)
        else:
            hull = False
            print(
                'ERROR: Convex hull calculation failed. Structure will be inaccurate.\n')
        self.hull = hull
