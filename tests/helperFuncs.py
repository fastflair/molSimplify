import json
import os
import random
import numpy as np
from molSimplify.Scripts.geometry import kabsch, distance
from molSimplify.Scripts.generator import startgen
from molSimplify.Classes.globalvars import (dict_oneempty_check_st,
                                            oneempty_angle_ref)
from molSimplify.Classes.mol3D import mol3D
from pkg_resources import resource_filename, Requirement


def fuzzy_equal(x1, x2, thresh):
    return np.fabs(float(x1) - float(x2)) < thresh


# check whether the string is a integral/float/scientific


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def fuzzy_compare_xyz(xyz1, xyz2, thresh):
    fuzzyEqual = False
    mol1 = mol3D()
    mol1.readfromxyz(xyz1)
    mol2 = mol3D()
    mol2.readfromxyz(xyz2)
    mol1, U, d0, d1 = kabsch(mol1, mol2)
    rmsd12 = mol1.rmsd(mol2)
    print(('rmsd is ' + '{0:.2f}'.format(rmsd12)))
    if rmsd12 < thresh:
        fuzzyEqual = True
    return fuzzyEqual


def getAllLigands(xyz):
    mymol3d = mol3D()
    mymol3d.readfromxyz(xyz)
    # OUTPUT
    #   -mol3D: mol3D of all ligands
    mm = mymol3d.findMetal()[0]
    mbonded = mymol3d.getBondedAtoms(mm)
    ligands = []
    ligAtoms = []
    # Get the 1st atom of one ligand
    for iatom in mbonded:
        if iatom not in ligAtoms:
            lig = [iatom]
            oldlig = []
            while len(lig) > len(oldlig):
                # make a copy of lig
                oldlig = lig[:]
                for i in oldlig:
                    lbonded = mymol3d.getBondedAtoms(i)
                    for j in lbonded:
                        if (j != mm) and (j not in lig):
                            lig.append(j)
            newlig = mol3D()
            for i in lig:
                newlig.addAtom(mymol3d.atoms[i])
                ligAtoms.append(i)
            ligands.append(newlig)
    print("Ligand analysis of xyz file: ", xyz)
    print("There are ", len(ligands), " ligand(s) bonded with metal center\
            ", mm, " in the complex")
    for i in range(0, len(ligands)):
        print("Number of atoms in ligand # ", i, " : ", ligands[i].natoms)
    return ligands


def getMetalLigBondLength(mymol3d):
    # findMetal only returns 1 metal atom?
    # TG: fixed findmetal to return a list
    mm = mymol3d.findMetal()[0]
    bonded = mymol3d.getBondedAtoms(mm)
    blength = []
    for i in bonded:
        blength.append(
            distance(mymol3d.atoms[mm].coords(), mymol3d.atoms[i].coords()))
    return blength


# Compare number of atoms


def compareNumAtoms(xyz1, xyz2):
    print("Checking total number of atoms")
    mol1 = mol3D()
    mol1.readfromxyz(xyz1)
    mol2 = mol3D()
    mol2.readfromxyz(xyz1)
    # Compare number of atoms
    passNumAtoms = (mol1.natoms == mol2.natoms)
    print("Pass total number of atoms check: ", passNumAtoms)
    return passNumAtoms


# Compare Metal Ligand Bond Length


def compareMLBL(xyz1, xyz2, thresh):
    print("Checking metal-ligand bond length")
    mol1 = mol3D()
    mol1.readfromxyz(xyz1)
    mol2 = mol3D()
    mol2.readfromxyz(xyz1)
    bl1 = getMetalLigBondLength(mol1)
    bl2 = getMetalLigBondLength(mol2)
    passMLBL = True
    for i in range(0, len(bl1)):
        if not fuzzy_equal(bl1[i], bl2[i], thresh):
            print("Error! Metal-Ligand bondlength mismatch for bond # ", i)
            passMLBL = False
    print("Pass metal-ligand bond length check: ", passMLBL)
    print("Threshold for bondlength difference: ", thresh)
    return passMLBL


# Compare Ligand Geometry


def compareLG(xyz1, xyz2, thresh):
    print("Checking the Ligand Geometries")
    passLG = True
    ligs1 = getAllLigands(xyz1)
    ligs2 = getAllLigands(xyz2)
    if len(ligs1) != len(ligs2):
        passLG = False
        return passLG
    for i in range(0, len(ligs1)): # Iterate over the ligands
        print("Checking geometry for ligand # ", i)
        ligs1[i], U, d0, d1 = kabsch(ligs1[i], ligs2[i])
        rmsd12 = ligs1[i].rmsd(ligs2[i])
        print(('rmsd is ' + '{0:.2f}'.format(rmsd12)))
        if rmsd12 > thresh:
            passLG = False
            return passLG
    print("Pass ligand geometry check: ", passLG)
    print("Threshold for ligand geometry RMSD difference: ", thresh)
    return passLG


def compareOG(xyz1, xyz2, thresh):
    print("Checking the overall geometry")
    passOG = fuzzy_compare_xyz(xyz1, xyz2, thresh)
    print("Pass overall geometry check: ", passOG)
    print("Threshold for overall geometry check: ", thresh)
    return passOG


def runtest_num_atoms_in_xyz(tmpdir, xyzfile):
    xyz_file1 = mol3D()
    xyz_file1.readfromxyz('tests/refs/' + xyzfile + '.xyz')
    xyz_file1.getNumAtoms()

    with open('tests/refs/' + xyzfile + '.xyz', 'r') as f:
        xyz_file2 = f.readlines()
    num_atoms = int(xyz_file2[0])

    if num_atoms != xyz_file1.getNumAtoms():
        print('Something is wrong with the number of atoms read from the XYZ file!')


def compareGeo(xyz1, xyz2, threshMLBL, threshLG, threshOG, slab=False):
    # Compare number of atoms
    passNumAtoms = compareNumAtoms(xyz1, xyz2)
    # Compare Metal ligand bond length
    if not slab:
        passMLBL = compareMLBL(xyz1, xyz2, threshMLBL)
        # Compare Single ligand geometry
        passLG = compareLG(xyz1, xyz2, threshLG)
    # Compare gross match of overall complex
    passOG = compareOG(xyz1, xyz2, threshOG)
    # FF free test
    # ANN set bond length test
    # covalent radii test
    if not slab:
        return [passNumAtoms, passMLBL, passLG, passOG]
    else:
        return [passNumAtoms, passOG]


def comparedict(ref, gen, thresh):
    passComp = True
    if not set(ref.keys()) <= set(gen.keys()):
        raise KeyError("Keys in the dictionay has been changed")
    for key in ref:
        try:
            valref, valgen = float(ref[key]), float(gen[key])
            if not abs(valref - valgen) < thresh:
                passComp = False
        except ValueError:
            valref, valgen = str(ref[key]), str(gen[key])
            if not valgen == valref:
                passComp = False
    return passComp


def jobname(infile):
    name = os.path.basename(infile)
    name = name.replace(".in", "")
    return name


def jobdir(infile):
    name = jobname(infile)
    # homedir = os.path.expanduser("~")
    homedir = os.getcwd()
    mydir = homedir + '/Runs/' + name
    return mydir


def parse4test(infile, tmpdir, isMulti=False, external={}):
    name = jobname(infile)
    f = tmpdir.join(os.path.basename(infile))
    newname = f.dirname + "/" + os.path.basename(infile)
    print(newname)
    print('&&&&&&&&&')
    with open(infile, 'r') as f_in:
        data = f_in.readlines()
    newdata = ""
    for line in data:
        if line.split()[0] in external.keys():
            newdata += (line.split()[0] + ' ' + str(os.path.dirname(infile))
                        + '/' + str(external[line.split()[0]]) + '\n')
            continue
        if not (("-jobdir" in line) or ("-name" in line)):
            newdata += line
        # Check if we need to parse the dir of smi file
        if ("-lig " in line) and (".smi" in line):
            smi = line.strip('\n').split()[1]
            abs_smi = os.path.dirname(infile) + '/' + smi
            newdata += "-lig " + abs_smi + "\n"
            # fsmi = tmpdir.join(smi)
            # oldsmi=os.path.dirname(infile)+"/"+smi
            # with open(oldsmi) as f:
            #     smidata=f.read()
            # fsmi.write(smidata)
            # print "smi file is copied to the temporary running folder!"
    newdata += "-jobdir " + name + "\n"
    print('=====')
    print(newdata)
    if not isMulti:
        newdata += "-name " + name + "\n"
    print(newdata)
    f.write(newdata)
    print("Input file parsed for test is located: ", newname)
    return newname


def parse4testNoFF(infile, tmpdir):
    name = jobname(infile)
    newname = name + "_noff"
    newinfile = name + "_noff.in"
    f = tmpdir.join(newinfile)
    fullnewname = f.dirname + "/" + newinfile
    with open(infile, 'r') as f_in:
        data = f_in.readlines()
    newdata = ""
    hasFF = False
    for line in data:
        if ("-ff " in line):
            hasFF = True
            break
    if not hasFF:
        print("No FF optimization used in original input file. "
              "No need to do further test.")
        fullnewname = ""
    else:
        print("FF optimization used in original input file. "
              "Now test for no FF result.")
        for line in data:
            # By getting rid of -ffoption, will use the force field option default, which is "N"
            if not (("-jobdir" in line) or ("-name" in line)
                    or ("-ff " in line) or ("-ffoption " in line)):
                newdata += line
        newdata += "-jobdir " + newname + "\n"
        newdata += "-name " + newname + "\n"
        print(newdata)
        f.write(newdata)
        print("Input file parsed for no FF test is located: ", fullnewname)
    return fullnewname


def report_to_dict(lines):
    """
    create a dictionary from comma
    separated files
    """
    d = dict()
    for line in lines:
        key, val = line.strip().split(',')[0:2]
        try:
            d[key] = float(val.strip('[]'))
        except ValueError:
            d[key] = str(val.strip('[]'))
    # extra proc for ANN_bond list:
    if 'ANN_bondl' in d.keys():
        d['ANN_bondl'] = [float(i.strip('[]')) for i in d['ANN_bondl'].split()]
    return (d)


# compare the report, split key and values, do
# fuzzy comparison on the values


def compare_report_new(report1, report2):
    with open(report1, 'r') as f_in:
        data1 = f_in.readlines()
    with open(report2, 'r') as f_in:
        data2 = f_in.readlines()  
    if data1 and data2:
        Equal = True
        dict1 = report_to_dict(data1)
        dict2 = report_to_dict(data2)
    else:
        Equal = False
        print('File not found:')
        if not data1:
            print(('missing: ' + str(report1)))
        if not data2:
            print(('missing: ' + str(report2)))
    if Equal:

        for k in dict1.keys():
            if Equal:
                val1 = dict1[k]
                if k not in dict2.keys():
                    Equal = False
                    print("Report compare failed for ", report1, report2)
                    print("keys " + str(k) + " not present in " + str(report2))
                else:
                    val2 = dict2[k]

                    if not k == "ANN_bondl":
                        # see whether the values are numbers or text
                        if is_number(val1) and is_number(val2):
                            Equal = fuzzy_equal(val1, val2, 1e-4)
                        else:
                            Equal = (val1 == val2)
                        if not Equal:
                            print("Report compare failed for ",
                                  report1, report2)
                            print("Values don't match for key", k)
                            print([val1, val2])
                    else:
                        # loop over ANN bonds?
                        # see whether the values are numbers or text
                        for ii, v in enumerate(val1):
                            Equal = fuzzy_equal(v, val2[ii], 1e-4)
                        if not Equal:
                            print("Report compare failed for ",
                                  report1, report2)
                            print("Values don't match for key", k)
                            print([val1, val2])
            else:
                break
    return Equal


# When generating multiple files from the 1 input file
# Compare the test directory and reference directory for
# Number of xyz file, xyz file names


def checkMultiFileGen(myjobdir, refdir):
    passMultiFileCheck = True
    myfiles = [i for i in os.listdir(myjobdir) if ".xyz" in i]
    reffiles = [i for i in os.listdir(refdir) if ".xyz" in i]
    print("Run directory:", myjobdir)
    print("Generated xyz:", myfiles)
    print("Reference directory:", refdir)
    print("Ref xyz:", reffiles)
    print("Generated ", len(myfiles), " files, expecting ", len(reffiles))
    if len(myfiles) != len(reffiles):
        passMultiFileCheck = False
        print("Error! Numbers don't match!")
    else:
        for ref in reffiles:
            if ref not in myfiles:
                print("xyz file ", ref, " is missing in generated file folder")
                passMultiFileCheck = False
    return [passMultiFileCheck, myfiles]


def compare_qc_input(inp, inp_ref):
    passQcInputCheck = True
    if not os.path.exists(inp_ref):
        return passQcInputCheck
    elif os.path.exists(inp_ref) and (not os.path.exists(inp)):
        passQcInputCheck = False
        print(inp + "not found")
        return passQcInputCheck

    with open(inp, 'r') as f_in:
        data1 = f_in.read()
    with open(inp_ref, 'r') as f_in:
        data_ref = f_in.read()
    if len(data1) != len(data_ref):
        passQcInputCheck = False
        return passQcInputCheck
    for i in range(0, len(data1)):
        if data1[i] != data_ref[i]:
            passQcInputCheck = False
            break
    return passQcInputCheck


def runtest(tmpdir, name, threshMLBL, threshLG, threshOG, seed=None):
    # Set seeds to eliminate randomness from test results
    random.seed(seed)
    np.random.seed(seed)
    infile = resource_filename(Requirement.parse(
        "molSimplify"), "tests/inputs/" + name + ".in")
    newinfile = parse4test(infile, tmpdir)
    args = ['main.py', '-i', newinfile]
    startgen(args, False, False)
    myjobdir = jobdir(infile)
    output_xyz = myjobdir + '/' + name + '.xyz'
    output_report = myjobdir + '/' + name + '.report'
    output_qcin = myjobdir + '/terachem_input'
    with open(newinfile, 'r') as f_in:
        molsim_data = f_in.read()
    if 'orca' in molsim_data.lower():
        # if not '-name' in molsim_data.lower():
        output_qcin = myjobdir + '/orca.in'

    if 'molcas' in molsim_data.lower():
        output_qcin = myjobdir + '/molcas.input'

    ref_xyz = resource_filename(Requirement.parse(
        "molSimplify"), "tests/refs/" + name + ".xyz")
    ref_report = resource_filename(Requirement.parse(
        "molSimplify"), "tests/refs/" + name + ".report")
    ref_qcin = resource_filename(Requirement.parse(
        "molSimplify"), "tests/refs/" + name + ".qcin")

    print("Test input file: ", newinfile)
    print("Test output files are generated in ", myjobdir)
    print("Output xyz file: ", output_xyz)
    pass_xyz = compareGeo(output_xyz, ref_xyz, threshMLBL, threshLG, threshOG)
    [passNumAtoms, passMLBL, passLG, passOG] = pass_xyz
    pass_report = compare_report_new(output_report, ref_report)
    print("Reference xyz file: ", ref_xyz)
    print("Test report file: ", output_report)
    print("Reference report file: ", ref_report)
    print("Reference xyz status: ", pass_xyz)
    print("Reference report status: ", pass_report)
    pass_qcin = compare_qc_input(output_qcin, ref_qcin)
    print("Reference qc input file: ", ref_qcin)
    print("Test qc input file:", output_qcin)
    print("Qc input status:", pass_qcin)
    return [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin]


def runtest_slab(tmpdir, name, threshOG):
    """
    Performs test for slab builder.

    Parameters
    ----------
        tmpdir : str
                tmp folder to run the test
        name : str
                name of the test
        axis : threshOG
                tolerance for RMSD comparison of overall geometries.
    """
    infile = resource_filename(Requirement.parse(
        "molSimplify"), "tests/inputs/" + name + ".in")
    newinfile = parse4test(infile, tmpdir)
    args = ['main.py', '-i', newinfile]
    startgen(args, False, False)
    myjobdir = jobdir(infile) + "/slab/"
    output_xyz = myjobdir + '/super332.xyz'
    ref_xyz = resource_filename(Requirement.parse(
        "molSimplify"), "tests/refs/" + name + ".xyz")
    print("Output xyz file: ", output_xyz)
    pass_xyz = compareGeo(output_xyz, ref_xyz, threshMLBL=0, threshLG=0,
                          threshOG=threshOG, slab=True)
    [passNumAtoms, passOG] = pass_xyz
    return [passNumAtoms, passOG]


def runtest_molecule_on_slab(tmpdir, name, threshOG):
    """
    Performs test for slab builder with a CO molecule adsorbed.

    Parameters
    ----------
        tmpdir : str
                tmp folder to run the test
        name : str
                name of the test
        axis : threshOG
                tolerance for RMSD comparison of overall geometries.
    """
    infile = resource_filename(Requirement.parse(
        "molSimplify"), "tests/inputs/" + name + ".in")
    newinfile = parse4test(infile, tmpdir, external={
        '-unit_cell': 'slab.xyz', '-target_molecule': 'co.xyz'})
    args = ['main.py', '-i', newinfile]
    startgen(args, False, False)
    myjobdir = os.path.split(jobdir(infile))[0] + "/loaded_slab/"
    output_xyz = myjobdir + '/loaded.xyz'
    ref_xyz = resource_filename(Requirement.parse(
        "molSimplify"), "tests/refs/" + name + ".xyz")
    print("Output xyz file: ", output_xyz)
    pass_xyz = compareGeo(output_xyz, ref_xyz, threshMLBL=0, threshLG=0,
                          threshOG=threshOG, slab=True)
    [passNumAtoms, passOG] = pass_xyz
    return [passNumAtoms, passOG]


def runtestgeo(tmpdir, name, thresh, deleteH=True, geo_type="oct"):
    initgeo = resource_filename(Requirement.parse(
        "molSimplify"), "tests/inputs/geocheck/" + name + "/init.xyz")
    optgeo = resource_filename(Requirement.parse(
        "molSimplify"), "tests/inputs/geocheck/" + name + "/opt.xyz")
    refjson = resource_filename(Requirement.parse(
        "molSimplify"), "tests/refs/geocheck/" + name + "/ref.json")
    mymol = mol3D()
    mymol.readfromxyz(optgeo)
    init_mol = mol3D()
    init_mol.readfromxyz(initgeo)
    if geo_type == "oct":
        _, _, dict_struct_info = mymol.IsOct(init_mol=init_mol,
                                             debug=False,
                                             flag_deleteH=deleteH)
    elif geo_type == "one_empty":
        _, _, dict_struct_info = mymol.IsStructure(
            init_mol=init_mol, dict_check=dict_oneempty_check_st,
            angle_ref=oneempty_angle_ref, num_coord=5, debug=False,
            flag_deleteH=deleteH)
    with open(refjson, "r") as fo:
        dict_ref = json.load(fo)
    # passGeo = (sorted(dict_ref.items()) == sorted(dict_struct_info.items()))
    print("ref: ", dict_ref)
    print("now: ", dict_struct_info)
    passGeo = comparedict(dict_ref, dict_struct_info, thresh)
    return passGeo


def runtestgeo_optonly(tmpdir, name, thresh, deleteH=True, geo_type="oct"):
    optgeo = resource_filename(Requirement.parse(
        "molSimplify"), "tests/inputs/geocheck/" + name + "/opt.xyz")
    refjson = resource_filename(Requirement.parse(
        "molSimplify"), "tests/refs/geocheck/" + name + "/ref.json")
    mymol = mol3D()
    mymol.readfromxyz(optgeo)
    if geo_type == "oct":
        _, _, dict_struct_info = mymol.IsOct(debug=False,
                                             flag_deleteH=deleteH)
    with open(refjson, "r") as fo:
        dict_ref = json.load(fo)
    passGeo = comparedict(dict_ref, dict_struct_info, thresh)
    return passGeo


def runtestNoFF(tmpdir, name, threshMLBL, threshLG, threshOG):
    infile = resource_filename(Requirement.parse(
        "molSimplify"), "tests/inputs/" + name + ".in")
    newinfile = parse4testNoFF(infile, tmpdir)
    [passNumAtoms, passMLBL, passLG, passOG, pass_report,
     pass_qcin] = [True, True, True, True, True, True]
    if newinfile != "":
        newname = jobname(newinfile)
        args = ['main.py', '-i', newinfile]
        startgen(args, False, False)
        myjobdir = jobdir(newinfile)
        output_xyz = myjobdir + '/' + newname + '.xyz'
        output_report = myjobdir + '/' + newname + '.report'
        with open(newinfile, 'r') as f_in:
            molsim_data = f_in.read()
        output_qcin = myjobdir + '/terachem_input'
        if 'orca' in molsim_data.lower():
            output_qcin = myjobdir + '/orca.in'
        if 'molcas' in molsim_data.lower():
            output_qcin = myjobdir + '/molcas.input'
        ref_xyz = resource_filename(Requirement.parse(
            "molSimplify"), "tests/refs/" + newname + ".xyz")
        ref_report = resource_filename(Requirement.parse(
            "molSimplify"), "tests/refs/" + newname + ".report")
        ref_qcin = resource_filename(Requirement.parse(
            "molSimplify"), "tests/refs/" + name + ".qcin")
        print("Test input file: ", newinfile)
        print("Test output files are generated in ", myjobdir)
        print("Output xyz file: ", output_xyz)
        pass_xyz = compareGeo(output_xyz, ref_xyz,
                              threshMLBL, threshLG, threshOG)
        [passNumAtoms, passMLBL, passLG, passOG] = pass_xyz
        pass_report = compare_report_new(output_report, ref_report)
        print("Reference xyz file: ", ref_xyz)
        print("Test report file: ", output_report)
        print("Reference report file: ", ref_report)
        print("Reference xyz status: ", pass_xyz)
        print("Reference report status: ", pass_report)
        pass_qcin = compare_qc_input(output_qcin, ref_qcin)
        print("Reference qc input file: ", ref_qcin)
        print("Test qc input file: ", output_qcin)
        print("Qc input status: ", pass_qcin)
    return [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin]


def runtestMulti(tmpdir, name, threshMLBL, threshLG, threshOG):
    infile = resource_filename(Requirement.parse(
        "molSimplify"), "tests/inputs/" + name + ".in")
    newinfile = parse4test(infile, tmpdir, True)
    args = ['main.py', '-i', newinfile]
    # Need to make the ligand file visible to the input file
    startgen(args, False, False)
    myjobdir = jobdir(infile) + "/"
    print("Test input file: ", newinfile)
    print("Test output files are generated in ", myjobdir)
    refdir = resource_filename(Requirement.parse(
        "molSimplify"), "tests/refs/" + name + "/")
    [passMultiFileCheck, myfiles] = checkMultiFileGen(myjobdir, refdir)
    pass_structures = []
    if not passMultiFileCheck:
        print("Test failed for checking number and names of generated files. "
              "Test ends")
    else:
        print("Checking each generated structure...")
        for f in myfiles:
            if ".xyz" in f:
                r = f.replace(".xyz", ".report")
                output_xyz = output_xyz = myjobdir + f
                ref_xyz = refdir + f
                output_report = myjobdir + r
                ref_report = refdir + r
                print("Output xyz file: ", output_xyz)
                print("Reference xyz file: ", ref_xyz)
                print("Test report file: ", output_report)
                print("Reference report file: ", ref_report)
                pass_xyz = compareGeo(
                    output_xyz, ref_xyz, threshMLBL, threshLG, threshOG)
                [passNumAtoms, passMLBL, passLG, passOG] = pass_xyz
                pass_report = compare_report_new(output_report, ref_report)
        pass_structures.append(
            [f, passNumAtoms, passMLBL, passLG, passOG, pass_report])
    return [passMultiFileCheck, pass_structures]
