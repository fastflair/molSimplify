# Written by JP Janet for HJK Group
# Dpt of Chemical Engineering, MIT



# import standard modules

# molS modules
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Scripts.geometry import (checkcolinear,
                                          distance,
                                          norm,
                                          rotate_around_axis,
                                          rotation_params,
                                          vecangle,
                                          vecdiff)
from molSimplify.Scripts.io import getlicores, lig_load

## FF dependence
##########################################
####### ligand decoration function #######
##########################################


def decorate_ligand(args, ligand_to_decorate, decoration, decoration_index):
    # structgen depends on decoration_manager, and decoration_manager depends on structgen.ffopt
    # Thus, this import needs to be placed here to avoid a circular dependence
    from molSimplify.Scripts.structgen import ffopt
    # INPUT
    #   - args: placeholder for input arguments
    #   - ligand_to_decorate: mol3D ligand
    #   - set ligand_to_decorate.ob_dict to False or empty dictionary first to avoid potential error
    #   - decoration: list of smiles/decorations
    #   - decoration_index: list of ligand atoms to replace
    # OUTPUT
    #   - new_ligand: built ligand
    #   - complex3D: list of all mol3D ligands and core
    #   - emsg: error messages
    #if args.debug:
    #    print  'decorating ligand'
    lig = ligand_to_decorate
    ## reorder to ensure highest atom index
    ## removed first
    sort_order = [i[0] for i in sorted(enumerate(decoration_index), key=lambda x:x[1])]
    sort_order = sort_order[::-1]  # reverse

    decoration_index = [decoration_index[i] for i in sort_order]
    decoration = [decoration[i] for i in sort_order]
    if args.debug:
        print(('decoration_index  is  ' + str(decoration_index)))
    licores = getlicores()
    if not isinstance(lig, mol3D):
        lig, emsg = lig_load(lig, licores)
    else:
        lig.convert2OBMol()
        lig.charge = lig.OBMol.GetTotalCharge()
    lig.convert2mol3D()  # convert to mol3D

    ## create new ligand
    merged_ligand = mol3D()
    merged_ligand.copymol3D(lig)
    for i, dec in enumerate(decoration):
        print(('** decoration number '  + str(i) + ' attaching ' + dec + ' at site '+str(decoration_index[i])
               + '**\n'))
        dec, emsg = lig_load(dec, licores)
        # dec.OBMol.AddHydrogens()
        dec.convert2mol3D() # convert to mol3D
        if args.debug:
            print(i)
            print(decoration_index)

            print((merged_ligand.getAtom(decoration_index[i]).symbol()))
            print((merged_ligand.getAtom(decoration_index[i]).coords()))
            merged_ligand.writexyz('basic.xyz')
        #dec.writexyz('dec' + str(i) + '.xyz')
        Hs = dec.getHsbyIndex(0)
        if len(Hs) > 0 and (not len(dec.cat)):
            dec.deleteatom(Hs[0])
            dec.charge = dec.charge - 1

        #dec.writexyz('dec_noH' + str(i) + '.xyz')
        if len(dec.cat) > 0:
            decind = dec.cat[0]
        else:
            decind = 0
        dec.alignmol(dec.getAtom(decind),merged_ligand.getAtom(decoration_index[i]))
        r1 = dec.getAtom(decind).coords()
        r2 = dec.centermass() # center of mass
        rrot = r1
        decb = mol3D()
        decb.copymol3D(dec)
        ####################################
        # center of mass of local environment (to avoid bad placement of bulky ligands)
        auxmol = mol3D()
        for at in dec.getBondedAtoms(decind):
            auxmol.addAtom(dec.getAtom(at))
        if auxmol.natoms > 0:
            r2 = auxmol.centermass() # overwrite global with local centermass
            ####################################
            # rotate around axis and get both images
            theta,u = rotation_params(merged_ligand.centermass(),r1,r2)
            #print('u = ' + str(u) + ' theta  = ' + str(theta))
            dec = rotate_around_axis(dec,rrot,u,theta)
            if args.debug:
                dec.writexyz('dec_ARA' + str(i) + '.xyz')
            decb = rotate_around_axis(decb,rrot,u,theta-180)
            if args.debug:
                decb.writexyz('dec_ARB' + str(i) + '.xyz')
            d1 = distance(dec.centermass(),merged_ligand.centermass())
            d2 = distance(decb.centermass(),merged_ligand.centermass())
            dec = dec if (d2 < d1) else decb # pick best one
        #####################################
        # check for linear molecule
        auxm = mol3D()
        for at in dec.getBondedAtoms(decind):
            auxm.addAtom(dec.getAtom(at))
        if auxm.natoms > 1:
            r0 = dec.getAtom(decind).coords()
            r1 = auxm.getAtom(0).coords()
            r2 = auxm.getAtom(1).coords()
            if checkcolinear(r1,r0,r2):
                theta,urot = rotation_params(r1,merged_ligand.getAtom(decoration_index[i]).coords(),r2)
                theta = vecangle(vecdiff(r0,merged_ligand.getAtom(decoration_index[i]).coords()),urot)
                dec = rotate_around_axis(dec,r0,urot,theta)

        ## get the default distance between atoms in question
        connection_neighbours = merged_ligand.getAtom(merged_ligand.getBondedAtomsnotH(decoration_index[i])[0])
        new_atom = dec.getAtom(decind)
        target_distance = connection_neighbours.rad + new_atom.rad
        position_to_place = vecdiff(new_atom.coords(),connection_neighbours.coords())
        old_dist= norm(position_to_place)
        missing = (target_distance - old_dist)/2
        dec.translate([missing*position_to_place[j] for j in [0,1,2]])

        r1 = dec.getAtom(decind).coords()
        u = vecdiff(r1,merged_ligand.getAtom(decoration_index[i]).coords())
        dtheta = 2
        optmax = -9999
        totiters = 0
        decb = mol3D()
        decb.copymol3D(dec)
        # check for minimum distance between atoms and center of mass distance
        while totiters < 180:
            #print('totiters '+ str(totiters))
            dec = rotate_around_axis(dec,r1,u,dtheta)
            d0 = dec.mindist(merged_ligand) # try to maximize minimum atoms distance
            d0cm = dec.distance(merged_ligand) # try to maximize center of mass distance
            iteropt = d0cm+d0 # optimization function
            if (iteropt > optmax): # if better conformation, keep
                decb = mol3D()
                decb.copymol3D(dec)
                optmax = iteropt
                #temp = mol3D()
                #temp.copymol3D(merged_ligand)
                #temp.combine(decb)
                #temp.writexyz('opt_iter_'+str(totiters)+'.xyz')
                #print('new max! ' + str(iteropt) )
            totiters += 1
        dec = decb
        if args.debug:
            dec.writexyz('dec_aligned' + str(i) + '.xyz')
            print(('natoms before delete ' + str(merged_ligand.natoms)))
            print(('obmol before delete at  ' + str(decoration_index[i]) + ' is ' + str(merged_ligand.OBMol.NumAtoms())))
        ## store connectivity for deleted H
        BO_mat = merged_ligand.populateBOMatrix()
        row_deleted = BO_mat[decoration_index[i]]
        bonds_to_add = []

        # find where to put the new bonds ->>> Issue here.
        for j,els in enumerate(row_deleted):
            if els > 0:
                # if there is a bond with an atom number
                # before the deleted atom, all is fine
                # else, we subtract one as the row will be be removed
                if j < decoration_index[i]:
                    bond_partner = j
                else:
                    bond_partner = j -1
                if len(dec.cat) > 0:
                    bonds_to_add.append((bond_partner,(merged_ligand.natoms-1)+dec.cat[0],els))
                else:
                    bonds_to_add.append((bond_partner,merged_ligand.natoms-1,els))

        ## perfrom delete
        merged_ligand.deleteatom(decoration_index[i])

        merged_ligand.convert2OBMol()
        if args.debug:
            merged_ligand.writexyz('merged del ' + str(i) + '.xyz')
        ## merge and bond
        merged_ligand.combine(dec,bond_to_add=bonds_to_add)
        merged_ligand.convert2OBMol()

        if args.debug:
            merged_ligand.writexyz('merged' + str(i) + '.xyz')
            merged_ligand.printxyz()
            print('************')

    merged_ligand.convert2OBMol()
    merged_ligand,emsg = ffopt('MMFF94',merged_ligand,[],0,[],False,[],100)
    BO_mat = merged_ligand.populateBOMatrix()
    if args.debug:
        merged_ligand.writexyz('merged_relaxed.xyz')
        print(BO_mat)
    return(merged_ligand)
