import CMTypes as Types
import Sim as simulator
import utility as uti
from mpi4py import MPI
from petsc4py import PETSc
import math
import numpy as np
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
my_size = comm.Get_size()

def set_initial(rods,tstart,deltaT,Tf):
    # type: (list,float,float, float) -> None
    uti.root_print('%s', 'allocate & initialing temperature field', my_rank)
    assert isinstance(rods,list)
    Types.PressureVessle.currentTime = tstart
    Types.PressureVessle.timePush(0)
    nowWater, nowPower = Types.PressureVessle.now()
    for rod in rods: # set the initial for blac/gray rod
        assert isinstance(rod,Types.RodUnit)
        if rod.type is Types.RodType.fuel:
            continue
        rod.T  = np.zeros((rod.nH,rod.nR))
        rod.T[:,:] = 373 - 20
        rod.qsource = np.zeros(rod.nH)

    for rod in rods: #set temperature initual for fuel rod
        assert isinstance(rod,Types.RodUnit)
        if rod.type is not Types.RodType.fuel:
            continue
        rodPower = rod.radialPowerFactor * nowPower
        rod.qsource = rod.axialPowerFactor * rodPower
        #print rod.qsource
        Trows = []
        vspace = ( rod.height[-1] - rod.height[0] ) / (rod.height.shape[0]-1)
        #print 'donging rod %s' % str(rod.address)
        for ih in range(0,rod.nH):
            L = rod.height[ih] + 0.1 # TODO L should NOT be ZERO
            q = rod.qsource[ih] / vspace
            h = simulator.calcBoilHeatTransferRate(simulator.calGr(deltaT,L),1,1,L) #assuming deltaT == 10
            Tco = Tf + q / (math.pi * 2 * rod.radious * h)
            Tci = Tco + q / (2 * math.pi * rod.material.lamdaOut) * math.log(rod.radious/rod.inRadious)
            To  = Tci + q / (math.pi * rod.inRadious * 2 *rod.gapHeatRate )
            #print '4 key temp for rod %d-%d-%d: flux: %f, Tco: %f, Tci: %f, To:%f Tf: %f' % (rod.address + (q,Tco,Tci,To,Tf) )
            assert Tco - Tf > 0.0
            tIn = np.linspace(To,To,rod.nRin)
            tOut= np.linspace(Tci,Tco,rod.nR - rod.nRin)
            Trows.append( np.hstack((tIn,tOut)) )
        Trows = tuple(Trows)
        rod.T = np.vstack(Trows)
        assert rod.T.shape == (rod.nH,rod.nR)

    # set other initials
    for rod in rods:
        rod.qbound = np.zeros(rod.nH)
        rod.qup = np.zeros(rod.nR)
        rod.qdown = np.zeros(rod.nR)
        rod.heatCoef = np.zeros(rod.nH)

def get_rank(mask, iass):
    for  rank, ass_arr in mask.items():
        if  iass in ass_arr and rank < my_size:
            return rank
    return -1

def set_mask(rank, rods, mask):
    #type: (int, list) -> None
    rodLocal = filter(lambda rod: rod.address[2] in mask.get(rank), rods)
    uti.mpi_print('rank %d got %d rods' ,(rank, len(rodLocal) ), my_rank)
    bound_ids = {}
    for rod in rodLocal:
        for direct, neigh_rod in rod.neighbour.items():
            if direct.count('+') + direct.count('-') != 1:
                continue
            if neigh_rod is None:
                continue
            iass = neigh_rod.address[2]
            if iass  not in mask.get(rank):
                that_rank = get_rank(mask, iass);
                if that_rank == -1:
                    continue
                if bound_ids.get(that_rank) is None:
                    bound_ids[that_rank] = 1
                else:
                    bound_ids[that_rank] += 1 

    uti.mpi_print('rank %d connect to %s' ,(rank, bound_ids.keys()), my_rank)
    assert len(bound_ids) <= 8 and len(bound_ids) >= 0
    bound_array = {}
    for bid, width in bound_ids.items():
        for rod in rodLocal:
            for direct, neigh_rod in rod.neighbour.items():
                if direct.count('+') + direct.count('-') != 1:
                    continue
                if neigh_rod is None:
                    continue
                if  neigh_rod.address[2]  in mask.get(bid):
                    sort_id = float(neigh_rod.index + rod.index) + float(abs( neigh_rod.index - rod.index) ) * 1/30000
                    interface = bound_array.get(bid)
                    if interface is None:
                        bound_array[bid] = [(sort_id, neigh_rod.index)]
                    else:
                        bound_array[bid].append((sort_id, neigh_rod.index))
    for bid, ids in bound_array.items():
        ids = sorted(ids, key = lambda v:v[0]) 
        ids = map(lambda v: v[1], ids)
        #ids = reduce(lambda x,y : x if y in x else x + [y], [[],] + ids) #delete duplicate
        uti.mpi_print('interface %d -> %d : width %d' ,(rank, bid, len(ids)), my_rank)
        interface_buffer = np.zeros((len(ids), rods[0].nH))
        interface_map = {}
        for i, rod_id in enumerate(ids):
            interface_map[rod_id] = interface_buffer[i,:]
        for rod in rodLocal:
            for neigh_rod in rod.neighbour.values():
                if neigh_rod is None:
                    continue
                if not interface_map.get(neigh_rod.index) is None:
                    neigh_rod.T = interface_map[neigh_rod.index] 
                    assert neigh_rod.T.shape == (neigh_rod.nH,)
                    assert neigh_rod.address[2] not in mask.get(rank)
        bound_array[bid] = interface_buffer
    assert len(bound_array) <= 8 and len(bound_array) >= 0
    #uti.root_print("%s", str(bound_array), rank);
    return rodLocal, bound_array

def initPetscTemplate(rods):
    #type: (list)->Types.PETScWrapper,Types.PETScWrapper
    fuleRodSample = None
    blackRodSample = None
    for rod in rods:
        if rod.type is Types.RodType.fuel:
            fuleRodSample = rod
        else:
            blackRodSample = rod
            break
    assert isinstance(fuleRodSample,Types.RodUnit)
    assert fuleRodSample.type is Types.RodType.fuel
    fueltemp = Types.PETScWrapper(fuleRodSample.nH*fuleRodSample.nR,fuleRodSample.nR,fuleRodSample.nH)
    fueltemp.fillTemplatefuel(fuleRodSample.nRin, fuleRodSample.material.lamdaIn, fuleRodSample.material.lamdaOut, fuleRodSample.inRadious,
                              fuleRodSample.radious, fuleRodSample.gapHeatRate, fuleRodSample.height[1]-fuleRodSample.height[0], fuleRodSample.rgrid)
    blacktemp = None
    if blackRodSample is not None:
        assert isinstance(blackRodSample,Types.RodUnit)
        blacktemp = Types.PETScWrapper(blackRodSample.nH*blackRodSample.nR,blackRodSample.nR,blackRodSample.nH)
        blacktemp.fillTemplateBlack(blackRodSample.material.lamdaIn,blackRodSample.radious,blackRodSample.height[1]-blackRodSample.height[0], blackRodSample.rgrid)

    return fueltemp, blacktemp, PETSc.Vec().createSeq(fuleRodSample.nH*fuleRodSample.nR)

