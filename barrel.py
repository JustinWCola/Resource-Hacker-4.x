import numpy as np
import CMTypes as Types
import math
import time
import Sim
from petsc4py import PETSc
from mpi4py import MPI
import utility as uti
import struct

barrel_template =  None
rhs_template = None

petsc_ksp = PETSc.KSP().create()
petsc_ksp.setType(PETSc.KSP.Type.CG)
#pc = petsc_ksp.getPC()
#pc.setType(PETSc.PC.Type.ILU)
#petsc_ksp.setPC(pc)

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
my_size = comm.Get_size()

class Barrel(Types.RodUnit):
    def __init__(self, nh, nr, h, rin, rout):
        super(Barrel, self).__init__(-1, nh, 0, nr,(0., 0.), (0, 0, 0), rin, rout, 0.0)
        self.qinbound = np.zeros(nh)
        self.hinCoef = np.zeros(nh)
        self.qbound = np.zeros(nh)
        self.heatCoef = np.zeros(nh)
        self.qup = np.zeros(nr)
        self.qdown = np.zeros(nr)
        self.T = np.zeros((nh, nr))
        self.height = np.linspace(0., h, nh)
        self.rgrid = np.linspace(rin, rout, nr)

    def getInSurface(self):
        return self.T[:,0]

    def get_in_area(self):
        return (self.height[1] - self.height[0]) * math.pi * 2 * self.inRadious

    def getSummary(self):
        return self.T[:,0].max(), \
       self.T[:,-1].max(),\
       self.qinbound.mean(), \
       self.hinCoef.mean(), \
       self.melted.melt_mass 

def set_barrel_material(bar):
    assert isinstance(bar, Barrel)
    bar.material = Types.MaterialProterty(
            'Barrel',
            25,25,
            7020, 7020,
            835, 835,
            1600, 1600)

def calc_barrel_bound(bar, Tf, nowWater, rod_temp):
    assert isinstance(bar, Barrel)
    assert isinstance(rod_temp, np.ndarray)
    nh = bar.nH
    surface = bar.getInSurface()
    deltaT = Tf - surface
    in_area = bar.get_in_area()
    bar.hinCoef[:] = 0.0
    bar.qinbound[:] = 0.0
    for i, L in enumerate(bar.height):
        # h in 
        h = 0.0
        if bar.height[i] < nowWater:
            h = Sim.calcBoilHeatTransferRate(Sim.calGr(deltaT[i], L + 1.0), 1.75, 1.75, L + 1.0)
            #bar.hinCoef[i] += h
        else:
            h = Sim.calcSteamHeatTransferRate(Sim.calSteamGr(deltaT[i], L - nowWater + 1.0), 1.003, 1.003, L - nowWater + 1.0)
            #bar.hinCoef[i] += h
        #uti.mpi_print('dt %d: %e', (i, rod_temp[i] - surface[i]), my_rank)
        qconv = h * (surface[i] - rod_temp[i] )
        #q in 
        qrad = in_area* \
               Types.Constant.EPSILONG * \
               Types.Constant.SIGMA * \
               Types.Constant.RADIO_ANGLE_AMPLIFIER * \
               (surface[i] ** 4 - rod_temp[i] ** 4) 
        bar.qinbound[i] =  qconv
    #h out
    bar.heatCoef[:] = 0.0
    #q out
    bar.qbound[:] = 0.0

def syncBarrel(bar, boundary_assembly_rank, temps): #only called on barrel
    assert isinstance(boundary_assembly_rank, list)
    assert isinstance(temps, np.ndarray)
    comm.Barrier()
    uti.mpi_print('%s', 'barrrel syncing...', my_rank)
    recv_list = boundary_assembly_rank
    nh = bar.nH
    buf = np.zeros((len(recv_list), nh))
    send_buf = np.zeros((nh))
    send_buf[:] = bar.getInSurface()
    reqs = []
    for i, recv_rank in enumerate(recv_list):
        req = comm.Bsend(send_buf, dest = recv_rank, tag = my_rank)
        req = comm.Irecv(buf[i, :], source = recv_rank, tag = recv_rank)
        reqs.append(req)
    MPI.Request.Waitall(reqs)
    for h in xrange(0, nh):
        temps[h] = buf[:, h].mean()

def calc_barrel_temperature(bar, Tf, dt):
    A = barrel_template.getMat()
    b = rhs_template.duplicate()
    b.zeroEntries()
    xsol = rhs_template.duplicate()
    for j in xrange(0, bar.nH):
        for i in xrange(0, bar.nR):
            row = j * bar.nR + i
            xsol.setValue(row, bar.T[j,i])
    Sim.build_basic_temperature(A, b, dt, Types.Constant.FLUID_TEMP, bar)
    insideArea = bar.get_in_area()
    for j in xrange(0, bar.nH):
        i = 0
        row = j * bar.nR + i
        #uti.mpi_print('b %e, A_coef %e', ((bar.hinCoef[j] * Tf  - bar.qinbound[j]) * insideArea, bar.hinCoef[j] * insideArea), my_rank)
        b.setValue(row, (bar.hinCoef[j] * Tf - bar.qinbound[j]) * insideArea, addv = True)
        A.setValue(row, row, bar.hinCoef[j] * insideArea, addv = True)
    A.assemblyBegin()
    b.assemblyBegin()
    xsol.assemblyBegin()
    A.assemblyEnd()
    b.assemblyEnd()
    xsol.assemblyEnd()
 
    Sim.build_melt_temperature(A, b, bar)
    raw_arr = Sim.petsc_solve(A, b, xsol, petsc_ksp)
    for row,val in enumerate(raw_arr):
        j = row / bar.nR
        i = row % bar.nR
        bar.T[j,i] = val

def barrel_init(nh, nr, h, rin, rout):
    bar = Barrel(nh, nr, h, rin, rout)
    set_barrel_material(bar)
    petsc_warpper = Types.PETScWrapper(nh * nr, nr, nh)
    hspace = bar.height[1] - bar.height[0]
    petsc_warpper.fillTemplateBlack(bar.material.lamdaIn, rout, hspace, bar.rgrid)
    global barrel_template
    global rhs_template
    rhs_template = PETSc.Vec().createSeq(nh * nr)
    barrel_template = petsc_warpper
    bar.T[:] = 403
    return bar

def summaraize_barrel(now, bar):
    ret = bar.getSummary()
    uti.mpi_print('time %e barrel inner T %f, outer T %f, qinbound %e, hCoef %e, mass %e', (now,) + tuple(ret), my_rank)

def barrel_save_restart_file(t, bar):
    title = 'sav/barrel.npy'
    _f = open(title,'wb')
    data = struct.pack('f',t)
    _f.write(data)
    np.save(_f, bar.T)

def barrel_load_restart_file(bar):
    title = 'sav/barrel.npy'
    try:
        _f = open(title,'r')
        data = _f.read(struct.calcsize('f'))
        t, = struct.unpack('f', data)
        bar.T[:,:] = np.load(_f)[:,:]
        return t
    except IOError :
        uti.mpi_print('%s', 'barrel new start', my_rank)
    return 

def barrel_start(bar, boundary_assembly_rank, timeLimit, dt):
    uti.mpi_print('%s', 'barrel started', my_rank)
    uti.mpi_print('%s', '', my_rank)
    buff = np.zeros(9999)
    core_temp = np.zeros(bar.nH)
    MPI.Attach_buffer(buff)
    syncBarrel(bar, boundary_assembly_rank, core_temp)
    Types.PressureVessle.timePush(0.0)
    nowWater, nowPower = Types.PressureVessle.now()
    calc_barrel_bound(bar, 373, nowWater, core_temp)
    calc_barrel_temperature(bar, Types.Constant.FLUID_TEMP, 5.0)
    uti.mpi_print('%s', 'finish calculationg barrel', my_rank)

    step_counter = 0
    summaraize_barrel(0.0, bar)
    restart_time = barrel_load_restart_file(bar)
    syncBarrel(bar, boundary_assembly_rank, core_temp)
    print 'boundary assembly', boundary_assembly_rank
    if restart_time is not None:
        Types.PressureVessle.currentTime = restart_time
        uti.mpi_print('barrel restarting from time %f, end time %f', (Types.PressureVessle.currentTime, timeLimit), my_rank)
    while Types.PressureVessle.currentTime <= timeLimit:
        uti.mpi_print('%s... now %s', ('calcing', time.strftime('%Y-%m-%d %X', time.localtime())), my_rank)
        Types.PressureVessle.timePush(dt)
        nowWater, nowPower = Types.PressureVessle.now()
        uti.mpi_print('barrel solving time %f, water level %f', (Types.PressureVessle.currentTime, nowWater), my_rank)
        uti.mpi_print('barrel: %s', 'calculating heat souce and temp bound', my_rank)

        calc_barrel_bound(bar, 373, nowWater, core_temp)
        calc_barrel_temperature(bar, Types.Constant.FLUID_TEMP, 1.0)
        Sim.set_melt(bar)

        if step_counter % 10 == 0:
            summaraize_barrel(Types.PressureVessle.currentTime, bar)
            syncBarrel(bar, boundary_assembly_rank, core_temp)
        if step_counter % 100 == 0:
            Sim.save_tec([bar])
            Sim.save_tec_2d(bar, Types.PressureVessle.currentTime)
        if step_counter % 500 == 0:
            barrel_save_restart_file(Types.PressureVessle.currentTime, bar)
        step_counter += 1
    uti.root_print('%s', 'simulation done', my_rank)
    MPI.Detach_buffer()
 

