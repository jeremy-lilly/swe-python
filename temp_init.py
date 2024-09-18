
import os
import time
import numpy as np
import netCDF4 as nc
import argparse

from stb import strtobool

from msh import load_mesh, load_flow, \
                sort_mesh, sort_flow
from ops import trsk_mats

from _dx import HH_TINY, UU_TINY
from _dx import invariant, diag_vars, tcpu
from _dt import step_eqns


def init_file(name, cnfg, save, mesh, flow):

    data = nc.Dataset(save, "w", format="NETCDF4")
    data.on_a_sphere = "YES"
    data.sphere_radius = mesh.rsph
    data.is_periodic = "NO"
    data.source = "swe-python"

    data.createDimension(
        "Time", cnfg.iteration // cnfg.save_freq + 1)
    data.createDimension(
        "Step", cnfg.iteration // cnfg.stat_freq + 1)

    data.createDimension("TWO", 2)
    data.createDimension("nCells", mesh.cell.size)
    data.createDimension("nEdges", mesh.edge.size)
    data.createDimension("nVertices", mesh.vert.size)
    data.createDimension("nVertLevels", 1)
    data.createDimension("maxEdges", np.max(mesh.cell.topo) * 1)
    data.createDimension("maxEdges2", np.max(mesh.cell.topo) * 2)
    data.createDimension("vertexDegree", 3)

    data.createVariable("lonCell", "f8", ("nCells"))
    data["lonCell"][:] = mesh.cell.xlon
    data.createVariable("latCell", "f8", ("nCells"))
    data["latCell"][:] = mesh.cell.ylat
    data.createVariable("xCell", "f8", ("nCells"))
    data["xCell"][:] = mesh.cell.xpos
    data.createVariable("yCell", "f8", ("nCells"))
    data["yCell"][:] = mesh.cell.ypos
    data.createVariable("zCell", "f8", ("nCells"))
    data["zCell"][:] = mesh.cell.zpos
    data.createVariable("areaCell", "f8", ("nCells"))
    data["areaCell"][:] = mesh.cell.area
    data.createVariable(
        "verticesOnCell", "i4", ("nCells", "maxEdges"))
    data["verticesOnCell"][:, :] = mesh.cell.vert
    data.createVariable(
        "edgesOnCell", "i4", ("nCells", "maxEdges"))
    data["edgesOnCell"][:, :] = mesh.cell.edge
    data.createVariable(
        "cellsOnCell", "i4", ("nCells", "maxEdges"))
    data["cellsOnCell"][:, :] = mesh.cell.cell
    data.createVariable("nEdgesOnCell", "i4", ("nCells"))
    data["nEdgesOnCell"][:] = mesh.cell.topo

    data.createVariable("lonEdge", "f8", ("nEdges"))
    data["lonEdge"][:] = mesh.edge.xlon
    data.createVariable("latEdge", "f8", ("nEdges"))
    data["latEdge"][:] = mesh.edge.ylat
    data.createVariable("xEdge", "f8", ("nEdges"))
    data["xEdge"][:] = mesh.edge.xpos
    data.createVariable("yEdge", "f8", ("nEdges"))
    data["yEdge"][:] = mesh.edge.ypos
    data.createVariable("zEdge", "f8", ("nEdges"))
    data["zEdge"][:] = mesh.edge.zpos
    data.createVariable("dvEdge", "f8", ("nEdges"))
    data["dvEdge"][:] = mesh.edge.vlen
    data.createVariable("dcEdge", "f8", ("nEdges"))
    data["dcEdge"][:] = mesh.edge.clen
    data.createVariable(
        "verticesOnEdge", "i4", ("nEdges", "TWO"))
    data["verticesOnEdge"][:, :] = mesh.edge.vert
    data.createVariable(
        "weightsOnEdge", "f8", ("nEdges", "maxEdges2"))
    data["weightsOnEdge"][:, :] = mesh.edge.wmul
    data.createVariable(
        "cellsOnEdge", "i4", ("nEdges", "TWO"))
    data["cellsOnEdge"][:, :] = mesh.edge.cell
    data.createVariable(
        "edgesOnEdge", "i4", ("nEdges", "maxEdges2"))
    data["edgesOnEdge"][:, :] = mesh.edge.edge
    data.createVariable("nEdgesOnEdge", "i4", ("nEdges"))
    data["nEdgesOnEdge"][:] = mesh.edge.topo

    data.createVariable("lonVertex", "f8", ("nVertices"))
    data["lonVertex"][:] = mesh.vert.xlon
    data.createVariable("latVertex", "f8", ("nVertices"))
    data["latVertex"][:] = mesh.vert.ylat
    data.createVariable("xVertex", "f8", ("nVertices"))
    data["xVertex"][:] = mesh.vert.xpos
    data.createVariable("yVertex", "f8", ("nVertices"))
    data["yVertex"][:] = mesh.vert.ypos
    data.createVariable("zVertex", "f8", ("nVertices"))
    data["zVertex"][:] = mesh.vert.zpos
    data.createVariable("areaTriangle", "f8", ("nVertices"))
    data["areaTriangle"][:] = mesh.vert.area
    data.createVariable(
        "kiteAreasOnVertex", "f8", ("nVertices", "vertexDegree"))
    data["kiteAreasOnVertex"][:, :] = mesh.vert.kite
    data.createVariable(
        "edgesOnVertex", "i4", ("nVertices", "vertexDegree"))
    data["edgesOnVertex"][:, :] = mesh.vert.edge
    data.createVariable(
        "cellsOnVertex", "i4", ("nVertices", "vertexDegree"))
    data["cellsOnVertex"][:, :] = mesh.vert.cell
   
    data.createVariable("zb_cell", "f8", ("nCells"))
    data["zb_cell"][:] = flow.zb_cell

    data.createVariable("ff_cell", "f8", ("nCells"))
    data["ff_cell"][:] = flow.ff_cell
    data.createVariable("ff_edge", "f8", ("nEdges"))
    data["ff_edge"][:] = flow.ff_edge
    data.createVariable("ff_vert", "f8", ("nVertices"))
    data["ff_vert"][:] = flow.ff_vert

    data.createVariable(
        "u0_edge", "f8", ("nEdges", "nVertLevels"))
    data["u0_edge"].long_name = "Normal velocity initial conditions" 
    data["u0_edge"][:] = flow.uu_edge[-1, :, :]
    data.createVariable(
        "h0_cell", "f8", ("nCells", "nVertLevels"))    
    data["h0_cell"].long_name = "Layer thickness initial conditions"
    data["h0_cell"][:] = flow.hh_cell[-1, :, :]

    data.createVariable("kp_sums", "f8", ("Step"))
    data["kp_sums"].long_name = \
        "Energetics invariant: total KE+PE over time"
    data.createVariable("en_sums", "f8", ("Step"))
    data["en_sums"].long_name = \
        "Rotational invariant: total PV**2 over time"

    data.createVariable(
        "uu_edge", "f8", ("Time", "nEdges", "nVertLevels"))
    data["uu_edge"].long_name = "Normal velocity on edges"    
    data.createVariable(
        "hh_cell", "f8", ("Time", "nCells", "nVertLevels"))    
    data["hh_cell"].long_name = "Layer thickness on cells"

    data.createVariable(
        "zt_cell", "f4", ("Time", "nCells", "nVertLevels"))    
    data["zt_cell"].long_name = "Top surface of layer on cells"

    data.createVariable(
        "du_cell", "f4", ("Time", "nCells", "nVertLevels"))    
    data["du_cell"].long_name = \
        "Divergence of velocity on cells"

    data.createVariable(
        "ke_bias", "f4", ("Time", "nVertices", "nVertLevels"))
    data["ke_bias"].long_name = \
        "Upwind-bias for KE, averaged to duals"
    data.createVariable(
        "pv_bias", "f4", ("Time", "nVertices", "nVertLevels"))
    data["pv_bias"].long_name = \
        "Upwind-bias for PV, averaged to duals"

    data.createVariable(
        "ke_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data["ke_cell"].long_name = "Kinetic energy on cells"
    data.createVariable(
        "pv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["pv_dual"].long_name = "Potential vorticity on duals"
    data.createVariable(
        "rv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["rv_dual"].long_name = "Relative vorticity on duals"
    
    """
    data.createVariable(
        "ke_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["ke_dual"].long_name = "Kinetic energy on duals"
    data.createVariable(
        "pv_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data["pv_cell"].long_name = "Potential vorticity on cells"
    data.createVariable(
        "rv_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data["rv_cell"].long_name = "Relative vorticity on cells"
    
    data.createVariable(
        "ux_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["ux_dual"].long_name = "x-component of velocity"
    data.createVariable(
        "uy_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["uy_dual"].long_name = "y-component of velocity"
    data.createVariable(
        "uz_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["uz_dual"].long_name = "z-component of velocity"
    """

    data.close()
