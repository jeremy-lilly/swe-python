
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.integrate import quadrature

import xarray
import argparse

from msh import load_mesh
from ops import trsk_mats


def ujet(alat, lat0, lat1, uamp, rsph):
    """
    Helper to integrate PSI from U = -1/R * d/d.lat(PSI) via
    quadrature...

    U(lat) = A * exp(1/((lat-lat0) * (lat-lat1)))

    """

    vals = -rsph * uamp * np.exp(
        1.0E+0 / ((alat - lat0) * (alat - lat1)))

    vals[alat < lat0] = 0.0
    vals[alat > lat1] = 0.0

    return vals


def init(name, save, rsph=1.E+0, pert=True):
    """
    INIT: Form SWE initial conditions for the barotropic jet
    case.
    
    Adds initial conditions to the MPAS mesh file NAME.nc,
    with the output IC's written to SAVE.nc.

    If PERT=TRUE, adds a perturbation to the layer thickness 

    """
    # Authors: Darren Engwirda, Sara Calandrini,
    #          Mark Petersen

#------------------------------------ load an MPAS mesh file

    print("Load the mesh file")

    mesh = load_mesh(name, rsph)

#------------------------------------ build TRSK matrix op's

    print("Forming coefficients")

    trsk = trsk_mats(mesh)

#------------------------------------ build a streamfunction

    print("Build streamfunction")

#-- J. Galewsky, R.K. Scott & L.M. Polvani (2004) An initial 
#-- value problem for testing numerical models of the global 
#-- shallow-water equations, Tellus A: Dynamic Meteorology & 
#-- Oceanography, 56:5, 429-440

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    lat0 = np.pi / 7.0          # jet lat width
    lat1 = np.pi / 2.0 - lat0

    umid = 8.000E+01            # jet max speed
    hbar = 1.000E+04            # mean layer hh

    uamp = umid / np.exp(-4. / (lat1 - lat0) ** 2)

#-- build a streamfunction at mesh vertices using quadrature

    vpsi = np.zeros(
        mesh.vert.size, dtype=np.float64)

    for node in range(mesh.vert.size):
        alat = mesh.vert.ylat[node]
        if (alat >= lat0 and alat < lat1):
            vpsi[node], _ = quadrature(
                ujet, lat0, alat, miniter=8,
                args=(lat0, lat1, uamp, mesh.rsph))

    vpsi[mesh.vert.ylat[:] >= lat1] = np.min(vpsi)

#-- form velocity on edges from streamfunction: ensures u is
#-- div-free in a discrete sense. 

#-- this comes from taking div(*) of the momentum equations,
#-- see: H. Weller, J. Thuburn, C.J. Cotter (2012):
#-- Computational Modes and Grid Imprinting on Five Quasi-
#-- Uniform Spherical C-Grids, M.Wea.Rev. 140(8): 2734-2755.

    print("Calc. velocity field")

    unrm = trsk.edge_grad_perp * vpsi * -1.00
    
    uprp = trsk.edge_reco_perp * unrm * -1.00
    
    udiv = trsk.cell_flux_sums * unrm

    print("--> max(abs(unrm)):", np.max(unrm))
    print("--> sum(div(unrm)):", np.sum(udiv))

#-- solve -g * del^2 h = div f * u_perp for layer thickness,
#-- leads to a h which is in discrete balance

    print("Find layer thickness")

    frot = 2.0 * erot * np.sin(mesh.edge.ylat)

    vrhs = trsk.cell_flux_sums * (frot * uprp)
    vrhs = vrhs * -1.00 / grav

    vrhs = vrhs - np.mean(vrhs)     # INT rhs dA must be 0.0
    vrhs = vrhs - np.mean(vrhs)

    hdel = spsolve(trsk.cell_del2_sums, vrhs)

    herr = hbar - hdel
    hdel = hdel + (                 # add offset for mean hh
        np.sum(mesh.cell.area * herr) /
        np.sum(mesh.cell.area * 1.00)
    )

#-- optional: add perturbation to the thickness distribution

    lat2 = np.pi / 4.               # perturbation constants
    lon2 = np.pi / 1.

    hmul = 120.0
    eta1 = 1. / 3.
    eta2 = 1. / 15.

    hadd = (hmul * np.cos(mesh.cell.ylat) *
       np.exp(-((mesh.cell.xlon - lon2) / eta1) ** 2) *
       np.exp(-((lat2 - mesh.cell.ylat) / eta2) ** 2)
    )

    hdel = hdel + float(pert) * hadd

#-- inject mesh with IC.'s and write output MPAS netCDF file  

    print("Output written to:", save)

    vmag = np.sqrt(unrm ** 2 + uprp ** 2)
    vvel = (
        vmag[mesh.vert.edge[:, 0] - 1] +
        vmag[mesh.vert.edge[:, 1] - 1] +
        vmag[mesh.vert.edge[:, 2] - 1]
    ) / 3.00

    init = xarray.open_dataset(name)
    init.attrs.update({"sphere_radius": mesh.rsph})
    init["xCell"] = (("nCells"), mesh.cell.xpos)
    init["yCell"] = (("nCells"), mesh.cell.ypos)
    init["zCell"] = (("nCells"), mesh.cell.zpos)
    init["areaCell"] = (("nCells"), mesh.cell.area)

    init["xEdge"] = (("nEdges"), mesh.edge.xpos)
    init["yEdge"] = (("nEdges"), mesh.edge.ypos)
    init["zEdge"] = (("nEdges"), mesh.edge.zpos)
    init["dvEdge"] = (("nEdges"), mesh.edge.vlen)    
    init["dcEdge"] = (("nEdges"), mesh.edge.clen)

    init["xVertex"] = (("nVertices"), mesh.vert.xpos)
    init["yVertex"] = (("nVertices"), mesh.vert.ypos)
    init["zVertex"] = (("nVertices"), mesh.vert.zpos)
    init["areaTriangle"] = (("nVertices"), mesh.vert.area)
    init["kiteAreasOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.kite)

    init["h"] = (
        ("Time", "nCells", "nVertLevels"),
        np.reshape(hdel, (1, mesh.cell.size, 1)))
    init["u"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(unrm, (1, mesh.edge.size, 1)))

    init["streamfunction"] = (("nVertices"), vpsi)
    init["velocityTotals"] = (("nVertices"), vvel)
    init["vorticity"] = (
        ("nVertices"), 
        (trsk.dual_curl_sums * unrm) / mesh.vert.area)

    init["tracers"] = (
        ("Time", "nCells", "nVertLevels", "nTracers"), 
        np.zeros((1, mesh.cell.size, 1, 1)))

    init["fCell"] = (("nCells"), 
        2.00E+00 * erot * np.sin(mesh.cell.ylat))
    init["fEdge"] = (("nEdges"), 
        2.00E+00 * erot * np.sin(mesh.edge.ylat))
    init["fVertex"] = (("nVertices"), 
        2.00E+00 * erot * np.sin(mesh.vert.ylat))

    print(init)

    init.to_netcdf(save, format="NETCDF3_64BIT_OFFSET")

    return


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, help="Path to user mesh file.")
    
    parser.add_argument(
        "--init-file", dest="init_file", type=str,
        required=True, help="IC's filename to write.")

    parser.add_argument(
        "--with-pert", dest="with_pert", type=bool,
        required=True, help="True to add hh perturb.")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        required=True, help="Value of sphere_radius.")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius,
         pert=args.with_pert)
