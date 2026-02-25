
import time
import numpy as np
from scipy.sparse.linalg import gcrotmk
from scipy.integrate import quad

import xarray
import argparse

from stb import strtobool

from msh import load_mesh, cell_quad, dual_quad
from ops import trsk_mats


def ujet(alat, lat0, lat1, uamp, rsph):
    """
    Helper to integrate PSI from U = -1/R * d/d.lat(PSI) via
    quadrature...

    U(lat) = A * exp(1/((lat-lat0) * (lat-lat1)))

    """

    if alat < lat0 or alat > lat1:
        val = 0
    else:
        val = -rsph * uamp * np.exp(
              1.0E+0 / ((alat - lat0) * (alat - lat1)))

    return val


def init(name, save,
         rsph=6371220.0,
         pert=True,
         umax=80.0,
         hmean=1.e4,
         jet_center=np.pi / 4.,
         jet_width=3. * np.pi / 14.,
         pert_lat=np.pi / 4.,
         pert_lon=np.pi,
         pert_scale=120.,
         pert_lat_scale=1. / 15.,
         pert_lon_scale=1. / 3.):
    """
    INIT: Form SWE initial conditions for the barotropic jet
    case.

    Adds initial conditions to the MPAS mesh file NAME.nc,
    with the output IC's written to SAVE.nc.

    If PERT=TRUE, adds a perturbation to the layer thickness

    """
    # Authors: Darren Engwirda, Sara Calandrini

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    mesh = load_mesh(name, rsph)
    
#------------------------------------ build TRSK matrix op's

    print("Forming coefficients...")

    trsk = trsk_mats(mesh)

#------------------------------------ build a streamfunction

    print("Computing streamfunction...")
    
#-- J. Galewsky, R.K. Scott & L.M. Polvani (2004) An initial
#-- value problem for testing numerical models of the global
#-- shallow-water equations, Tellus A: Dynamic Meteorology &
#-- Oceanography, 56:5, 429-440

#-- this routine returns a scaled version of the Galewsky et
#-- al flow in cases where RSPH != 6371220m, with {u, h, g}
#-- multiplied by RSPH / 6371220m. This preserves (vortical)
#-- dynamics across arbitrarily sized spheres.

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616 * (rsph / 6371220.0)     # gravity

    lat0 = jet_center - jet_width / 2.
    lat1 = jet_center + jet_width / 2.

    umid = umax * (rsph / 6371220.0)   # jet max speed
    hbar = hmean * (rsph / 6371220.0)   # mean layer hh

    uamp = umid / np.exp(-4. / (lat1 - lat0) ** 2)

#-- build a streamfunction at mesh vertices using quadrature

    vpsi = np.zeros(
        mesh.vert.size, dtype=np.float64)
    cpsi = np.zeros(
        mesh.cell.size, dtype=np.float64)

    for vert in range(mesh.vert.size):
        alat = mesh.vert.ylat[vert]
        if (alat >= lat0 and alat < lat1):
            vpsi[vert], _ = quad(
                ujet, lat0, alat, #miniter=8,
                args=(lat0, lat1, uamp, mesh.rsph))

    vpsi[mesh.vert.ylat[:] >= lat1] = np.min(vpsi)

    print("--> done: vert!")

    for cell in range(mesh.cell.size):
        alat = mesh.cell.ylat[cell]
        if (alat >= lat0 and alat < lat1):
            cpsi[cell], _ = quad(
                ujet, lat0, alat, #miniter=8,
                args=(lat0, lat1, uamp, mesh.rsph))

    cpsi[mesh.cell.ylat[:] >= lat1] = np.min(cpsi)

    print("--> done: cell!")

#-- form velocity on edges from streamfunction: ensures u is
#-- div-free in a discrete sense.

#-- this comes from taking div(*) of the momentum equations,
#-- see: H. Weller, J. Thuburn, C.J. Cotter (2012):
#-- Computational Modes and Grid Imprinting on Five Quasi-
#-- Uniform Spherical C-Grids, M.Wea.Rev. 140(8): 2734-2755.

    print("Computing velocity field...")

    unrm = trsk.edge_grad_perp * vpsi * -1.00

    uprp = trsk.edge_grad_norm * cpsi * -1.00

    udiv = trsk.cell_flux_sums * unrm

    print("--> max(abs(unrm)):", np.max(unrm))
    print("--> sum(div(unrm)):", np.sum(udiv))

#-- solve -g * del^2 h = div f * u_perp for layer thickness,
#-- leads to a h which is in discrete balance

    print("Computing flow thickness...")

    frot = 2.0 * erot * np.sin(mesh.edge.ylat)

    vrhs = trsk.cell_flux_sums * (frot * uprp)
    vrhs = vrhs * -1.00 / grav

    vrhs = vrhs - np.mean(vrhs)     # INT rhs dA must be 0.0
    vrhs = vrhs - np.mean(vrhs)

    ttic = time.time()
    hdel, info = gcrotmk(
        trsk.cell_del2_sums, vrhs, 
            rtol=1.E-8, atol=1.E-8, m=50, k=25)
    ttoc = time.time()
   #print(ttoc - ttic)    
    
    if (info != +0): raise Exception("Did not converge!")
    
    herr = hbar - hdel
    hdel = hdel + (                 # add offset for mean hh
        np.sum(mesh.cell.area * herr) /
        np.sum(mesh.cell.area * 1.00)
    )

#-- optional: add perturbation to the thickness distribution

    hmul = pert_scale * (rsph / 6371220.0)
    eta1 = pert_lon_scale
    eta2 = pert_lat_scale

    hadd = (hmul * np.cos(mesh.cell.ylat) *
       np.exp(-((mesh.cell.xlon - pert_lon) / eta1) ** 2) *
       np.exp(-((pert_lat - mesh.cell.ylat) / eta2) ** 2)
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
    init.attrs.update({"config_gravity": grav})
    init["xCell"] = (("nCells"), mesh.cell.xpos)
    init["yCell"] = (("nCells"), mesh.cell.ypos)
    init["zCell"] = (("nCells"), mesh.cell.zpos)
    init["areaCell"] = (("nCells"), mesh.cell.area)

    init["xEdge"] = (("nEdges"), mesh.edge.xpos)
    init["yEdge"] = (("nEdges"), mesh.edge.ypos)
    init["zEdge"] = (("nEdges"), mesh.edge.zpos)
    init["dvEdge"] = (("nEdges"), mesh.edge.vlen)
    init["dcEdge"] = (("nEdges"), mesh.edge.clen)
    if hasattr(mesh.edge, 'isrt'):
        init["edgeSortedInds"] = (("nEdges"), mesh.edge.isrt)

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

    init.to_netcdf(save, format="NETCDF4")

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
        "--with-pert", dest="with_pert",
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, default=True,
        help=("True to add h perturbation. "
              "Default: True")
    )

    parser.add_argument(
        "--radius", dest="radius", type=float,
        required=False, default=6371220.0,
        help=("Value of sphere_radius [m]. "
              "Default: 6,371,220")
    )

    parser.add_argument(
        "--umax", dest="umax", type=float,
        required=False, default=80.0,
        help=("Maximum speed of the jet [m/s]. "
              "Default: 80")
    )

    parser.add_argument(
        "--hmean", dest="hmean", type=float,
        required=False, default=1e4,
        help=("Mean fluid thickness [m]. "
              "Default: 10,000")
    )

    parser.add_argument(
        "--jet-center", dest="jet_center", type=float,
        required=False, default=np.pi / 4.,
        help=("Latitudinal center of jet [deg lat]. "
              "Default: pi/4")
    )

    parser.add_argument(
        "--jet-width", dest="jet_width", type=float,
        required=False, default=3. * np.pi / 14.,
        help=("Latitudinal width of jet [deg lat]. "
              "Default: 3pi/14")
    )
    
    parser.add_argument(
        "--pert-lat", dest="pert_lat", type=float,
        required=False, default=np.pi / 4.,
        help=("Latitude of thickness perturbation [deg lat]. "
              "Default: pi/4")
    )

    parser.add_argument(
        "--pert-lon", dest="pert_lon", type=float,
        required=False, default=np.pi,
        help=("Longitude of thickness perturbation [deg lon]. "
              "Default: pi")
    )
    
    parser.add_argument(
        "--pert-scale", dest="pert_scale", type=float,
        required=False, default=120.,
        help=("Scaling constant for thickness perturbation [m]. "
              "Default: 120")
    )
    
    parser.add_argument(
        "--pert-lat-scale", dest="pert_lat_scale", type=float,
        required=False, default=1. / 15.,
        help=("Latitudinal scaling constant for thickness perturbation [m]. "
              "Default: 1/15")
    )

    parser.add_argument(
        "--pert-lon-scale", dest="pert_lon_scale", type=float,
        required=False, default=1. / 3.,
        help=("Longitudinal scaling constant for thickness perturbation [m]. "
              "Default: 1/3")
    )
    
    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius,
         pert=args.with_pert,
         umax=args.umax,
         hmean=args.hmean,
         jet_center=args.jet_center,
         jet_width=args.jet_width,
         pert_lat=args.pert_lat,
         pert_lon=args.pert_lon,
         pert_scale=args.pert_scale,
         pert_lat_scale=args.pert_lat_scale,
         pert_lon_scale=args.pert_lon_scale)
