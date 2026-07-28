from nefertiti.functions.write_pdb import write_pdb
import numpy as np

rings = {
    # nucleic acids
    "T": ["C6", "C5", "C7", "N3", "O2", "O4"],
    "U": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "C": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "G": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "A": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    # proteins
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "ARG": ["CD", "NE", "CZ", "NH1", "NH2"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
}


def get_structure_tensor(conf):
    curr_tensor = np.eye(3)
    niter = 0
    while 1:
        conft = conf.dot(curr_tensor)

        conf0 = conft - conft.mean(axis=0)
        v, s, wt = np.linalg.svd(conf0)
        scalevec = s / np.sqrt(len(conf))
        tensor = wt.T
        if np.linalg.det(tensor) < 0:
            tensor[2] *= -1
        assert np.linalg.det(tensor) > 0.999

        curr_tensor = curr_tensor.dot(tensor)
        assert np.linalg.det(curr_tensor) > 0.999

        if np.abs(tensor - np.eye(3)).sum() < 0.01:
            break
        niter += 1
        if niter > 1000:
            if (np.abs(tensor) - np.eye(3)).sum() < 0.01:
                break

        if niter > 10000:
            print(niter, np.abs(tensor - np.eye(3)).sum(), tensor, curr_tensor)
        if niter > 10010:
            exit(1)

    return curr_tensor


prot_struc = np.load("1m4x.npy")
prot_struc = prot_struc[prot_struc["chain"] == np.unique(prot_struc["chain"])[0]]
for resname in ("PHE", "HIS", "ARG", "TRP", "A", "C"):
    if len(resname) == 1:
        struc = np.load(f"{resname}-ppdb.npy")
    else:
        struc = prot_struc
    resid = struc[struc["resname"] == resname.encode()]["resid"][0]
    atoms = struc[struc["resid"] == resid]
    ring_mask = np.isin(atoms["name"], [a.encode() for a in rings[resname]])
    ring_atoms = atoms[ring_mask]
    ring_coors = np.stack((ring_atoms["x"], ring_atoms["y"], ring_atoms["z"]), axis=1)
    ring_coors -= ring_coors.mean(axis=0)
    tensor = get_structure_tensor(ring_coors)
    refe_coors = ring_coors.dot(tensor)
    print(refe_coors.mean(axis=0))
    ring_atoms["x"], ring_atoms["y"], ring_atoms["z"] = refe_coors.T
    np.save(f"refe-{resname}.npy", refe_coors)
    np.save(f"refe-{resname}-ppdb.npy", ring_atoms)
    pdbtxt = write_pdb(ring_atoms)
    with open(f"refe-{resname}.pdb", "w") as f:
        f.write(pdbtxt)
