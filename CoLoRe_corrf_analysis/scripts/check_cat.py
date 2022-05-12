"""
    Script to check whether cat comes from rsd or norsd
"""

import numpy as np
import fitsio
import argparse
from pathlib import Path
from LyaPlotter.sims import CoLoReSim
from LyaPlotter.file_types import FilesBase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cat",
        type=Path,
        required=True,
        help="Catalog",
    )

    parser.add_argument(
        "--box",
        type=Path,
        required=True,
        help="CoLoRe Path",
    )

    parser.add_argument(
        "--guess-rsd", action="store_true", help="Start by checking if cat is rsd"
    )

    args = parser.parse_args()
    check_cat(args.cat, args.box, args.guess_rsd)


def check_cat(cat, box, first_rsd=False):
    d_cat = FilesBase(cat)
    d_col = CoLoReSim(0, box).get_Sources()

    if first_rsd:
        if (check_rsd(d_cat, d_col)) or (check_norsd(d_cat, d_col)):
            pass
        else:
            raise ValueError("not compatible")
    else:
        if (check_norsd(d_cat, d_col)) or (check_rsd(d_cat, d_col)):
            pass
        else:
            raise ValueError("not compatible")


def check_rsd(dcat, dbox):
    rand_pos = np.random.randint(0, len(dbox.z), size=20)

    if np.all(np.in1d(dbox.z[rand_pos] + dbox.dz_rsd[rand_pos], dcat.z)):
        print("Cat is rsd")
        return True
    else:
        return False


def check_norsd(dcat, dbox):
    rand_pos = np.random.randint(0, len(dbox.z), size=20)

    if np.all(np.in1d(dbox.z[rand_pos], dcat.z)):
        print("Cat is norsd")
        return True
    else:
        return False


if __name__ == "__main__":
    main()
