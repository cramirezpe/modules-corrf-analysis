#!/usr/bin/env python
"""
    Script to compute power spectrum from a CoLoRe snapshot box.
"""

import argparse
import logging
import sys
from pathlib import Path

from CoLoRe_corrf_analysis.pk_helper import PKComputations

logger = logging.getLogger(__name__)

def getArgs(): # pragma: no cover
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--colore-box", type=Path, required=True, help="CoLoRe box."
    )

    parser.add_argument('--rsd', action='store_true', help="Use Redshift Space Distortions.")

    parser.add_argument('--source', type=int, default=0, help="CoLoRe source to use.")

    parser.add_argument('--n-poles', type=int, nargs='+', default=[0, 2, 4], 
    help="Npoles to compute.")

    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()
    return args


def main(args=None):
    if args is None:
        args = getArgs()
    
    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(name)s:%(funcName)s:%(message)s",
    )

    box = PKComputations(
        box_dir = args.colore_box,
        source = args.source,
        rsd = args.rsd
    )

    for n in args.n_poles:
        _ = box.compute_npole(n=n)
