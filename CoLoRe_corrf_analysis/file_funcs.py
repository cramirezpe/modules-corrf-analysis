from pathlib import Path
from CoLoRe_corrf_analysis.cf_helper import CFComputations

import logging

logger = logging.getLogger(__name__)
from tabulate import tabulate


class FileFuncs:
    @staticmethod
    def get_full_path(
        basedir,
        rsd=True,
        rmin=0.1,
        rmax=200,
        zmin=0.7,
        zmax=0.9,
        nside=2,
        N_bins=41,
        rsd2=None,
    ):
        """Method to get the full path of a auto or cross correlation already existing.

        Args:
            rsd (bool, optional): Whether to use RSD or not. (Default: True).
            rmin (float, optional): Min. separation in the correlation output. (Default: 0.1).
            rmax (float, optional): Max. separation in the correlation output. (Default: 200).
            zmin (float, optional): Min. redshift for the correlation. (Default: 0.7).
            zmax (float, optional): Max. redshift for the correlation. (Default: 0.9).
            nside (float, optional): nside for the separation of the sky in pixels. Used to compute errorbars. (Default: 2).
            N_bins (int, optional):  Number of bins for r. (Default: 41).
            rsd2 (bool, optional): If cross-correlation, use RSD for the second field. (Default: None, cross-correlation search disabled).

        Returns:
            Path to the results for each box (multiple boxes can be combined).
        """
        rsd = "rsd" if rsd else "norsd"
        if rsd2 != None:
            if rsd2:  # pragma: no cover
                rsd += "_rsd"
            else:
                rsd += "_norsd"

        return (
            Path(basedir)
            / f"nside_{nside}"
            / rsd
            / f"{rmin}_{rmax}_{N_bins}"
            / f"{zmin}_{zmax}"
        )

    @staticmethod
    def get_available_count_files(path):
        """Method to get the available count files in a given path.

        Args:
            path (Path): Path to the output_files.

        Returns:
            set of pairs of the form 'DD', 'DR', 'RD', 'RR'
        """
        options = ["DD", "DR", "RD", "RR"]
        available = set()

        for option in options:
            file = path / (option + ".dat")
            
            if file.is_file():
                if file.stat().st_size < 20:
                    jobid = int(file.read_text().splitlines()[0])
                    status = FileFuncs.check_jobid_status(jobid)
                    if status in (
                        "COMPLETED",
                        "RUNNING",
                        "PENDING",
                        "REQUEUED",
                        "SUSPENDED",
                    ):
                        available.add(option)
                else:
                    available.add(option)

        return available

    @staticmethod
    def check_jobid_status(jobid: int) -> str:
        tries = 0
        while tries < 10:
            sbatch_process = run(
                f"sacct -j {jobid} -o State --parsable2 -n",
                shell=True,
                capture_output=True,
            )

            try:
                return sbatch_process.stdout.decode("utf-8").splitlines()[0]
            except:
                logger.info(
                    f"Retrieving status for jobid {jobid} failed. Retrying in 2 seconds..."
                )
                time.sleep(2)

        logger.info(f"Retrieving status failed. Assuming job failed.")
        return "FAILED"       

    @staticmethod
    def copy_counts_file(in_path, out_path, in_counts, out_counts=None):
        import shutil

        """ Method to copy count output files from one path to another in order to save the time it would take to compute it again.
        
        Args:
            in_path (Path or str): input path to copy the files from.
            out_path (Path or str): output path to copy the files to.
            counts (str): Counts that we want to copy. Options: ('DD', 'DR', 'RD', 'RR').

        Returns:
            Writes a file origin_{counts}.txt in out_path with the original path fo the file copied.
        """
        in_path = Path(in_path)
        out_path = Path(out_path)

        if out_counts == None:
            out_counts = in_counts

        if in_counts not in ("DD", "DR", "RD", "RR") or out_counts not in (
            "DD",
            "DR",
            "RD",
            "RR",
        ):  # pragma: no cover
            raise ValueError("Invalid value of count")

        if (out_path / f"{out_counts}.dat").is_file():
            raise ValueError("File already exists", out_path / f"{out_counts}.dat")
        for file in out_path.glob("npole*.dat"):
            if file.is_file():
                raise ValueError(
                    "Computed npole in output path. Aborting copy...",
                    str(file.resolve()),
                )

        if (in_path / f"0_{in_counts}.dat").is_file():  # pragma: no cover
            in_file = in_path / f"0_{in_counts}.dat"
        elif (in_path / f"{in_counts}.dat").is_file():
            in_file = in_path / f"{in_counts}.dat"
        else:  # pragma: no cover
            raise ValueError("Couldn't find input file in", in_path)

        shutil.copy(in_file, out_path / f"{out_counts}.dat")

        info_file = out_path / f"origin_{out_counts}.txt"
        info_file.write_text(str(in_file.resolve()))

        return

    @staticmethod
    def get_available_pixels(path, boxes=None):
        """Method to search pixels with results inside of a given auto or cross correlation path

        Args:
            boxes (array, optional): Array of the boxes we wan to include. (Default: Use all boxes available).

        Returns:
            1D array of Paths, pointing to each of the pixels for which there are correlations computed.
        """
        available_pixels = []

        if boxes is None:
            boxes = [x.name for x in path.iterdir()]

        for box in boxes:
            _boxdir = path / str(box)
            for _subdir in _boxdir.iterdir():
                for count in ("DD", "DR", "RD", "RR"):
                    if (
                        not (_subdir / f"0_{count}.dat").is_file()
                        and not (_subdir / f"{count}.dat").is_file()
                    ):
                        break
                else:
                    available_pixels.append(_subdir)
        return available_pixels

    @classmethod
    def mix_sims(cls, path, boxes=None, pixels=None):
        """Method to create a CFComputations object for each of the available pixels in one or more boxes.

        Args:
            path (Path): Path to the boxes. It can be obtained for auto-correlations using cls.get_full_path.
            boxes (array, optional): Array of the boxes we want to include. (Default: All boxes available).
            pixels (array, optional): Array of pixels we want to include. (Default: All available pixels).

        Returns:
            1D array of CFComputations objects.
        """
        boxes = [x.name for x in path.iterdir()] if boxes is None else boxes

        if pixels is None:
            paths = cls.get_available_pixels(path, boxes=boxes)
        else:
            paths = []
            for box in boxes:
                for pixel in pixels:
                    paths.append(path / str(box) / str(pixel))

        output = []
        for _boxpath in paths:
            output.append(CFComputations(_boxpath))
        return output

    @staticmethod
    def get_available_runs(
        path, sort_keys=None, reverse=None, show_incompleted=False
    ):  # pragma: no cover
        """Method to get current runs for a given set of boxes.

        Args:
            path (Path or str): Path to the set of boxes.
            sort_keys (list, optional): List defining the keys to sort the columns. Options: (nside, rsd, rmin, rmax, N_bins, zmin, zmax, N). (Default: nside, rsd, rmin, rmax, zmin, zmax, N)
            reverse (bool, optional): Whether to reverse the previous list. (Default: False)
            show_incompleted (bool, optional): Show path to sub-boxes that are not completed. (Default: False)

        Returns:
            A tabulate table with the available runs. Can be easily printed using print()
        """
        path = Path(path)
        t_header = ["nside", "rsd", "rmin", "rmax", "N_bins", "zmin", "zmax", "N"]
        t_rows = []
        nsides_path = path.glob("nside*")
        for nside_path in nsides_path:
            nside = nside_path.name[6:]
            for rsd_path in nside_path.iterdir():
                rsd = rsd_path.name
                for range_path in rsd_path.iterdir():
                    rmin = range_path.name.split("_")[0]
                    rmax = range_path.name.split("_")[1]
                    N_bins = range_path.name.split("_")[2]
                    for zbin_path in range_path.iterdir():
                        zmin = zbin_path.name.split("_")[0]
                        zmax = zbin_path.name.split("_")[1]
                        sum_ = 0
                        for box in zbin_path.iterdir():
                            for sim_path in box.iterdir():
                                if (sim_path / "0_DD.dat").is_file():
                                    sum_ += 1
                                elif (sim_path / "DD.dat").is_file():
                                    sum_ += 1
                                elif (
                                    show_incompleted
                                    and (sim_path / "run_corr.sl").is_file()
                                ):
                                    for file in sim_path.glob("*out"):
                                        if file.is_file():
                                            print(sim_path.resolve())
                                            break
                        # print(f'nside: {nside_path.name[6:]}\t{rsd_path.name}\t{range_path.name}\t\t{zbin_path.name}\t\t{sims}')
                        t_rows.append(
                            (nside, rsd, rmin, rmax, N_bins, zmin, zmax, sum_)
                        )

        if sort_keys is not None:
            if reverse is None:
                reverse = [False for i in sort_keys]
            assert len(reverse) == len(sort_keys)

            for key, rev in zip(reversed(sort_keys), reversed(reverse)):
                t_rows.sort(key=lambda x: float(x[t_header.index(key)]), reverse=rev)

        # if sort_keys is not None:
        #     indices = []
        #     for key in sort_keys:
        #         indices.append( t_header.index(key))

        #     t_rows.sort(key=lambda x: [x[i] for i in indices])
        # t_rows.sort(key=lambda x: x[4])
        return tabulate(t_rows, t_header, tablefmt="pretty")
