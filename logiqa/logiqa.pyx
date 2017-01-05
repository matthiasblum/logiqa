import gzip
import io
import json
import logging
import os
import sys

import h5py
import numpy as np
cimport cython
cimport numpy as np
from scipy import sparse

cdef extern from "stdio.h":
    int sscanf(const char *str, const char *format, ...);

cdef extern from "string.h":
    int strcmp(const char *str1, const char *str2)


ctypedef struct contact_count_t:
    unsigned long s90_d15
    unsigned long s90_d10
    unsigned long s90_d5
    unsigned long s70_d15
    unsigned long s70_d10
    unsigned long s70_d5
    unsigned long s50_d15
    unsigned long s50_d10
    unsigned long s50_d5


cdef class HiCTrack:
    cdef:
        # All pairs ({chrom1: {chrom2: array})
        dict pairs

        # Valid (cis-chromosomal, long) pairs ({chrom: array})
        dict pairs_valid

        # Size (bp) of chromosomes
        dict chrom_sizes

        # Contact maps for cis-chromosomal pairs
        dict maps

        # Contact maps for valid pairs
        dict maps_valid

        # Size of contact maps
        dict map_sizes

        # Pointers (keep track of the number of pairs)
        dict ptr

         # Default size for pairs arrays
        unsigned long buffer_size

        # Pairs stats
        unsigned long npairs
        unsigned long npairs_cis
        unsigned long npairs_trans
        unsigned long npairs_valid
        unsigned long npairs_unique

        unsigned int resolution

        # Counts (dRCI < 15%, dRCI < 10%, dRCI < 5%)
        contact_count_t counts

        # if true, do not reorient pairs positions to always have pos1 <= pos2
        asymmetric

    def __init__(self, chrom_sizes, asymmetric=False, buffer_size=100000):
        self.chrom_sizes = chrom_sizes
        self.asymmetric = asymmetric
        self.buffer_size = buffer_size

        self.pairs = {}
        self.pairs_valid = {}
        self.ptr = {}
        self.maps = {}
        self.maps_valid = {}
        self.map_sizes = {}
        self.npairs = 0
        self.npairs_cis = 0
        self.npairs_trans = 0
        self.npairs_unique = 0
        self.npairs_valid = 0
        self.counts.s90_d15 = 0
        self.counts.s90_d10 = 0
        self.counts.s90_d5 = 0
        self.counts.s70_d15 = 0
        self.counts.s70_d10 = 0
        self.counts.s70_d5 = 0
        self.counts.s50_d15 = 0
        self.counts.s50_d10 = 0
        self.counts.s50_d5 = 0
        self.resolution = 0

    cpdef load_pairs(self, str filename, int q=0, int verbose=0):
        cdef:
            str line
            char chrom1[32]
            char chrom2[32]
            char strand1
            char strand2
            int pos1
            int pos2
            int nfields
            short int istrand1
            short int istrand2
            int mapq
            unsigned long i = 0
            unsigned long npairs = 0
            unsigned long npairs_trans = 0
            unsigned long npairs_cis = 0

        if filename == '-':
            fh = sys.stdin
        elif isgzip(filename):
            fh = io.BufferedReader(gzip.open(filename, 'rt'))
        else:
            fh = open(filename, 'rt')

        for line in fh:
            i += 1
            nfields = sscanf(line, "%s\t%d\t%c\t%s\t%d\t%c\t%d", chrom1, &pos1, &strand1, chrom2, &pos2, &strand2, &mapq)

            if verbose and i % 1000000 == 0:
                logging.info('\t{}'.format(i))

            if nfields != 7:
                continue
            elif chrom1 not in self.chrom_sizes:
                continue
            elif chrom2 not in self.chrom_sizes:
                continue
            elif pos1 < 0 or pos1 >= self.chrom_sizes[chrom1]:
                continue
            elif pos2 < 0 or pos2 >= self.chrom_sizes[chrom2]:
                continue
            elif mapq < q:
                continue
            elif strcmp(chrom1, chrom2):
                npairs_trans += 1
            else:
                npairs_cis += 1

            npairs += 1
            istrand1 = strand1 == '+'
            istrand2 = strand2 == '+'
            self.add_pair(chrom1, pos1, istrand1, chrom2, pos2, istrand2)

        fh.close()

        self.npairs = npairs
        self.npairs_cis = npairs_cis
        self.npairs_trans = npairs_trans
        self.finalize()

    cdef add_pair(self, char[32] chrom1, int pos1, short int strand1, char[32] chrom2, int pos2, short int strand2):
        cdef:
            unsigned long ptr
            int i = strcmp(chrom1, chrom2)

        if i > 0:
            chrom1, chrom2 = chrom2, chrom1
        elif i == 0 and pos1 > pos2 and not self.asymmetric:
            pos1, pos2 = pos2, pos1

        if chrom1 not in self.pairs:
            self.pairs[chrom1] = {chrom2: np.empty(self.buffer_size, dtype=[('l', np.int32), ('r', np.int32), ('s1', np.int8), ('s2', np.int8)])}
            self.pairs[chrom1][chrom2][0] = (pos1, pos2, strand1, strand2)
            self.ptr[chrom1] = {chrom2: 1}
        elif chrom2 not in self.pairs[chrom1]:
            self.pairs[chrom1][chrom2] = np.empty(self.buffer_size, dtype=[('l', np.int32), ('r', np.int32), ('s1', np.int8), ('s2', np.int8)])
            self.pairs[chrom1][chrom2][0] = (pos1, pos2, strand1, strand2)
            self.ptr[chrom1][chrom2] = 1
        else:
            ptr = self.ptr[chrom1][chrom2]
            if ptr % self.buffer_size == 0:
                self.expand(self.pairs[chrom1][chrom2])
            self.pairs[chrom1][chrom2][ptr] = (pos1, pos2, strand1, strand2)
            self.ptr[chrom1][chrom2] += 1

    cdef expand(self, np.ndarray arr):
        arr.resize(arr.size + self.buffer_size, refcheck=False)

    cdef finalize(self):
        for chrom1 in self.pairs:
            for chrom2 in self.pairs[chrom1]:
                self.pairs[chrom1][chrom2].resize(self.ptr[chrom1][chrom2], refcheck=False)
                self.pairs[chrom1][chrom2].sort(order=['l', 'r', 's1', 's2'])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef remove_dup(self):
        cdef:
            unsigned long i
            int pos1
            int pos2
            int prev_pos1
            int prev_pos2
            short int strand1
            short int strand2
            short int prev_strand1
            short int prev_strand2
            unsigned long npairs_unique

        for chrom1 in self.pairs:
            for chrom2 in self.pairs[chrom1]:
                arr = self.pairs[chrom1][chrom2]
                """
                is_unique = np.ones(arr.size, dtype=bool)

                for i in np.range(1, arr.size):
                    if arr[i] == arr[i-1]:
                        is_unique[i] = 0

                self.pairs[chrom1][chrom2] = self.pairs[chrom1][chrom2][is_unique]
                self.npairs_unique += self.pairs[chrom1][chrom2].size

                """
                arr_unique = np.empty(arr.size, dtype=[('l', np.int32), ('r', np.int32)])
                prev_pos1 = arr[0]['l']
                prev_pos2 = arr[0]['r']
                prev_strand1 = arr[0]['s1']
                prev_strand2 = arr[0]['s2']
                arr_unique[0] = (prev_pos1, prev_pos2)
                npairs_unique = 1

                for i in np.range(1, arr.size):
                    pos1 = arr[i]['l']
                    pos2 = arr[i]['r']
                    strand1 = arr[i]['s1']
                    strand2 = arr[i]['s2']

                    if pos1 != prev_pos1 or pos2 != prev_pos2 or strand1 != prev_strand1 or strand2 != prev_strand2:
                        # Not a PCR duplicate
                        arr_unique[npairs_unique] = (pos1, pos2)
                        npairs_unique += 1

                    prev_pos1 = pos1
                    prev_pos2 = pos2
                    prev_strand1 = strand1
                    prev_strand2 = strand2

                arr_unique.resize(npairs_unique, refcheck=False)
                self.pairs[chrom1][chrom2] = arr_unique
                self.npairs_unique += npairs_unique


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef remove_trans(self):
        pairs = {}

        for chrom1 in self.pairs:
            for chrom2 in self.pairs[chrom1]:
                if chrom1 == chrom2:
                    pairs[chrom1] = self.pairs[chrom1][chrom2]

        self.pairs = pairs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef remove_close(self, unsigned int mindist):
        for chrom in self.pairs:
            self.pairs_valid[chrom] = self.pairs[chrom][np.absolute(self.pairs[chrom]['l'] - self.pairs[chrom]['r']) >= mindist]
            self.npairs_valid += self.pairs_valid[chrom].size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef bin(self, unsigned int resolution, dict chrom_sizes):
        cdef:
            unsigned long i
            unsigned int midpoint
            unsigned int size

        self.resolution = resolution

        for chrom in self.pairs:
            size = (chrom_sizes[chrom] + resolution - 1) / resolution
            self.map_sizes[chrom] = size

            # Assign cis-chrom pairs to bins
            arr = self.pairs[chrom]
            data = np.ones(arr.size, np.int8)
            self.maps[chrom] = sparse.coo_matrix(
                (data, (arr['l']/resolution, arr['r']/resolution)),
                shape=(size, size), dtype=np.int32
            ).tocsr()

            # Assign cis-chrom, long-range to bins
            arr = self.pairs_valid[chrom]
            data = np.ones(arr.size, np.int8)
            self.maps_valid[chrom] = sparse.coo_matrix(
                (data, (arr['l']/resolution, arr['r']/resolution)),
                shape=(size, size), dtype=np.int32
            ).tocsr()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef sample(self):
        cdef:
            np.ndarray[np.int8_t, ndim=1, cast=True] rs90 = np.zeros(self.npairs, dtype=np.int8)
            np.ndarray[np.int8_t, ndim=1, cast=True] rs70 = np.zeros(self.npairs, dtype=np.int8)
            np.ndarray[np.int8_t, ndim=1, cast=True] rs50 = np.zeros(self.npairs, dtype=np.int8)
            unsigned long x = 0
            unsigned long y = 0

        rs90[:self.npairs*0.9] = 1
        rs70[:self.npairs*0.7] = 1
        rs50[:self.npairs*0.5] = 1
        np.random.shuffle(rs90)
        np.random.shuffle(rs70)
        np.random.shuffle(rs50)

        for chrom in self.maps:
            # Compute dispersions, then count number of bins passing threshold for the contact map of valid pairs
            d90, d70, d50, x = sample_mat(self.maps_valid[chrom].tocoo(), x, rs90, rs70, rs50)
            self.maps_valid[chrom] = None
            count_bins(d90, d70, d50, &self.counts)

            mat = self.maps[chrom].tocoo()
            d90, d70, d50, x = sample_mat(mat, y, rs90, rs70, rs50)
            arr = np.empty(mat.data.size, dtype=[('row', np.int32), ('col', np.int32), ('data', np.int32), ('disp90', np.float32), ('disp70', np.float32), ('disp50', np.float32)])
            arr['row'] = mat.row
            arr['col'] = mat.col
            arr['data'] = mat.data
            arr['disp90'] = d90
            arr['disp70'] = d70
            arr['disp50'] = d50
            self.maps[chrom] = arr

    cpdef write(self, str output, unsigned int resolution, astext=False):
        cdef:
            # Maximum number of contacts (half-matrix if balancing matrix, otherwise full matrix)
            unsigned long max_contacts = 0
            # Number of contacts (non-empty)
            unsigned long contacts = 0
            double score15
            double score10
            double score5

        for chrom in self.maps:
            matsize = self.map_sizes[chrom]
            contacts += self.maps[chrom].size

            if self.asymmetric:
                max_contacts += pow(matsize, 2)
            else:
                max_contacts += (pow(matsize, 2) + matsize) / 2

        score15 = calc_score(self.counts.s90_d15, self.counts.s70_d15, self.counts.s50_d15, max_contacts)
        score10 = calc_score(self.counts.s90_d10, self.counts.s70_d10, self.counts.s50_d10, max_contacts)
        score5 = calc_score(self.counts.s90_d5, self.counts.s70_d5, self.counts.s50_d5, max_contacts)

        if score15 == -1 or score10 == -1 or score5 == -1:
            logging.warning('cannot compute score for a {}bp resolution'.format(self.resolution))

        if astext:
            if output.lower().endswith('.gz'):
                fh = gzip.open(output, 'w')
            else:
                fh = open(output, 'w')

            fh.writelines([
                '# File generated by LOGIQA\n'
                '# PETs: {}\n'.format(self.npairs),
                '# Intra-chromosomal PETs: {}\n'.format(self.npairs_cis),
                '# Inter-chromosomal PETs: {}\n'.format(self.npairs_trans),
                '# Unique PETs: {}\n'.format(self.npairs_unique),
                '# Filtered PETs: {}\n'.format(self.npairs_valid),
                '# Score (5%): {}\n'.format(score5),
                '# Score (10%): {}\n'.format(score10),
                '# Score (15%): {}\n'.format(score15),
                '#\n',
                '# chr\tpos1\tpos2\tcount\tdisp90\tdisp70\tdisp50\n'
            ])

            for chrom in sorted(self.maps):
                arr = self.maps[chrom]
                arr = arr[
                    (arr['disp90'] < 15) &
                    (arr['disp70'] < 15) &
                    (arr['disp50'] < 15)
                ]

                chrom_arr = np.array(([chrom] * arr.size))
                arr2 = np.empty(
                    arr.size,
                    dtype=[
                        ('chrom', chrom_arr.dtype), ('row', np.int32), ('col', np.int32), ('data', np.int32),
                        ('disp90', np.float32),('disp70', np.float32), ('disp50', np.float32)
                    ]
                )

                arr2['chrom'] = chrom_arr
                arr2['row'] = arr['row'] * resolution
                arr2['col'] = arr['col'] * resolution
                arr2['data'] = arr['data']
                arr2['disp90'] = arr['disp90']
                arr2['disp70'] = arr['disp70']
                arr2['disp50'] = arr['disp50']
                np.savetxt(fh, arr2, fmt=['%s', '%u', '%u', '%u', '%.6f', '%.6f', '%.6f'], delimiter='\t')

            fh.close()
        else:
            with h5py.File(output, 'w') as fh:
                # Write contacts
                for chrom in self.maps:
                    dset = fh.create_dataset(chrom, data=self.maps[chrom], compression='gzip')
                    dset.attrs['size'] = self.chrom_sizes[chrom]
                    dset.attrs['matsize'] = self.map_sizes[chrom]

                # Write contact counts (to be able to recompute score)
                dset = fh.create_dataset('contacts', data=np.array([
                    (90, 15, self.counts.s90_d15),
                    (90, 10, self.counts.s90_d10),
                    (90, 5, self.counts.s90_d5),
                    (70, 15, self.counts.s70_d15),
                    (70, 10, self.counts.s70_d10),
                    (70, 5, self.counts.s70_d5),
                    (50, 15, self.counts.s50_d15),
                    (50, 10, self.counts.s50_d10),
                    (50, 5, self.counts.s50_d5),
                ], dtype=np.int32), compression='gzip')
                dset.attrs['max'] = max_contacts
                dset.attrs['num'] = contacts

                # Write stats on pairs
                dset = fh.create_dataset('pairs', data=[])
                dset.attrs['all'] = self.npairs
                dset.attrs['cis'] = self.npairs_cis
                dset.attrs['trans'] = self.npairs_trans
                dset.attrs['unique'] = self.npairs_unique
                dset.attrs['valid'] = self.npairs_valid

                # Write scores
                scores = np.array([(15, score15), (10, score10), (5, score5)], dtype=[('disp', np.int32), ('score', np.float64)])
                dset = fh.create_dataset('scores', data=scores, compression='gzip')

        return self.npairs_valid, score15


cdef double calc_score(unsigned long contacts90, unsigned long contacts70, unsigned long contacts50, unsigned long ncontacts):
    """
    The classic way to compute score (with NGS-QC Generator) is:
        score = log2( denqc50 / simqc )
    with
        simqc = denqc90 / denqc50
    so
        score = log2( denqc50 / (denqc90 / denqc50) )
    which is equivalent to
        score = log2( denqc50^2 / denqc90)

    But, contacts with an intensity of 1 are kept when using this method (s100=1, s90=1, d90=10 < 15% so kept)
    To discard these artefactual contacts, we can use denqc70 instead of denqc90:
        score = log2( denqc50^2 / denqc70 )
    Or combine it with the classic method:
        score = log2( (denqc50^2 / denqc90) * (denqc50^2 / denqc70) )

    Note:
    -----
        We don't return the score in log2
    """
    cdef:
        double denqc90
        double denqc70
        double denqc50
        double score

    try:
        denqc90 = <double>contacts90 / <double>ncontacts
        denqc70 = <double>contacts70 / <double>ncontacts
        denqc50 = <double>contacts50 / <double>ncontacts
        score = (pow(denqc50, 2) / denqc90) * (pow(denqc70, 2) / denqc90)
    except ZeroDivisionError:
        score = -1
    finally:
        return score


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef sample_mat(mat, unsigned long x,
                np.ndarray[np.int8_t, ndim=1, cast=True] rs90,
                np.ndarray[np.int8_t, ndim=1, cast=True] rs70,
                np.ndarray[np.int8_t, ndim=1, cast=True] rs50
                ):
    cdef:
        np.ndarray[np.int32_t, ndim=1] row = mat.row
        np.ndarray[np.int32_t, ndim=1] col = mat.col
        np.ndarray[np.int32_t, ndim=1] data = mat.data
        unsigned int n = data.size
        np.ndarray[np.float32_t, ndim=1] data90 = np.zeros(n, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=1] data70 = np.zeros(n, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=1] data50 = np.zeros(n, dtype=np.float32)
        unsigned int i
        unsigned int j

    for i in np.range(n):
        for j in np.range(data[i]):
            if rs90[x]:
                data90[i] += 1
            if rs70[x]:
                data70[i] += 1
            if rs50[x]:
                data50[i] += 1
            x += 1

    return (
        np.absolute(90 - 100 * data90 / data).astype(np.float32),
        np.absolute(70 - 100 * data70 / data).astype(np.float32),
        np.absolute(50 - 100 * data50 / data).astype(np.float32),
        x
    )


cdef count_bins(np.ndarray[np.float32_t, ndim=1] d90,
                np.ndarray[np.float32_t, ndim=1] d70,
                np.ndarray[np.float32_t, ndim=1] d50,
                contact_count_t *counts
                ):
    cdef:
        unsigned int i = 0
        unsigned int n = d90.size

    for i in np.range(n):
        if d90[i] < 15:
            counts.s90_d15 += 1

            if d70[i] < 15:
                counts.s70_d15 += 1

                if d50[i] < 15:
                    counts.s50_d15 += 1

            if d90[i] < 10:
                counts.s90_d10 += 1

                if d70[i] < 10:
                    counts.s70_d10 += 1

                    if d50[i] < 10:
                        counts.s50_d10 += 1

                if d90[i] < 5:
                    counts.s90_d5 += 1

                    if d70[i] < 5:
                        counts.s70_d5 += 1

                        if d50[i] < 5:
                            counts.s50_d5 += 1


cdef isgzip(filename):
    with gzip.open(filename, 'rb') as fh:
        try:
            fh.read(10)
        except IOError:
            return False
        else:
            return True


def load_assembly(assembly):
    filename = os.path.join(os.path.dirname(__file__), 'chrom_sizes', assembly.lower())

    if os.path.isfile(filename):
        return load_chrom_sizes(filename)
    else:
        return None


def load_chrom_sizes(filename):
    if os.path.isfile(filename):
        pass

    chrom_sizes = {}

    with open(filename, 'rt') as fh:
        for line in fh:
            try:
                chrom, size = line.rstrip().split()
                chrom_sizes[chrom] = int(size)
            except (IndexError, ValueError):
                sys.stderr.write('{}: invalid line ({})'.format(filename, line.rstrip()))
                exit(1)

    return chrom_sizes


def load_db(assembly):
    db = {}
    filename = os.path.join(os.path.dirname(__file__), 'db.json')
    if os.path.isfile(filename):
        with open(filename, 'rt') as fh:
            data = json.load(fh)
        data = data.get(assembly, {})

    return data
