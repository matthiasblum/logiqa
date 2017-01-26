# LOGIQA: long-range genome interactions quality assessment

LOGIQA is a tool for assessing the quality of long-range genome interaction assays.

## Installation

LOGIQA requires the following dependencies:

- Python (&ge; 2.7) with the libraries:
    - Cython (&ge; 0.22) - http://cython.org/
    - h5py (&ge; 2.5.0) - https://github.com/h5py/
    - numpy (&ge; 1.6) - http://www.scipy.org/
    - scipy (&ge; 0.15.1) - http://www.scipy.org/
    - matplotlib (&ge; 1.5) - http://matplotlib.org/
- GCC to compile .c files generated by Cython

To install LOGIQA, run:
```
python setup.py install
```

## Usage
```
logiqa [-h] -g GENOME -r RESOLUTION [RESOLUTION ...] [-d DISTANCE] [-q MAPQ] [--verbose] [--asymmetric] [--text] [--plot] FILENAME.tsv[.gz] FILENAME.h5 [FILENAME.h5 ...]

FILENAME.tsv[.gz]
    Input file containing long-range contacts, can be gzip-compressed.
    If "-" is passed, LOGIQA will read from the standard input

FILENAME.h5
    Output HDF5 file(s). One per resolution.

-g GENOME
    Genome assembly of file containing the size of each chromosome.
    Integrated assemblies are: dm3, hg19, and mm9.
    If a file is passed, it must be a tab-separated file with two columns (chromosome name, chromosome size)

-r RESOLUTION
    Resolution(s) (in bp).

-d DISTANCE
    Distance threshold (in bp). Default: 10000.
    Contacts with a distance lower than DISTANCE will not be considered.

-q MAPQ
    Mappinq quality threshold. Default: 0.
    Contacts with a quality lower than MAPQ will not be considered.

--verbose
    Display progress messages

--asymmetric
    Run LOGIQA in legacy mode: do not reorient contacts, which create asymmetric contact maps.

--text
    Output files are in text format instead of HDF5

--plot
    Generate a scatter plot of the input file in the context of the LOGIQA database
```

## Input format

The input file contains a list of genome contacts (one per line). Each line is composed of seven fields, separated by a tab.
```
chr1	240907184	+	chr1	240907207	-	44
chr4	155486931	-	chr4	167000015	+	44
chr6	125524846	-	chr6	126932443	+	37
chr7	147046004	+	chr7	155158113	-	44
chr12	98858268	-	chr14	30669118	+	36
chr14	77101888	-	chr2	62372370	-	39
chr15	27276011	-	chr15	27946497	+	44
chr18	21086612	+	chr18	21086679	-	40
chr19	56427308	+	chr2	29943041	+	44
chr21	20211726	-	chr21	20829843	+	44
```

| Field    | Description |
| -------- |-------------|
|chrom_1   | name of the forward read's chromosome |
|pos_1     | forward read's position |
|strand_1  | forward read's strand (either "+" or "-") |
|chrom_2   | name of the reverse read's chromosome |
|pos_2     | reverse read's position |
|strand_2  | reverse read's strand (either "+" or "-") |
|mapq      | mapping quality of the contact: lowest MAPQ between the forward read's and the reverse's |

## Output format

If the output files are in HDF5 format, they contain:
- basic statistics (number of total/cis/trans/unique contacts)
- counts of retrieved contacts after random sampling
- quality scores (15%, 10%, 5%). NOT IN LOG2!
- cis-contact maps

If they are in textual format, they contain a header (line starts with "#") and the loops with the following format:

| chr | pos1 | pos2 | count | disp90 | disp70 | disp50 |
|-----|------|------|-------|--------|--------|--------|
| chr1 | 32305000 | 32345000 | 9 | 10.000000 | 14.444445 | 5.555555 |
| chr1 | 35290000 | 35290000 | 11 | 10.000000 | 6.363636 | 13.636364 |
| chr1 | 38885000 | 38885000 | 4 | 10.000000 | 5.000000 | 0.000000 |
| chr1 | 42240000 | 42240000 | 24 | 1.666667 | 0.833333 | 8.333333 |
| chr1 | 42240000 | 42255000 | 28 | 0.714286 | 12.142858 | 7.142857 |
| chr1 | 42240000 | 42330000 | 6 | 10.000000 | 3.333333 | 0.000000 |
| chr1 | 42255000 | 42255000 | 60 | 1.666667 | 10.000000 | 1.666667 |
| chr1 | 42255000 | 42260000 | 5 | 10.000000 | 10.000000 | 10.000000 |

## Citation

*LOGIQA: a database dedicated to long-range genome interactions quality assessment.* Mendoza-Parra MA., Blum M., Malysheva V., Cholley PE., Gronemeyer H. BMC Genomics. 2016 May 16;17:355. [doi: 10.1186/s12864-016-2642-1](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-2642-1)
