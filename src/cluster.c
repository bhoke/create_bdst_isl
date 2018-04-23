/* The C clustering library.
 * Copyright (C) 2002 Michiel Jan Laurens de Hoon.
 *
 * This library was written at the Laboratory of DNA Information Analysis,
 * Human Genome Center, Institute of Medical Science, University of Tokyo,
 * 4-6-1 Shirokanedai, Minato-ku, Tokyo 108-8639, Japan.
 * Contact: mdehoon 'AT' gsc.riken.jp
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation with or without modifications and for any purpose and
 * without fee is hereby granted, provided that any copyright notices
 * appear in all copies and that both those copyright notices and this
 * permission notice appear in supporting documentation, and that the
 * names of the contributors or copyright holders not be used in
 * advertising or publicity pertaining to distribution of the software
 * without specific prior permission.
 *
 * THE CONTRIBUTORS AND COPYRIGHT HOLDERS OF THIS SOFTWARE DISCLAIM ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT
 * OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOFTWARE.
 *
 */

#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include "cluster.h"
#include <stdio.h>

static const double* sortdata = NULL; /* used in the quicksort algorithm */

/* ---------------------------------------------------------------------- */

static
int compare(const void* a, const void* b)
/* Helper function for sort. Previously, this was a nested function under
 * sort, which is not allowed under ANSI C.
 */
{ const int i1 = *(const int*)a;
    const int i2 = *(const int*)b;
    const double term1 = sortdata[i1];
    const double term2 = sortdata[i2];
    if (term1 < term2) return -1;
    if (term1 > term2) return +1;
    return 0;
}

/* ---------------------------------------------------------------------- */

void sort(int n, const double data[], int index[])
/* Sets up an index table given the data, such that data[index[]] is in
 * increasing order. Sorting is done on the indices; the array data
 * is unchanged.
 */
{ int i;
    sortdata = data;
    for (i = 0; i < n; i++) index[i] = i;
    qsort(index, n, sizeof(int), compare);
}


/* ---------------------------------------------------------------------- */

static void
makedatamask(int nrows, int ncols, double*** pdata)
{
    int i;
    double** data;
    data = malloc(nrows*sizeof(double*));

    for (i = 0; i < nrows; i++) data[i] = malloc(ncols*sizeof(double));
    *pdata = data;
}

/* ---------------------------------------------------------------------- */

static void
freedatamask(int n, double** data, int** mask)
{ int i;
    for (i = 0; i < n; i++)
    { free(mask[i]);
        free(data[i]);
    }
    free(mask);
    free(data);
}

/* ---------------------------------------------------------------------- */

static
double find_closest_pair(int n, double** distmatrix, int* ip, int* jp)
/*
This function searches the distance matrix to find the pair with the shortest
distance between them. The indices of the pair are returned in ip and jp; the
distance itself is returned by the function.

n          (input) int
The number of elements in the distance matrix.

distmatrix (input) double**
A ragged array containing the distance matrix. The number of columns in each
row is one less than the row index.

ip         (output) int*
A pointer to the integer that is to receive the first index of the pair with
the shortest distance.

jp         (output) int*
A pointer to the integer that is to receive the second index of the pair with
the shortest distance.
*/
{ int i, j;
    double temp;
    double distance = distmatrix[1][0];
    *ip = 1;
    *jp = 0;
    for (i = 1; i < n; i++)
    { for (j = 0; j < i; j++)
        { temp = distmatrix[i][j];
            if (temp<distance)
            { distance = temp;
                *ip = i;
                *jp = j;
            }
        }
    }
    return distance;
}

/* ---------------------------------------------------------------------- */

static
double ward_closest(int n, double** distmatrix, int* ip, int* jp,int* count)
/*
This function searches the distance matrix to find the pair with the shortest
distance between them. The indices of the pair are returned in ip and jp; the
distance itself is returned by the function.

n          (input) int
The number of elements in the distance matrix.

distmatrix (input) double**
A ragged array containing the distance matrix. The number of columns in each
row is one less than the row index.

ip         (output) int*
A pointer to the integer that is to receive the first index of the pair with
the shortest distance.

jp         (output) int*
A pointer to the integer that is to receive the second index of the pair with
the shortest distance.
*/
{
    int i, j;
    double temp;
    int nr,ns;
    double distance = DBL_MAX;
    double wardCoeff;
    *ip = 1;
    *jp = 0;
    for (i = 1; i < n; i++)
    {
        for (j = 0; j < i; j++)
        {
            nr = count[j];
            ns = count[i];
            wardCoeff = sqrt( 2.0 * nr * ns / (nr + ns));
            temp = wardCoeff * distmatrix[i][j];
            if (temp < distance)
            {
                distance = temp;
                *ip = i;
                *jp = j;
            }
        }
    }

    return distance;
}

/* ********************************************************************* */

static
double euclid (int n, double** data,int index1, int index2, int transpose)

/*
Purpose
=======

The euclid routine calculates the weighted Euclidean distance between two
rows or columns in a matrix.

Arguments
=========

n      (input) int
The number of elements in a row or column. If transpose==0, then n is the number
of columns; otherwise, n is the number of rows.

data1  (input) double array
The data array containing the first vector.

data2  (input) double array
The data array containing the second vector.

mask1  (input) int array
This array which elements in data1 are missing. If mask1[i][j]==0, then
data1[i][j] is missing.

mask2  (input) int array
This array which elements in data2 are missing. If mask2[i][j]==0, then
data2[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance.

index1     (input) int
Index of the first row or column.

index2     (input) int
Index of the second row or column.

transpose (input) int
If transpose==0, the distance between two rows in the matrix is calculated.
Otherwise, the distance between two columns in the matrix is calculated.

============================================================================
*/
{ double result = 0.;
    int i;
    if (transpose==0) /* Calculate the distance between two rows */
    { for (i = 0; i < n; i++)
        {
            double term = data[index1][i] - data[index2][i];
            result += term*term;
        }
    }
    else
    { for (i = 0; i < n; i++)
        {
            double term = data[i][index1] - data[i][index2];
            result += term*term;
        }
    }
    return sqrt(result);
}

/* ********************************************************************* */

double** distancematrix (int nrows, int ncolumns, double** data, int transpose)
/*
Purpose
=======

The distancematrix routine calculates the distance matrix between genes or
microarrays using their measured gene expression data. Several distance measures
can be used. The routine returns a pointer to a ragged array containing the
distances between the genes. As the distance matrix is symmetric, with zeros on
the diagonal, only the lower triangular half of the distance matrix is saved.
The distancematrix routine allocates space for the distance matrix. If the
parameter transpose is set to a nonzero value, the distances between the columns
(microarrays) are calculated, otherwise distances between the rows (genes) are
calculated.
If sufficient space in memory cannot be allocated to store the distance matrix,
the routine returns a NULL pointer, and all memory allocated so far for the
distance matrix is freed.


Arguments
=========

nrows      (input) int
The number of rows in the gene expression data matrix (i.e., the number of
genes)

ncolumns   (input) int
The number of columns in the gene expression data matrix (i.e., the number of
microarrays)

data       (input) double[nrows][ncolumns]
The array containing the gene expression data.

mask       (input) int[nrows][ncolumns]
This array shows which data values are missing. If mask[i][j]==0, then
data[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance. The length of this vector
is equal to the number of columns if the distances between genes are calculated,
or the number of rows if the distances between microarrays are calculated.

dist       (input) char
Defines which distance measure is used, as given by the table:
dist=='e': Euclidean distance
dist=='b': City-block distance
dist=='c': correlation
dist=='a': absolute value of the correlation
dist=='u': uncentered correlation
dist=='x': absolute uncentered correlation
dist=='s': Spearman's rank correlation
dist=='k': Kendall's tau
For other values of dist, the default (Euclidean distance) is used.

transpose  (input) int
If transpose is equal to zero, the distances between the rows is
calculated. Otherwise, the distances between the columns is calculated.
The former is needed when genes are being clustered; the latter is used
when microarrays are being clustered.

========================================================================
*/
{ /* First determine the size of the distance matrix */
    const int n = (transpose==0) ? nrows : ncolumns;
    const int ndata = (transpose==0) ? ncolumns : nrows;
    int i,j;
    double** matrix;

    /* Set the metric function as indicated by dist */

    if (n < 2) return NULL;

    /* Set up the ragged array */
    matrix = malloc(n*sizeof(double*));
    if(matrix==NULL) return NULL; /* Not enough memory available */
    matrix[0] = NULL;
    /* The zeroth row has zero columns. We allocate it anyway for convenience.*/
    for (i = 1; i < n; i++)
    { matrix[i] = malloc(i*sizeof(double));
        if (matrix[i]==NULL) break; /* Not enough memory available */
    }
    if (i < n) /* break condition encountered */
    { j = i;
        for (i = 1; i < j; i++) free(matrix[i]);
        return NULL;
    }

    /* Calculate the distances and save them in the ragged array */
    for (i = 1; i < n; i++)
        for (j = 0; j < i; j++)
            matrix[i][j]=euclid(ndata,data,i,j,transpose);

    return matrix;
}

/* ******************************************************************** */

void cuttree (int nelements, treeNode* tree, int nclusters, int clusterid[])

/*
Purpose
=======

The cuttree routine takes the output of a hierarchical clustering routine, and
divides the elements in the tree structure into clusters based on the
hierarchical clustering result. The number of clusters is specified by the user.

Arguments
=========

nelements      (input) int
The number of elements that were clustered.

tree           (input) treeNode[nelements-1]
The clustering solution. Each treeNode in the array describes one linking event,
with tree[i].left and tree[i].right representig the elements that were joined.
The original elements are numbered 0..nelements-1, treeNodes are numbered
-1..-(nelements-1).

nclusters      (input) int
The number of clusters to be formed.

clusterid      (output) int[nelements]
The number of the cluster to which each element was assigned. Space for this
array should be allocated before calling the cuttree routine. If a memory
error occured, all elements in clusterid are set to -1.

========================================================================
*/
{ int i, j, k;
    int icluster = 0;
    const int n = nelements-nclusters; /* number of treeNodes to join */
    int* treeNodeid;
    for (i = nelements-2; i >= n; i--)
    { k = tree[i].left;
        if (k>=0)
        { clusterid[k] = icluster;
            icluster++;
        }
        k = tree[i].right;
        if (k>=0)
        { clusterid[k] = icluster;
            icluster++;
        }
    }
    treeNodeid = malloc(n*sizeof(int));
    if(!treeNodeid)
    { for (i = 0; i < nelements; i++) clusterid[i] = -1;
        return;
    }
    for (i = 0; i < n; i++) treeNodeid[i] = -1;
    for (i = n-1; i >= 0; i--)
    { if(treeNodeid[i]<0)
        { j = icluster;
            treeNodeid[i] = j;
            icluster++;
        }
        else j = treeNodeid[i];
        k = tree[i].left;
        if (k<0) treeNodeid[-k-1] = j; else clusterid[k] = j;
        k = tree[i].right;
        if (k<0) treeNodeid[-k-1] = j; else clusterid[k] = j;
    }
    free(treeNodeid);
    return;
}

/* ******************************************************************** */

static
treeNode* pclcluster (int nrows, int ncolumns, double** data,double** distmatrix, int transpose)

/*

Purpose
=======

The pclcluster routine performs clustering using pairwise centroid-linking
on a given set of gene expression data, using the distance metric given by dist.

Arguments
=========

nrows     (input) int
The number of rows in the gene expression data matrix, equal to the number of
genes.

ncolumns  (input) int
The number of columns in the gene expression data matrix, equal to the number of
microarrays.

data       (input) double[nrows][ncolumns]
The array containing the gene expression data.

mask       (input) int[nrows][ncolumns]
This array shows which data values are missing. If
mask[i][j] == 0, then data[i][j] is missing.

weight     (input) double[ncolumns] if transpose==0;
                   double[nrows]    if transpose==1
The weights that are used to calculate the distance. The length of this vector
is ncolumns if genes are being clustered, and nrows if microarrays are being
clustered.

transpose  (input) int
If transpose==0, the rows of the matrix are clustered. Otherwise, columns
of the matrix are clustered.

dist       (input) char
Defines which distance measure is used, as given by the table:
dist=='e': Euclidean distance
dist=='b': City-block distance
dist=='c': correlation
dist=='a': absolute value of the correlation
dist=='u': uncentered correlation
dist=='x': absolute uncentered correlation
dist=='s': Spearman's rank correlation
dist=='k': Kendall's tau
For other values of dist, the default (Euclidean distance) is used.

distmatrix (input) double**
The distance matrix. This matrix is precalculated by the calling routine
treecluster. The pclcluster routine modifies the contents of distmatrix, but
does not deallocate it.

Return value
============

A pointer to a newly allocated array of treeNode structs, describing the
hierarchical clustering solution consisting of nelements-1 treeNodes. Depending on
whether genes (rows) or microarrays (columns) were clustered, nelements is
equal to nrows or ncolumns. See src/cluster.h for a description of the treeNode
structure.
If a memory error occurs, pclcluster returns NULL.
========================================================================
*/
{ int i, j;
    const int nelements = (transpose==0) ? nrows : ncolumns;
    int itreeNode;
    const int ndata = transpose ? nrows : ncolumns;
    const int ntreeNodes = nelements - 1;

    /* Set the metric function as indicated by dist */

    treeNode* result;
    double** newdata;
    int* level;
    int* distid = malloc(nelements*sizeof(int));
    if(!distid) return NULL;
    result = malloc(ntreeNodes*sizeof(treeNode));
    if(!result)
    { free(distid);
        return NULL;
    }

    for (i = 0; i < nelements; i++) distid[i] = i;
    /* To remember which row/column in the distance matrix contains what */

    /* Storage for treeNode data */
    if (transpose)
    { for (i = 0; i < nelements; i++)
            for (j = 0; j < ndata; j++)
                newdata[i][j] = data[j][i];
        data = newdata;
    }
    else
    { for (i = 0; i < nelements; i++)
            memcpy(newdata[i], data[i], ndata*sizeof(double));
        data = newdata;
    }

    for (itreeNode = 0; itreeNode < ntreeNodes; itreeNode++)
    { /* Find the pair with the shortest distance */
        int is = 1;
        int js = 0;
        result[itreeNode].distance = find_closest_pair(nelements-itreeNode, distmatrix, &is, &js);
        result[itreeNode].left = distid[js];
        result[itreeNode].right = distid[is];

        /* Make treeNode js the new treeNode */
        for (i = 0; i < ndata; i++)
        {
            data[js][i] = data[js][i]*level[js] + data[is][i]*level[is];
            level[js] += level[is];
            data[js][i] /= level[js];
        }
        free(data[is]);
        data[is] = data[ntreeNodes-itreeNode];
        level[is] = level[ntreeNodes-itreeNode];

        /* Fix the distances */
        distid[is] = distid[ntreeNodes-itreeNode];
        for (i = 0; i < is; i++)
            distmatrix[is][i] = distmatrix[ntreeNodes-itreeNode][i];
        for (i = is + 1; i < ntreeNodes-itreeNode; i++)
            distmatrix[i][is] = distmatrix[ntreeNodes-itreeNode][i];

        distid[js] = -itreeNode-1;
        for (i = 0; i < js; i++)
            distmatrix[js][i] = euclid(ndata,data,js,i,0);
        for (i = js + 1; i < ntreeNodes-itreeNode; i++)
            distmatrix[i][js] = euclid(ndata,data,js,i,0);
    }

    /* Free temporarily allocated space */
    free(data[0]);
    free(data);
    free(level);
    free(distid);

    return result;
}

/* ******************************************************************** */

static
treeNode* wards (int nrows, int ncolumns, double** data, double** distmatrix, int transpose)

/*

Purpose
=======

The wards routine performs clustering using ward's method
on a given set of gene expression data, using the Euclidean distance.

Arguments
=========

nrows     (input) int
The number of rows in the gene expression data matrix, equal to the number of
genes.

ncolumns  (input) int
The number of columns in the gene expression data matrix, equal to the number of
microarrays.

data       (input) double[nrows][ncolumns]
The array containing the gene expression data.

transpose  (input) int
If transpose==0, the rows of the matrix are clustered. Otherwise, columns
of the matrix are clustered.

distmatrix (input) double**
The distance matrix. This matrix is precalculated by the calling routine
treecluster. The pclcluster routine modifies the contents of distmatrix, but
does not deallocate it.

Return value
============

A pointer to a newly allocated array of treeNode structs, describing the
hierarchical clustering solution consisting of nelements-1 treeNodes. Depending on
whether genes (rows) or microarrays (columns) were clustered, nelements is
equal to nrows or ncolumns. See src/cluster.h for a description of the treeNode
structure.
If a memory error occurs, pclcluster returns NULL.
========================================================================
*/
{
    int i, j;
    const int nelements = (transpose==0) ? nrows : ncolumns;
    int itreeNode;
    const int ndata = transpose ? nrows : ncolumns;
    const int ntreeNodes = nelements - 1;

    treeNode* result;
    double** newdata;
    int * count = malloc(nelements*sizeof(int));
    int* distid = malloc(nelements*sizeof(int));
    if(!distid) return NULL;
    result = malloc(ntreeNodes*sizeof(treeNode));
    if(!result)
    {
        free(distid);
        return NULL;
    }

    makedatamask(nelements, ndata, &newdata);

    for (i = 0; i < nelements; i++)
    {
        distid[i] = i; // To remember which row/column in the distance matrix contains what
        count[i] = 1;
    }

    /* Storage for treeNode data */
    if (transpose)
    { for (i = 0; i < nelements; i++)
            for (j = 0; j < ndata; j++)
                newdata[i][j] = data[j][i];
        data = newdata;
    }
    else
    { for (i = 0; i < nelements; i++)
            memcpy(newdata[i], data[i], ndata*sizeof(double));
        data = newdata;
    }

    for (itreeNode = 0; itreeNode < ntreeNodes; itreeNode++)
    { /* Find the pair with the shortest distance */
        int is = 1;
        int js = 0;
        result[itreeNode].distance = ward_closest(nelements-itreeNode, distmatrix, &is, &js,count);
        result[itreeNode].left = distid[js];
        result[itreeNode].right = distid[is];
        /* Make treeNode js the new treeNode */
        for (i = 0; i < ndata; i++)
        {
            data[js][i] = data[js][i]*count[js] + data[is][i]*count[is];
            data[js][i] /= (count[js] + count[is]);
        }
        count[js] += count[is];
        free(data[is]);
        data[is] = data[ntreeNodes-itreeNode];
        count[is] = count[ntreeNodes - itreeNode];

        /* Fix the distances */
        distid[is] = distid[ntreeNodes-itreeNode];
        for (i = 0; i < is; i++)
            distmatrix[is][i] = distmatrix[ntreeNodes-itreeNode][i];
        for (i = is + 1; i < ntreeNodes-itreeNode; i++)
            distmatrix[i][is] = distmatrix[ntreeNodes-itreeNode][i];

        distid[js] = -itreeNode-1;
        for (i = 0; i < js; i++)
            distmatrix[js][i] = euclid(ndata,data,js,i,0);
        for (i = js + 1; i < ntreeNodes-itreeNode; i++)
            distmatrix[i][js] = euclid(ndata,data,js,i,0);
    }

    /* Free temporarily allocated space */
    free(data[0]);
    free(count);
    free(data);
    free(distid);

    return result;
}

/* ******************************************************************** */

static
int treeNodecompare(const void* a, const void* b)
/* Helper function for qsort. */
{ const treeNode* treeNode1 = (const treeNode*)a;
    const treeNode* treeNode2 = (const treeNode*)b;
    const double term1 = treeNode1->distance;
    const double term2 = treeNode2->distance;
    if (term1 < term2) return -1;
    if (term1 > term2) return +1;
    return 0;
}

/* ---------------------------------------------------------------------- */

static
treeNode* pslcluster (int nrows, int ncolumns, double** data, double** distmatrix, int transpose)

/*

Purpose
=======

The pslcluster routine performs single-linkage hierarchical clustering, using
either the distance matrix directly, if available, or by calculating the
distances from the data array. This implementation is based on the SLINK
algorithm, described in:
Sibson, R. (1973). SLINK: An optimally efficient algorithm for the single-link
cluster method. The Computer Journal, 16(1): 30-34.
The output of this algorithm is identical to conventional single-linkage
hierarchical clustering, but is much more memory-efficient and faster. Hence,
it can be applied to large data sets, for which the conventional single-
linkage algorithm fails due to lack of memory.


Arguments
=========

nrows     (input) int
The number of rows in the gene expression data matrix, equal to the number of
genes.

ncolumns  (input) int
The number of columns in the gene expression data matrix, equal to the number of
microarrays.

data       (input) double[nrows][ncolumns]
The array containing the gene expression data.

mask       (input) int[nrows][ncolumns]
This array shows which data values are missing. If
mask[i][j] == 0, then data[i][j] is missing.

weight (input) double[n]
The weights that are used to calculate the distance. The length of this vector
is ncolumns if genes are being clustered, and nrows if microarrays are being
clustered.

transpose  (input) int
If transpose==0, the rows of the matrix are clustered. Otherwise, columns
of the matrix are clustered.

dist       (input) char
Defines which distance measure is used, as given by the table:
dist=='e': Euclidean distance
dist=='b': City-block distance
dist=='c': correlation
dist=='a': absolute value of the correlation
dist=='u': uncentered correlation
dist=='x': absolute uncentered correlation
dist=='s': Spearman's rank correlation
dist=='k': Kendall's tau
For other values of dist, the default (Euclidean distance) is used.

distmatrix (input) double**
The distance matrix. If the distance matrix is passed by the calling routine
treecluster, it is used by pslcluster to speed up the clustering calculation.
The pslcluster routine does not modify the contents of distmatrix, and does
not deallocate it. If distmatrix is NULL, the pairwise distances are calculated
by the pslcluster routine from the gene expression data (the data and mask
arrays) and stored in temporary arrays. If distmatrix is passed, the original
gene expression data (specified by the data and mask arguments) are not needed
and are therefore ignored.


Return value
============

A pointer to a newly allocated array of treeNode structs, describing the
hierarchical clustering solution consisting of nelements-1 treeNodes. Depending on
whether genes (rows) or microarrays (columns) were clustered, nelements is
equal to nrows or ncolumns. See src/cluster.h for a description of the treeNode
structure.
If a memory error occurs, pslcluster returns NULL.

========================================================================
*/
{ int i, j, k;
    const int nelements = transpose ? ncolumns : nrows;
    const int ntreeNodes = nelements - 1;
    int* vector;
    double* temp;
    int* index;
    treeNode* result;
    temp = malloc(ntreeNodes*sizeof(double));
    if(!temp) return NULL;
    index = malloc(nelements*sizeof(int));
    if(!index)
    { free(temp);
        return NULL;
    }
    vector = malloc(ntreeNodes*sizeof(int));
    if(!vector)
    { free(index);
        free(temp);
        return NULL;
    }
    result = malloc(nelements*sizeof(treeNode));
    if(!result)
    { free(vector);
        free(index);
        free(temp);
        return NULL;
    }

    for (i = 0; i < ntreeNodes; i++) vector[i] = i;

    if(distmatrix)
    { for (i = 0; i < nrows; i++)
        { result[i].distance = DBL_MAX;
            for (j = 0; j < i; j++) temp[j] = distmatrix[i][j];
            for (j = 0; j < i; j++)
            { k = vector[j];
                if (result[j].distance >= temp[j])
                { if (result[j].distance < temp[k]) temp[k] = result[j].distance;
                    result[j].distance = temp[j];
                    vector[j] = i;
                }
                else if (temp[j] < temp[k]) temp[k] = temp[j];
            }
            for (j = 0; j < i; j++)
            {
                if (result[j].distance >= result[vector[j]].distance) vector[j] = i;
            }
        }
    }
    else
    { const int ndata = transpose ? nrows : ncolumns;
        /* Set the metric function as indicated by dist */

        for (i = 0; i < nelements; i++)
        { result[i].distance = DBL_MAX;
            for (j = 0; j < i; j++) temp[j] =
                    euclid(ndata, data, i, j, transpose);
            for (j = 0; j < i; j++)
            { k = vector[j];
                if (result[j].distance >= temp[j])
                { if (result[j].distance < temp[k]) temp[k] = result[j].distance;
                    result[j].distance = temp[j];
                    vector[j] = i;
                }
                else if (temp[j] < temp[k]) temp[k] = temp[j];
            }
            for (j = 0; j < i; j++)
                if (result[j].distance >= result[vector[j]].distance) vector[j] = i;
        }
    }
    free(temp);

    for (i = 0; i < ntreeNodes; i++) result[i].left = i;
    qsort(result, ntreeNodes, sizeof(treeNode), treeNodecompare);

    for (i = 0; i < nelements; i++) index[i] = i;
    for (i = 0; i < ntreeNodes; i++)
    { j = result[i].left;
        k = vector[j];
        result[i].left = index[j];
        result[i].right = index[k];
        index[k] = -i-1;
    }
    free(vector);
    free(index);

    result = realloc(result, ntreeNodes*sizeof(treeNode));

    return result;
}
/* ******************************************************************** */

static treeNode* pmlcluster (int nelements, double** distmatrix)
/*

Purpose
=======

The pmlcluster routine performs clustering using pairwise maximum- (complete-)
linking on the given distance matrix.

Arguments
=========

nelements     (input) int
The number of elements to be clustered.

distmatrix (input) double**
The distance matrix, with nelements rows, each row being filled up to the
diagonal. The elements on the diagonal are not used, as they are assumed to be
zero. The distance matrix will be modified by this routine.

Return value
============

A pointer to a newly allocated array of treeNode structs, describing the
hierarchical clustering solution consisting of nelements-1 treeNodes. Depending on
whether genes (rows) or microarrays (columns) were clustered, nelements is
equal to nrows or ncolumns. See src/cluster.h for a description of the treeNode
structure.
If a memory error occurs, pmlcluster returns NULL.
========================================================================
*/
{ int j;
    int n;
    int* clusterid;
    treeNode* result;

    clusterid = malloc(nelements*sizeof(int));
    if(!clusterid) return NULL;
    result = malloc((nelements-1)*sizeof(treeNode));
    if (!result)
    { free(clusterid);
        return NULL;
    }

    /* Setup a list specifying to which cluster a gene belongs */
    for (j = 0; j < nelements; j++) clusterid[j] = j;

    for (n = nelements; n > 1; n--)
    { int is = 1;
        int js = 0;
        result[nelements-n].distance = find_closest_pair(n, distmatrix, &is, &js);

        /* Fix the distances */
        for (j = 0; j < js; j++)
            distmatrix[js][j] = max(distmatrix[is][j],distmatrix[js][j]);
        for (j = js+1; j < is; j++)
            distmatrix[j][js] = max(distmatrix[is][j],distmatrix[j][js]);
        for (j = is+1; j < n; j++)
            distmatrix[j][js] = max(distmatrix[j][is],distmatrix[j][js]);

        for (j = 0; j < is; j++) distmatrix[is][j] = distmatrix[n-1][j];
        for (j = is+1; j < n-1; j++) distmatrix[j][is] = distmatrix[n-1][j];

        /* Update clusterids */
        result[nelements-n].left = clusterid[is];
        result[nelements-n].right = clusterid[js];
        clusterid[js] = n-nelements-1;
        clusterid[is] = clusterid[n-1];
    }
    free(clusterid);

    return result;
}

/* ******************************************************************* */

static treeNode* palcluster (int nelements, double** distmatrix)
/*
Purpose
=======

The palcluster routine performs clustering using pairwise average
linking on the given distance matrix.

Arguments
=========

nelements     (input) int
The number of elements to be clustered.

distmatrix (input) double**
The distance matrix, with nelements rows, each row being filled up to the
diagonal. The elements on the diagonal are not used, as they are assumed to be
zero. The distance matrix will be modified by this routine.

Return value
============

A pointer to a newly allocated array of treeNode structs, describing the
hierarchical clustering solution consisting of nelements-1 treeNodes. Depending on
whether genes (rows) or microarrays (columns) were clustered, nelements is
equal to nrows or ncolumns. See src/cluster.h for a description of the treeNode
structure.
If a memory error occurs, palcluster returns NULL.
========================================================================
*/
{ int j;
    int n;
    int* clusterid;
    int* number;
    treeNode* result;

    clusterid = malloc(nelements*sizeof(int));
    if(!clusterid) return NULL;
    number = malloc(nelements*sizeof(int));
    if(!number)
    { free(clusterid);
        return NULL;
    }
    result = malloc((nelements-1)*sizeof(treeNode));
    if (!result)
    { free(clusterid);
        free(number);
        return NULL;
    }

    /* Setup a list specifying to which cluster a gene belongs, and keep track
   * of the number of elements in each cluster (needed to calculate the
   * average). */
    for (j = 0; j < nelements; j++)
    { number[j] = 1;
        clusterid[j] = j;
    }

    for (n = nelements; n > 1; n--)
    { int sum;
        int is = 1;
        int js = 0;
        result[nelements-n].distance = find_closest_pair(n, distmatrix, &is, &js);

        /* Save result */
        result[nelements-n].left = clusterid[is];
        result[nelements-n].right = clusterid[js];

        /* Fix the distances */
        sum = number[is] + number[js];
        for (j = 0; j < js; j++)
        { distmatrix[js][j] = distmatrix[is][j]*number[is]
                    + distmatrix[js][j]*number[js];
            distmatrix[js][j] /= sum;
        }
        for (j = js+1; j < is; j++)
        { distmatrix[j][js] = distmatrix[is][j]*number[is]
                    + distmatrix[j][js]*number[js];
            distmatrix[j][js] /= sum;
        }
        for (j = is+1; j < n; j++)
        { distmatrix[j][js] = distmatrix[j][is]*number[is]
                    + distmatrix[j][js]*number[js];
            distmatrix[j][js] /= sum;
        }

        for (j = 0; j < is; j++) distmatrix[is][j] = distmatrix[n-1][j];
        for (j = is+1; j < n-1; j++) distmatrix[j][is] = distmatrix[n-1][j];

        /* Update number of elements in the clusters */
        number[js] = sum;
        number[is] = number[n-1];

        /* Update clusterids */
        clusterid[js] = n-nelements-1;
        clusterid[is] = clusterid[n-1];
    }
    free(clusterid);
    free(number);

    return result;
}

/* ******************************************************************* */

treeNode* treecluster (int nrows, int ncolumns, double** data, int transpose, char method, double **distmatrix)
/*
Purpose
=======

The treecluster routine performs hierarchical clustering using pairwise
single-, maximum-, centroid-, or average-linkage, as defined by method, on a
given set of gene expression data, using the distance metric given by dist.
If successful, the function returns a pointer to a newly allocated Tree struct
containing the hierarchical clustering solution, and NULL if a memory error
occurs. The pointer should be freed by the calling routine to prevent memory
leaks.

Arguments
=========

nrows     (input) int
The number of rows in the data matrix, equal to the number of genes.

ncolumns  (input) int
The number of columns in the data matrix, equal to the number of microarrays.

data       (input) double[nrows][ncolumns]
The array containing the data of the vectors to be clustered.

transpose  (input) int
If transpose==0, the rows of the matrix are clustered. Otherwise, columns
of the matrix are clustered.

method     (input) char
Defines which hierarchical clustering method is used:
method=='s': pairwise single-linkage clustering
method=='m': pairwise maximum- (or complete-) linkage clustering
method=='a': pairwise average-linkage clustering
method=='c': pairwise centroid-linkage clustering
method=='w': ward's method clustering
For the first three, either the distance matrix or the gene expression data is
sufficient to perform the clustering algorithm. For pairwise centroid-linkage
clustering, however, the gene expression data are always needed, even if the
distance matrix itself is available.

distmatrix (input) double**
The distance matrix. If the distance matrix is zero initially, the distance
matrix will be allocated and calculated from the data by treecluster, and
deallocated before treecluster returns. If the distance matrix is passed by the
calling routine, treecluster will modify the contents of the distance matrix as
part of the clustering algorithm, but will not deallocate it. The calling
routine should deallocate the distance matrix after the return from treecluster.

Return value
============

A pointer to a newly allocated array of treeNode structs, describing the
hierarchical clustering solution consisting of nelements-1 treeNodes. Depending on
whether genes (rows) or microarrays (columns) were clustered, nelements is
equal to nrows or ncolumns. See src/cluster.h for a description of the treeNode
structure.
If a memory error occurs, treecluster returns NULL.

========================================================================
*/
{ treeNode* result = NULL;
    const int nelements = (transpose==0) ? nrows : ncolumns;
    const int ldistmatrix = (distmatrix==NULL && method!='s') ? 1 : 0;
    if (nelements < 2) return NULL;
    /* Calculate the distance matrix if the user didn't give it */
    if(ldistmatrix)
        distmatrix = distancematrix(nrows, ncolumns, data, transpose);

    switch(method)
    { case 's':
        result = pslcluster(nrows, ncolumns, data, distmatrix,transpose);
        break;
    case 'm':
        result = pmlcluster(nelements, distmatrix);
        break;
    case 'a':
        result = palcluster(nelements, distmatrix);
        break;
    case 'c':
        result = pclcluster(nrows, ncolumns, data, distmatrix,transpose);
        break;
    case 'w':
        result = wards(nrows, ncolumns, data, distmatrix, transpose);
    }
    /* Deallocate space for distance matrix, if it was allocated by treecluster */
    if(ldistmatrix)
    { int i;
        for (i = 1; i < nelements; i++) free(distmatrix[i]);
        free (distmatrix);
    }
    return result;
}
