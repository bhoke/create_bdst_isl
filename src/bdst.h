#ifndef BDST_H
#define BDST_H

#include "Utility.h"
#include <QVector>

class TreeLeaf
{
public:
    int left;
    int right;
    float val;
    int parentConnection;
    bool isused;
};

class BDST
{
public:
    BDST();
    ~BDST();
    QList<Level> levels;
};

#endif
