#ifndef PLACEDIVIDER_H
#define PLACEDIVIDER_H

#include "Utility/Place.h"
#include "bubble/bubbleprocess.h"

extern "C" {
  #include "cluster.h"
}

class PlaceDivider{
public:
  static std::vector<Place> clusterPlace(Place pl);
private:
  PlaceDivider(){};
};
#endif
