//GenerateSurfaceMask.h.
#pragma once

#include <any>
#include <functional>
#include <list>


template <class T, class R> class planar_image_collection;
template <class T> class contour_collection;


struct GenerateSurfaceMaskUserData {
    float background_val = 0.0;
    float surface_val    = 1.0;
    float interior_val   = 2.0;

//    bool assume_boundary_is_surface = false; //If the ROI overshoots an image boundary, assume the boundary is the 
//    int64_t voxel_neighbour_family = 1; 
};

bool ComputeGenerateSurfaceMask(planar_image_collection<float,double> &,
                          std::list<std::reference_wrapper<planar_image_collection<float,double>>>,
                          std::list<std::reference_wrapper<contour_collection<double>>> ccsl,
                          std::any ud );

